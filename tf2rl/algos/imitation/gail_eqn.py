"""EQN-GAIL with equation penalties, delayed updates and log-odds rewards."""
import numpy as np
import tensorflow as tf

from tf2rl.algos.imitation.gail import GAIL
from tf2rl.networks.equation import EquationNetwork
from tf2rl.utils.tensorflow import update_target_variables

Discriminator_EQL = EquationNetwork


def equation_arguments(parser):
    """Add equation-network architecture, sparsity, and export options to the parser.

    Args:
        parser (argparse.ArgumentParser): Existing parser extended with EQN architecture and staged-
            sparsity options.

    Returns:
        argparse.ArgumentParser: Supplied parser extended with equation-network and sparsity
            options.
    """
    parser.add_argument('--update-delay', type=int, default=3)
    parser.add_argument('--dropout', default='.5,0.')
    parser.add_argument('-v', '--v', default='2,1')
    parser.add_argument('--exclude', default=None)
    parser.add_argument('--eqn-l1', type=float, default=0.,
                        help='L1 coefficient for all equation weights and biases (opt-in)')
    parser.add_argument('--eqn-stages', default=None,
                        help='Warmup,sparse discriminator update counts, then fixed-mask refitting')
    parser.add_argument('--eqn-prune-threshold', type=float, default=.001)
    parser.add_argument('--export-equation', action='store_true',
                        help='Export inference equations at checkpoints and at the end')
    return parser


class GAIL_EQL(GAIL):
    """GAIL with an equation discriminator and optional staged parameter pruning."""
    equation_discriminator = True

    def __init__(self, state_shape, action_dim, training_stages=None, num_layers=2,
                 lr=0.001, enable_sn=False, name="GAIL_EQL", is_debug=False,
                 update_delay=1, atol=0, atol_starting_step=0, v=(2, 1), mask_rate=0,
                 masking_starting_step=0, lmbda=0, regularizer_starting_step=0,
                 drop_out=None, tau=0., exclude=None, grad_penalty_coeff=0,
                 prune_threshold=.001, **kwargs):
        """Configure the imitation objective, discriminator networks, and optimizer settings.

        Args:
            state_shape (tuple[int, ...]): Shape of one observation, excluding the batch axis.
            action_dim (int): Number of discrete choices or width of a continuous action vector.
            training_stages (sequence[int] or None): Counts of warmup and sparse optimizer updates
                before fixed-mask refitting.
            num_layers (int): Number of equation layers.
            lr (float): Optimizer learning rate.
            enable_sn (bool): Request spectral normalization where the selected implementation supports
                it.
            name (str): Name assigned to the model, operation, or registered component.
            is_debug (bool): Enable additional discriminator/parameter diagnostics where implemented.
            update_delay (int): Run discriminator training only when its supplied iteration is divisible
                by this value.
            atol (float): Absolute weight threshold used by legacy tolerance-based inference copies.
            atol_starting_step (int): Environment-step threshold after which tolerance-based inference
                is selected.
            v (sequence[int] or int): Operator repetition counts, one per equation layer; an individual
                layer takes one count.
            mask_rate (float): Historical random-masking option; the active staged-sparsity
                implementations reject nonzero values.
            masking_starting_step (int): Historical masking start step; unsupported active
                configurations are rejected.
            lmbda (float): L1 regularization coefficient for equation parameters.
            regularizer_starting_step (int): Historical regularizer start step; use training_stages in
                the active EQN implementations.
            drop_out (float or sequence[float] or None): Dropout probability, or one probability per
                equation layer; None disables dropout.
            tau (float): Target-network interpolation coefficient: target <- (1-tau)*target +
                tau*online; 1 copies all weights.
            exclude (sequence or None): Equation operators to exclude; network constructors use one
                exclusion list per layer.
            grad_penalty_coeff (float): Coefficient multiplying the discriminator gradient penalty.
            prune_threshold (float): Prune parameters whose absolute values are below this threshold at
                the refitting boundary.
            **kwargs (dict): Additional keyword arguments forwarded to GAIL construction.

        Returns:
            None: Initializes the instance.

        Raises:
            NotImplementedError: Use training_stages for deterministic sparsity and pruning.
            ValueError: Staged sparsity cannot be combined with legacy tau/atol inference copies.
        """
        if mask_rate or masking_starting_step or regularizer_starting_step:
            raise NotImplementedError("Use training_stages for deterministic sparsity and pruning")
        if (lmbda or training_stages is not None) and (tau or atol):
            raise ValueError("Staged sparsity cannot be combined with legacy tau/atol inference copies")
        self.atol = atol
        self.atol_starting_step = atol_starting_step if atol > 0 else None
        self.tau = tau
        self._equation_config = {'state_shape': state_shape, 'action_dim': action_dim,
                                     'num_layers': num_layers, 'v': v, 'drop_out': drop_out,
                                     'exclude': exclude, 'lmbda': lmbda, 'output_activation': tf.nn.sigmoid,
                                     'training_stages': training_stages, 'prune_threshold': prune_threshold}
        super().__init__(state_shape, action_dim, lr=lr, enable_sn=enable_sn, name=name,
                         is_debug=is_debug, update_delay=update_delay,
                         grad_penalty_coeff=grad_penalty_coeff, **kwargs)
        if self.atol_starting_step is not None or tau > 0:
            self.disc_target = EquationNetwork(name="Discriminator_target", **self._equation_config)
            if tau > 0:
                update_target_variables(self.disc_target.weights, self.disc.weights, tau=1.)

    def _make_discriminator(self, *args):
        """Build the discriminator used to distinguish expert and agent samples.

        Args:
            *args (tuple): Positional arguments forwarded to the wrapped operation.

        Returns:
            EquationNetwork: Equation discriminator constructed from the saved equation configuration.
        """
        return EquationNetwork(**self._equation_config)

    @property
    def equation_model(self):
        """Return the equation network used for reward inference and export.

        Returns:
            EquationNetwork: Reward-producing online model, or its target copy when tau is positive.
        """
        return self.disc_target if self.tau > 0 else self.disc

    def _before_update(self):
        """Prepare the equation sparsity stage before the discriminator update.

        Returns:
            None.
        """
        self.disc.prepare_update()

    def train(self, *args, **kwargs):
        """Update the imitation discriminator from agent and expert minibatches and report
        diagnostics.

        Args:
            *args (tuple): Agent and expert minibatches forwarded to the inherited discriminator
                training method in its declared order.
            **kwargs (dict): Additional discriminator-training options forwarded to the inherited
                implementation.

        Returns:
            tuple[tf.Tensor, tf.Tensor, tf.Tensor] or None: Loss, classification accuracy, and
                probability-based JS diagnostic; None when update_delay skips the update.
        """
        result = super().train(*args, **kwargs)
        if result is not None:
            self.disc.write_sparsity_summaries()
        return result

    def _after_update(self):
        """Finalize sparsity masks and update any target discriminator after training.

        Returns:
            None.
        """
        self.disc.finish_update()
        if self.tau > 0:
            update_target_variables(self.disc_target.weights, self.disc.weights, tau=self.tau)

    def inference(self, states, actions, next_states=None, apply_atol=False):
        """Return learned imitation rewards for the supplied observations and actions or
        transitions.

        Args:
            states (array_like): Observation batch shaped (batch, *state_shape).
            actions (array_like): Actions aligned with the observation batch; continuous actions have a
                final action_dim axis.
            next_states (array_like or None): Physical successor observations aligned with the current-
                state batch. This implementation does not read this argument.
            apply_atol (bool): Use the tolerance-filtered reward-network copy for inference.

        Returns:
            tf.Tensor: EQN-GAIL log-odds rewards log(D+epsilon)-log(1-D+epsilon), shaped (batch, 1),
                using the selected online/target/tolerance model.
        """
        states, actions = np.asarray(states, np.float32), np.asarray(actions, np.float32)
        if states.ndim == 1:
            states, actions = states[None], actions[None]
        inputs = np.concatenate((states, actions), axis=1)
        return self._inference_body_atol(inputs) if apply_atol else self._inference_body(inputs)

    @tf.function
    def _inference_body_atol(self, inputs):
        """Evaluate rewards using a copy with small weights suppressed by the tolerance.

        Args:
            inputs (tf.Tensor): Batched discriminator inputs with concatenated state/action or
                state/successor coordinates as required by the model.

        Returns:
            tf.Tensor: Log-odds rewards from the tolerance-filtered target discriminator.
        """
        update_target_variables(self.disc_target.weights,
                                [tf.where(tf.abs(w) > self.atol, w, 0.) for w in self.disc.weights])
        return self.disc_target.compute_reward(inputs)

    @tf.function
    def _inference_body(self, inputs):
        """Return learned imitation rewards for the supplied observations and actions or
        transitions.

        Args:
            inputs (tf.Tensor): Batched discriminator inputs with concatenated state/action or
                state/successor coordinates as required by the model.

        Returns:
            tf.Tensor: Log-odds reward predictions from the active online or target equation
                discriminator.
        """
        with tf.device(self.device):
            return (self.disc_target if self.tau > 0 else self.disc).compute_reward(inputs)

    @staticmethod
    def get_argument(parser=None):
        """Add this component's command-line options and return the argument parser.

        Args:
            parser (argparse.ArgumentParser or None): Parser to extend; None constructs a new parser.

        Returns:
            argparse.ArgumentParser: Parser extended with the supported component options.
        """
        return equation_arguments(GAIL.get_argument(parser))


EQNGAIL = GAIL_EQL
