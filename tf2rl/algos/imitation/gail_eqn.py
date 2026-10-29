"""EQN-GAIL with equation penalties, delayed updates and log-odds rewards."""
import numpy as np
import tensorflow as tf

from tf2rl.algos.imitation.gail import GAIL
from tf2rl.networks.equation import EquationNetwork
from tf2rl.utils.tensorflow import update_target_variables

Discriminator_EQL = EquationNetwork


def equation_arguments(parser):
    """Add equation-network architecture, sparsity, and export options to the parser."""
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
        """Configure the imitation objective, discriminator networks, and optimizer settings."""
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
        """Build the discriminator used to distinguish expert and agent samples."""
        return EquationNetwork(**self._equation_config)

    @property
    def equation_model(self):
        """Return the equation network used for reward inference and export."""
        return self.disc_target if self.tau > 0 else self.disc

    def _before_update(self):
        """Prepare the equation sparsity stage before the discriminator update."""
        self.disc.prepare_update()

    def train(self, *args, **kwargs):
        """
        Update the imitation discriminator from agent and expert minibatches and report diagnostics.
        """
        result = super().train(*args, **kwargs)
        if result is not None:
            self.disc.write_sparsity_summaries()
        return result

    def _after_update(self):
        """Finalize sparsity masks and update any target discriminator after training."""
        self.disc.finish_update()
        if self.tau > 0:
            update_target_variables(self.disc_target.weights, self.disc.weights, tau=self.tau)

    def inference(self, states, actions, next_states=None, apply_atol=False):
        """
        Return learned imitation rewards for the supplied observations and actions or transitions.
        """
        states, actions = np.asarray(states, np.float32), np.asarray(actions, np.float32)
        if states.ndim == 1:
            states, actions = states[None], actions[None]
        inputs = np.concatenate((states, actions), axis=1)
        return self._inference_body_atol(inputs) if apply_atol else self._inference_body(inputs)

    @tf.function
    def _inference_body_atol(self, inputs):
        """Evaluate rewards using a copy with small weights suppressed by the tolerance."""
        update_target_variables(self.disc_target.weights,
                                [tf.where(tf.abs(w) > self.atol, w, 0.) for w in self.disc.weights])
        return self.disc_target.compute_reward(inputs)

    @tf.function
    def _inference_body(self, inputs):
        """
        Return learned imitation rewards for the supplied observations and actions or transitions.
        """
        with tf.device(self.device):
            return (self.disc_target if self.tau > 0 else self.disc).compute_reward(inputs)

    @staticmethod
    def get_argument(parser=None):
        """Add this component's command-line options and return the argument parser."""
        return equation_arguments(GAIL.get_argument(parser))


EQNGAIL = GAIL_EQL
