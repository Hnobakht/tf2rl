"""Dense GAIL with its original log(D) policy reward."""
import numpy as np
import tensorflow as tf

from tf2rl.algos.base import RewardLearner
from tf2rl.algos.imitation.losses import (
    agent_gradient_penalty,
    gail_loss,
    js_divergence,
    singular_penalty,
)


class Discriminator(tf.keras.Model):
    """Neural discriminator distinguishing expert samples from agent samples."""
    def __init__(self, state_shape, action_dim, units=(32, 32), enable_sn=False,
                 output_activation='sigmoid', name="Discriminator", activation='relu',
                 w_ini='orthogonal'):
        """Build the expert-versus-agent discriminator with the requested architecture.

        Args:
            state_shape (tuple[int, ...]): Shape of one observation, excluding the batch axis.
            action_dim (int): Number of discrete choices or width of a continuous action vector.
            units (sequence[int]): Hidden-layer widths in forward order.
            enable_sn (bool): Request spectral normalization where the selected implementation supports
                it.
            output_activation (str or callable or None): Activation used by the scalar or discriminator
                output layer.
            name (str): Name assigned to the model, operation, or registered component.
            activation (str or callable or None): Output activation; None leaves the affine output
                linear.
            w_ini (str or keras.initializers.Initializer): Initializer for discriminator dense kernels.

        Returns:
            None: Initializes the instance.
        """
        super().__init__(name=name)
        # enable_sn was ignored by the original dense GAIL; retain its behavior.
        self._enable_sn = enable_sn
        self.l1 = tf.keras.layers.Dense(units[0], name="L1", activation=activation,
                                        kernel_initializer=w_ini)
        self.l2 = tf.keras.layers.Dense(units[1], name="L2", activation=activation,
                                        kernel_initializer=w_ini)
        self.l3 = tf.keras.layers.Dense(1, name="L3", activation=output_activation)
        self(tf.zeros((1, state_shape[0] + action_dim)), training=False)

    def call(self, inputs, training=None):
        """Return discriminator outputs for the supplied batch of inputs.

        Args:
            inputs (tf.Tensor): Batched discriminator inputs with concatenated state/action or
                state/successor coordinates as required by the model.
            training (bool or tf.Tensor or None): Keras training flag controlling stochastic or stateful
                layers; None follows the layer default. This implementation does not read this argument.

        Returns:
            tf.Tensor: Scalar discriminator or reward outputs shaped (batch, 1), with the configured
                output activation.
        """
        return self.l3(self.l2(self.l1(inputs)))

    def compute_reward(self, inputs):
        """Return the log discriminator probability as an imitation reward.

        Args:
            inputs (tf.Tensor): Batched discriminator inputs with concatenated state/action or
                state/successor coordinates as required by the model.

        Returns:
            tf.Tensor: Scalar learned rewards for each supplied input row, shaped (batch, 1).
        """
        return tf.math.log(self(inputs, training=False) + 1e-8)


class GAIL(RewardLearner):
    """Generative adversarial imitation learning from expert state-action samples."""
    equation_discriminator = False
    requires_log_probs = False

    def __init__(self, state_shape, action_dim, units=(32, 32), lr=0.001,
                 enable_sn=False, name="GAIL", is_debug=False, n_training=1,
                 grad_penalty_coeff=0, is_absorbed_state=False, update_delay=1, **kwargs):
        """Configure the imitation objective, discriminator networks, and optimizer settings.

        Args:
            state_shape (tuple[int, ...]): Shape of one observation, excluding the batch axis.
            action_dim (int): Number of discrete choices or width of a continuous action vector.
            units (sequence[int]): Hidden-layer widths in forward order.
            lr (float): Optimizer learning rate.
            enable_sn (bool): Request spectral normalization where the selected implementation supports
                it.
            name (str): Name assigned to the model, operation, or registered component.
            is_debug (bool): Enable additional discriminator/parameter diagnostics where implemented.
            n_training (int): Number of discriminator training attempts per scheduled generator update.
            grad_penalty_coeff (float): Coefficient multiplying the discriminator gradient penalty.
            is_absorbed_state (bool): Use DAC absorbing Bellman targets and omit synthetic loops from
                actor losses.
            update_delay (int): Run discriminator training only when its supplied iteration is divisible
                by this value.
            **kwargs (dict): Additional keyword arguments forwarded to RewardLearner construction.

        Returns:
            None: Initializes the instance.

        Raises:
            ValueError: update_delay must be positive and GP coefficient nonnegative.
        """
        super().__init__(name=name, n_training=n_training, **kwargs)
        if update_delay < 1 or grad_penalty_coeff < 0:
            raise ValueError("update_delay must be positive and GP coefficient nonnegative")
        self._is_debug = is_debug
        self._is_absorbed_state = is_absorbed_state
        self.update_delay = update_delay
        self.grad_penalty_coeff = grad_penalty_coeff
        self._is_grad_penalty = grad_penalty_coeff > 0
        self.disc = self._make_discriminator(state_shape, action_dim, units, enable_sn)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5)
        self.optimizer.build(self.disc.trainable_variables)

    def _make_discriminator(self, state_shape, action_dim, units, enable_sn):
        """Build the discriminator used to distinguish expert and agent samples.

        Args:
            state_shape (tuple[int, ...]): Shape of one observation, excluding the batch axis.
            action_dim (int): Number of discrete choices or width of a continuous action vector.
            units (sequence[int]): Hidden-layer widths in forward order.
            enable_sn (bool): Request spectral normalization where the selected implementation supports
                it.

        Returns:
            Discriminator: Dense state-action discriminator whose sigmoid output estimates expert
                probability.
        """
        return Discriminator(state_shape, action_dim, units, enable_sn)

    def train(self, agent_states, agent_acts, expert_states, expert_acts, itr=1, **kwargs):
        """Update the imitation discriminator from agent and expert minibatches and report
        diagnostics.

        Args:
            agent_states (array_like): Agent observation batch shaped (agent_batch, *state_shape).
            agent_acts (array_like): Agent actions aligned with agent_states.
            expert_states (array_like): Expert observation batch shaped (expert_batch, *state_shape).
            expert_acts (array_like): Expert actions aligned with expert_states.
            itr (int): Training iteration/environment-step counter used by discriminator update-delay
                logic.
            **kwargs (dict): Additional compatibility keyword options, currently unused.

        Returns:
            tuple[tf.Tensor, tf.Tensor, tf.Tensor] or None: Loss, classification accuracy, and
                probability-based JS diagnostic; None when update_delay skips the update.
        """
        if itr % self.update_delay:
            return None
        self._before_update()
        result = self._train_body(agent_states, agent_acts, expert_states, expert_acts)
        for label, value in zip(("DiscriminatorLoss", "Accuracy", "JSdivergence"), result):
            tf.summary.scalar(self.policy_name + "/" + label, value)
        if self._is_debug:
            for variable in self.disc.weights:
                tf.summary.histogram(self.policy_name + "/" + variable.name, variable)
        return result

    def _before_update(self):
        """Provide a no-op hook for subclasses before discriminator updates.

        Returns:
            None.
        """
        pass

    def _after_update(self):
        """Provide a no-op hook for subclasses after discriminator updates.

        Returns:
            None.
        """
        pass

    @tf.function
    def _train_body(self, agent_states, agent_acts, expert_states, expert_acts, l1_regularizers=0):
        """Apply discriminator gradients for the expert and agent batches and return training
        metrics.

        Args:
            agent_states (array_like): Agent observation batch shaped (agent_batch, *state_shape).
            agent_acts (array_like): Agent actions aligned with agent_states.
            expert_states (array_like): Expert observation batch shaped (expert_batch, *state_shape).
            expert_acts (array_like): Expert actions aligned with expert_states.
            l1_regularizers (float): Historical per-call L1 coefficient; some modern equation paths
                retain but ignore it. This implementation does not read this argument.

        Returns:
            tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Scalar discriminator loss, classification accuracy,
                and JS diagnostic after applying gradients.
        """
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                real = self.disc(tf.concat((expert_states, expert_acts), axis=1))
                fake = self.disc(tf.concat((agent_states, agent_acts), axis=1))
                loss = gail_loss(real, fake)
                if self._is_grad_penalty:
                    loss += self.grad_penalty_coeff * agent_gradient_penalty(
                        self.disc, tf.concat((agent_states, agent_acts), axis=1))
                if self.equation_discriminator:
                    # Match the historical last-call model.losses semantics (also
                    # when the GP's third forward call uses a fresh dropout mask).
                    loss += singular_penalty(self.disc)
            variables = self.disc.trainable_variables
            self.optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))
            self._after_update()
        accuracy = (tf.reduce_mean(tf.cast(real >= 0.5, tf.float32))
                    + tf.reduce_mean(tf.cast(fake < 0.5, tf.float32))) / 2.
        return loss, accuracy, js_divergence(fake, real)

    _compute_js_divergence = staticmethod(js_divergence)

    def inference(self, states, actions, next_states=None):
        """Return learned imitation rewards for the supplied observations and actions or
        transitions.

        Args:
            states (array_like): Observation batch shaped (batch, *state_shape).
            actions (array_like): Actions aligned with the observation batch; continuous actions have a
                final action_dim axis.
            next_states (array_like or None): Physical successor observations aligned with the current-
                state batch. This implementation does not read this argument.

        Returns:
            tf.Tensor: Learned log(D(s,a)+epsilon) rewards shaped (batch, 1).
        """
        states, actions = np.asarray(states, np.float32), np.asarray(actions, np.float32)
        if states.ndim == 1:
            states, actions = states[None], actions[None]
        return self._inference_body(np.concatenate((states, actions), axis=1))

    @tf.function
    def _inference_body(self, inputs):
        """Return learned imitation rewards for the supplied observations and actions or
        transitions.

        Args:
            inputs (tf.Tensor): Batched discriminator inputs with concatenated state/action or
                state/successor coordinates as required by the model.

        Returns:
            tf.Tensor: Batched log-discriminator rewards shaped (batch, 1).
        """
        with tf.device(self.device):
            return self.disc.compute_reward(inputs)

    @staticmethod
    def get_argument(parser=None):
        """Add this component's command-line options and return the argument parser.

        Args:
            parser (argparse.ArgumentParser or None): Parser to extend; None constructs a new parser.

        Returns:
            argparse.ArgumentParser: Parser extended with the supported component options.
        """
        return RewardLearner.get_argument(parser)
