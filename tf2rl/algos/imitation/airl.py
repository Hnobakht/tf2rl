"""AIRL discriminator with reward/potential shaping and the original log-sum-exp loss."""
import numpy as np
import tensorflow as tf

from tf2rl.algos.base import RewardLearner
from tf2rl.algos.imitation.gail import Discriminator
from tf2rl.algos.imitation.losses import airl_loss, singular_penalty
from tf2rl.networks.spectral_normalization import SpectralNormalization


class StateActionModel(Discriminator):
    """AIRL reward model over concatenated state-action inputs."""
    def compute_reward(self, inputs):
        """Return the log discriminator probability as an imitation reward.

        Args:
            inputs (tf.Tensor): Batched discriminator inputs with concatenated state/action or
                state/successor coordinates as required by the model.

        Returns:
            tf.Tensor: Stabilized log of the scalar model output, shaped (batch, 1).
        """
        return tf.math.log(self(inputs, training=False) + 1e-8)


class StateModel(tf.keras.Model):
    """Dense scalar model over states for AIRL rewards or shaping potentials."""
    def __init__(self, state_shape, units=(32, 32), enable_sn=False,
                 output_activation="sigmoid", name="StateModel"):
        """Build a dense scalar state model with optional spectral normalization.

        Args:
            state_shape (tuple[int, ...]): Shape of one observation, excluding the batch axis.
            units (sequence[int]): Hidden-layer widths in forward order.
            enable_sn (bool): Request spectral normalization where the selected implementation supports
                it.
            output_activation (str or callable or None): Activation used by the scalar or discriminator
                output layer.
            name (str): Name assigned to the model, operation, or registered component.

        Returns:
            None: Initializes the instance.
        """
        super().__init__(name=name)
        def dense(size, activation, name):
            """Create a dense layer, optionally wrapped with spectral normalization.

            Args:
                size (int or None): Replay capacity; None selects the agent capacity or this helper default.
                activation (str or callable or None): Output activation; None leaves the affine output
                    linear.
                name (str): Name assigned to the model, operation, or registered component.

            Returns:
                keras.layers.Layer: Dense layer, wrapped in spectral normalization when requested by the
                    enclosing model.
            """
            layer = tf.keras.layers.Dense(size, activation=activation, name=name)
            return SpectralNormalization(layer) if enable_sn else layer
        self.l1 = dense(units[0], tf.nn.tanh, "L1")
        self.l2 = dense(units[1], tf.nn.tanh, "L2")
        self.l3 = dense(1, output_activation, "L3")
        self(tf.zeros((1,) + tuple(state_shape)), training=False)

    def call(self, inputs, training=None):
        """Return scalar AIRL model outputs for the supplied input batch.

        Args:
            inputs (tf.Tensor): Batched discriminator inputs with concatenated state/action or
                state/successor coordinates as required by the model.
            training (bool or tf.Tensor or None): Keras training flag controlling stochastic or stateful
                layers; None follows the layer default.

        Returns:
            tf.Tensor: Scalar discriminator or reward outputs shaped (batch, 1), with the configured
                output activation.
        """
        training = True if training is None else training
        return self.l3(self.l2(self.l1(inputs, training=training), training=training),
                       training=training)


class AIRL(RewardLearner):
    """Learn a reward and potential using the AIRL discriminator factorization.

    ``f(s,a,s') = r(s,a) + gamma*V(s') - V(s)`` separates a reward candidate
    from potential-based shaping (Fu et al., https://arxiv.org/abs/1710.11248).
    Inference returns ``r`` alone. State-only parameterization supports the
    paper's transfer motivation, but does not itself guarantee identifiability
    or recovery of a physical reward, especially with stored policy densities.
    """
    requires_log_probs = True
    equation_reward = False

    def __init__(self, state_shape, action_dim, state_only=True, units=(32, 32),
                 lr=0.001, enable_sn=False, name="AIRL", is_debug=False,
                 update_delay=1, **kwargs):
        """Configure the imitation objective, discriminator networks, and optimizer settings.

        Args:
            state_shape (tuple[int, ...]): Shape of one observation, excluding the batch axis.
            action_dim (int): Number of discrete choices or width of a continuous action vector.
            state_only (bool): Build the learned reward from observations without concatenating actions.
            units (sequence[int]): Hidden-layer widths in forward order.
            lr (float): Optimizer learning rate.
            enable_sn (bool): Request spectral normalization where the selected implementation supports
                it.
            name (str): Name assigned to the model, operation, or registered component.
            is_debug (bool): Enable additional discriminator/parameter diagnostics where implemented.
            update_delay (int): Run discriminator training only when its supplied iteration is divisible
                by this value.
            **kwargs (dict): Additional keyword arguments forwarded to RewardLearner construction.

        Returns:
            None: Initializes the instance.

        Raises:
            ValueError: update_delay must be positive.
        """
        super().__init__(name=name, n_training=kwargs.pop("n_training", 1), **kwargs)
        if update_delay < 1:
            raise ValueError("update_delay must be positive")
        self._state_only = state_only
        self._is_debug = is_debug
        self.update_delay = update_delay
        self.rew_net = self._make_reward(state_shape, action_dim, units, enable_sn)
        self.val_net = StateModel(state_shape, units, enable_sn, "linear", name="val_net")
        self.rew_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.rew_optimizer.build(self.rew_net.trainable_variables + self.val_net.trainable_variables)

    def _make_reward(self, state_shape, action_dim, units, enable_sn):
        """Build the state-only or state-action reward network.

        Args:
            state_shape (tuple[int, ...]): Shape of one observation, excluding the batch axis.
            action_dim (int): Number of discrete choices or width of a continuous action vector.
            units (sequence[int]): Hidden-layer widths in forward order.
            enable_sn (bool): Request spectral normalization where the selected implementation supports
                it.

        Returns:
            keras.Model: Unshaped scalar reward model using states alone or concatenated states/actions.
        """
        if self._state_only:
            return StateModel(state_shape, units, enable_sn, "linear", name="reward_net")
        return StateActionModel(state_shape, action_dim, units, enable_sn,
                                "linear", name="reward_net")

    def train(self, agent_states, agent_acts, agent_next_states, agent_logps,
              expert_states, expert_acts, expert_next_states, expert_logps, itr=1):
        """Update the imitation discriminator from agent and expert minibatches and report
        diagnostics.

        Args:
            agent_states (array_like): Agent observation batch shaped (agent_batch, *state_shape).
            agent_acts (array_like): Agent actions aligned with agent_states.
            agent_next_states (array_like): Agent successor observations aligned with agent_states.
            agent_logps (array_like): Stored agent action log densities, with one value per transition.
            expert_states (array_like): Expert observation batch shaped (expert_batch, *state_shape).
            expert_acts (array_like): Expert actions aligned with expert_states.
            expert_next_states (array_like): Expert successor observations aligned with expert_states.
            expert_logps (array_like): Stored expert action log densities, with one value per
                transition.
            itr (int): Training iteration/environment-step counter used by discriminator update-delay
                logic.

        Returns:
            tuple[tf.Tensor, tf.Tensor, tf.Tensor] or None: Discriminator loss, mean agent reward, and
                mean agent potential; None when update_delay skips this iteration.
        """
        if itr % self.update_delay:
            return None
        self._before_update()
        result = self._train_body(agent_states, agent_acts, agent_next_states, agent_logps,
                                  expert_states, expert_acts, expert_next_states, expert_logps)
        for label, value in zip(("Loss", "rewardF", "valueF"), result):
            tf.summary.scalar(self.policy_name + "/" + label, value)
        if self._is_debug:
            for variable in self.rew_net.weights + self.val_net.weights:
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
    def _train_body(self, agent_states, agent_acts, agent_next_states, agent_logps,
                    expert_states, expert_acts, expert_next_states, expert_logps):
        """Apply discriminator gradients for the expert and agent batches and return training
        metrics.

        Args:
            agent_states (array_like): Agent observation batch shaped (agent_batch, *state_shape).
            agent_acts (array_like): Agent actions aligned with agent_states.
            agent_next_states (array_like): Agent successor observations aligned with agent_states.
            agent_logps (array_like): Stored agent action log densities, with one value per transition.
            expert_states (array_like): Expert observation batch shaped (expert_batch, *state_shape).
            expert_acts (array_like): Expert actions aligned with expert_states.
            expert_next_states (array_like): Expert successor observations aligned with expert_states.
            expert_logps (array_like): Stored expert action log densities, with one value per
                transition.

        Returns:
            tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Scalar discriminator loss, mean agent reward, and
                mean agent state potential after updating reward and value networks.
        """
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                real_inputs = expert_states if self._state_only else tf.concat((expert_states, expert_acts), 1)
                fake_inputs = agent_states if self._state_only else tf.concat((agent_states, agent_acts), 1)
                real_rewards = self.rew_net(real_inputs)
                fake_rewards = self.rew_net(fake_inputs)
                real_values = self.val_net(expert_states)
                real_next_values = self.val_net(expert_next_states)
                fake_values = self.val_net(agent_states)
                fake_next_values = self.val_net(agent_next_states)
                # Terminal rows already point to the explicit absorbing state;
                # do not add a boolean terminal mask to this shaping difference.
                real_f = real_rewards + self.discount * real_next_values - real_values
                fake_f = fake_rewards + self.discount * fake_next_values - fake_values
                loss = airl_loss(real_f, fake_f, expert_logps, agent_logps)
                if self.equation_reward:
                    loss += singular_penalty(self.rew_net)
            variables = self.rew_net.trainable_variables + self.val_net.trainable_variables
            self.rew_optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))
            self._after_update()
        return loss, tf.reduce_mean(fake_rewards), tf.reduce_mean(fake_values)

    def inference(self, states, actions, next_states=None, logps=None, **kwargs):
        """Return raw learned rewards with dropout disabled, without the shaping potential.

        Args:
            states (array_like): Observation batch shaped (batch, *state_shape).
            actions (array_like): Actions aligned with the observation batch; continuous actions have a
                final action_dim axis.
            next_states (array_like or None): Physical successor observations aligned with the current-
                state batch. This implementation does not read this argument.
            logps (array_like or None): Compatibility density argument, unused for raw reward inference.
            **kwargs (dict): Additional compatibility keyword options, currently unused.

        Returns:
            tf.Tensor: Raw learned rewards of shape (batch, 1); successor states and log densities do
                not enter the exported policy reward.
        """
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if states.ndim == 1:
            states, actions = states[None], actions[None]
        return self._inference_body(states, actions)

    @tf.function
    def _inference_body(self, states, actions):
        """Return raw learned rewards with dropout disabled, without the shaping potential.

        Args:
            states (array_like): Observation batch shaped (batch, *state_shape).
            actions (array_like): Actions aligned with the observation batch; continuous actions have a
                final action_dim axis.

        Returns:
            tf.Tensor: Raw state-only or state-action reward predictions shaped (batch, 1), without
                potential shaping.
        """
        with tf.device(self.device):
            inputs = states if self._state_only else tf.concat((states, actions), 1)
            return self.rew_net(inputs, training=False)

    @tf.function
    def _inference_body_disc(self, states, actions, next_states, logps):
        """Return AIRL discriminator logits including potential shaping and policy log density.

        Args:
            states (array_like): Observation batch shaped (batch, *state_shape).
            actions (array_like): Actions aligned with the observation batch; continuous actions have a
                final action_dim axis.
            next_states (array_like): Physical successor observations aligned with the current-state
                batch.
            logps (array_like): Collection-time action log probabilities aligned with observations.

        Returns:
            tf.Tensor: Shaped discriminator log odds r(s,a)+gamma*V(next_s)-V(s)-logpi(a|s), shaped
                (batch, 1).
        """
        return (self._inference_body(states, actions)
                + self.discount * self.val_net(next_states, training=False)
                - self.val_net(states, training=False) - tf.reshape(logps, (-1, 1)))

    def inference_bellman(self, *args, **kwargs):
        """Raise NotImplementedError for the unsupported experimental Bellman reward.

        Args:
            *args (tuple): Positional arguments forwarded to the wrapped operation.
            **kwargs (dict): Additional compatibility keyword options, currently unused.

        Returns:
            NoReturn: The base or unsupported implementation always raises NotImplementedError.

        Raises:
            NotImplementedError: The experimental AIRL Bellman reward is not implemented.
        """
        raise NotImplementedError("The experimental AIRL Bellman reward is not implemented")

    @staticmethod
    def get_argument(parser=None):
        """Add this component's command-line options and return the argument parser.

        Args:
            parser (argparse.ArgumentParser or None): Parser to extend; None constructs a new parser.

        Returns:
            argparse.ArgumentParser: Parser extended with the supported component options.
        """
        parser = RewardLearner.get_argument(parser)
        parser.add_argument('--state-only', action='store_true')
        return parser
