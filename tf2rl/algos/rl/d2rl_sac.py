import tensorflow as tf
import tensorflow_probability as tfp

from tf2rl.algos.rl.sac import SAC, CriticQ
from tf2rl.networks.actors import GaussianActor
from tf2rl.utils.tensorflow import update_target_variables


class DenseCriticQ(CriticQ):
    """Q-value network that concatenates state-action inputs between hidden layers."""
    def call(self, states, actions):
        """Return Q-value estimates for a batch of state-action pairs.

        Args:
            states (array_like): Observation batch shaped (batch, *state_shape).
            actions (array_like): Actions aligned with the observation batch; continuous actions have a
                final action_dim axis.

        Returns:
            tf.Tensor: State-action Q estimates shaped (batch,).
        """
        state_action_pairs = tf.concat((states, actions), axis=1)
        features = state_action_pairs
        for layer_idx, cur_layer in enumerate(self.base_layers):
            features = cur_layer(features)
            # Do not concatenate for the last layer
            if not layer_idx + 1 == len(self.base_layers):
                features = tf.concat((features, state_action_pairs), axis=1)
        values = self.out_layer(features)
        return tf.squeeze(values, axis=1)


class DenseGaussianActor(GaussianActor):
    """Gaussian policy that concatenates input states between hidden layers."""
    def _compute_dist(self, states):
        """Return the diagonal Gaussian action distribution with dense input connections.

        Args:
            states (array_like): Observation batch shaped (batch, *state_shape).

        Returns:
            tfp.distributions.MultivariateNormalDiag: Diagonal Gaussian action distribution for the
                supplied observation batch.
        """
        features = states
        for layer_idx, cur_layer in enumerate(self.base_layers):
            features = cur_layer(features)
            # Do not concatenate for the last layer
            if not layer_idx + 1 == len(self.base_layers):
                features = tf.concat((features, states), axis=1)
        mean = self.out_mean(features)

        if self._state_independent_std:
            log_std = tf.tile(
                input=tf.expand_dims(self.out_logstd, axis=0),
                multiples=[mean.shape[0], 1])
        else:
            log_std = self.out_logstd(features)
            log_std = tf.clip_by_value(log_std, self.LOG_STD_CAP_MIN, self.LOG_STD_CAP_MAX)

        return tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=tf.exp(log_std))


class D2RLSAC(SAC):
    """SAC with dense connections that reintroduce inputs at hidden layers."""
    def __init__(self, *args, **kwargs):
        """Configure the agent's networks, optimizers, and training hyperparameters.

        Args:
            *args (tuple): Positional arguments forwarded to SAC construction.
            **kwargs (dict): Additional keyword arguments forwarded to SAC construction.

        Returns:
            None: Initializes the instance.
        """
        kwargs["name"] = "D2RL_SAC"
        super().__init__(*args, **kwargs)

    def _setup_critic_q(self, state_shape, action_dim, critic_units, lr):
        """Create Q-value networks, initialize their targets, and configure optimization.

        Args:
            state_shape (tuple[int, ...]): Shape of one observation, excluding the batch axis.
            action_dim (int): Number of discrete choices or width of a continuous action vector.
            critic_units (sequence[int]): Widths of the critic hidden layers.
            lr (float): Optimizer learning rate.

        Returns:
            None.
        """
        initializer = tf.keras.initializers.GlorotUniform()
        self.qf1 = DenseCriticQ(state_shape, action_dim, critic_units,
                                name="qf1", initializer=initializer, layer_norm=self.critic_layer_norm)
        self.qf2 = DenseCriticQ(state_shape, action_dim, critic_units,
                                name="qf2", initializer=initializer, layer_norm=self.critic_layer_norm)
        self.qf1_target = DenseCriticQ(state_shape, action_dim, critic_units,
                                       name="qf1_target", initializer=initializer, layer_norm=self.critic_layer_norm)
        self.qf2_target = DenseCriticQ(state_shape, action_dim, critic_units,
                                       name="qf2_target", initializer=initializer, layer_norm=self.critic_layer_norm)
        states = tf.zeros((1,) + tuple(state_shape), tf.float32)
        actions = tf.zeros((1, action_dim), tf.float32)
        for critic in (self.qf1, self.qf2, self.qf1_target, self.qf2_target):
            critic(states, actions, training=False)
        self.qf_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.qf_optimizer.build(self.qf1.trainable_variables + self.qf2.trainable_variables)
        update_target_variables(self.qf1_target.weights, self.qf1.weights, tau=1.)
        update_target_variables(self.qf2_target.weights, self.qf2.weights, tau=1.)

    def _setup_actor(self, state_shape, action_dim, actor_units, lr, max_action=1.):
        """Create the action policy and its optimizer.

        Args:
            state_shape (tuple[int, ...]): Shape of one observation, excluding the batch axis.
            action_dim (int): Number of discrete choices or width of a continuous action vector.
            actor_units (sequence[int]): Widths of the actor hidden layers.
            lr (float): Optimizer learning rate.
            max_action (float): Symmetric continuous-action bound or output scale.

        Returns:
            None.
        """
        self.actor = DenseGaussianActor(
            state_shape, action_dim, max_action, squash=True, units=actor_units)
        self.actor(tf.zeros((1,) + tuple(state_shape), tf.float32), training=False)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.actor_optimizer.build(self.actor.trainable_variables)
