import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from tf2rl.algos.policy_base import OffPolicyAgent
from tf2rl.misc.absorbing_ops import ReplaySemantics
from tf2rl.misc.target_update_ops import update_target_variables


class Actor(tf.keras.Model):
    """Deterministic policy mapping state vectors to bounded continuous actions."""
    def __init__(self, state_shape, action_dim, max_action, units=(400, 300), name="Actor"):
        """Build the deterministic policy and set its continuous action bounds."""
        super().__init__(name=name)

        initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=2e-3)

        self.l1 = Dense(units[0], name="L1", kernel_initializer=initializer)
        self.l2 = Dense(units[1], name="L2", kernel_initializer=initializer)
        self.l3 = Dense(action_dim, name="L3", kernel_initializer=initializer)

        self.max_action = max_action


    def call(self, inputs):
        """Return bounded continuous actions for the input observation batch."""
        features = tf.nn.relu(self.l1(inputs))
        features = tf.nn.relu(self.l2(features))
        features = self.l3(features)
        action = self.max_action * tf.nn.tanh(features)
        return action


class Critic(tf.keras.Model):
    """Action-value network for deterministic continuous-control agents."""
    def __init__(self, state_shape, action_dim, units=(400, 300), name="Critic"):
        """Build the state-action value network and initialize its parameters."""
        super().__init__(name=name)

        self.l1 = Dense(units[0], name="L1")
        self.l2 = Dense(units[1], name="L2")
        self.l3 = Dense(1, name="L3")


    def call(self, states, actions):
        """Return Q-value estimates for a batch of state-action pairs."""
        features = tf.concat((states, actions), axis=1)
        features = tf.nn.relu(self.l1(features))
        features = tf.nn.relu(self.l2(features))
        values = self.l3(features)
        return tf.squeeze(values, axis=1)


class DDPG(OffPolicyAgent):
    """Deep deterministic policy gradient with optional DAC absorbing states.

    ``is_absorbed_state=True`` expects DAC masks (1, 0, -1) in replay;
    otherwise ``done`` means true termination. Timeouts always bootstrap.
    """
    def __init__(
            self,
            state_shape,
            action_dim,
            name="DDPG",
            max_action=1.,
            lr_actor=0.001,
            lr_critic=0.001,
            actor_units=(400, 300),
            critic_units=(400, 300),
            sigma=0.1,
            tau=0.005,
            n_warmup=int(1e4),
            memory_capacity=int(1e6),
            is_absorbed_state=False,
            **kwargs):
        """Configure the agent's networks, optimizers, and training hyperparameters."""
        super().__init__(name=name, memory_capacity=memory_capacity, n_warmup=n_warmup, **kwargs)
        self._is_absorbed_state = is_absorbed_state
        self._replay = ReplaySemantics(is_absorbed_state)
        self.replay_mask_encoding = self._replay.mask_encoding

        # Define and initialize Actor network
        self.actor = Actor(state_shape, action_dim, max_action, actor_units)
        self.actor_target = Actor(state_shape, action_dim, max_action, actor_units)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_actor)

        # Define and initialize Critic network
        self.critic = Critic(state_shape, action_dim, critic_units)
        self.critic_target = Critic(state_shape, action_dim, critic_units)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_critic)

        # Set hyperparameters
        self.sigma = sigma
        self.tau = tau
        self._initialize_networks(state_shape, action_dim)

    def _initialize_networks(self, state_shape, action_dim):
        """Build lazy Keras networks before copying targets or creating optimizer slots."""
        with tf.device(self.device):
            dtype = tf.uint8 if getattr(self, "_is_pixels", False) else tf.float32
            states = tf.zeros((1, *state_shape), dtype)
            actions = tf.zeros((1, action_dim), tf.float32)
            self.actor(states)
            self.actor_target(states)
            self.critic(states, actions)
            self.critic_target(states, actions)
            update_target_variables(self.actor_target.weights, self.actor.weights, tau=1.)
            update_target_variables(self.critic_target.weights, self.critic.weights, tau=1.)
            self.actor_optimizer.build(self.actor.trainable_variables)
            self.critic_optimizer.build(self.critic.trainable_variables)

    def get_action_tensor(self, states, test=False):
        """Tensor inference shared by DDPG and TD3 vector collectors."""
        with tf.device(self.device):
            return self.get_action(states, test=test, tensor=True)

    def get_action(self, state, test=False, tensor=False):
        """Select an action for one observation or a batch; test mode disables exploration."""
        is_single_state = len(state.shape) == 1
        if not tensor:
            assert isinstance(state, np.ndarray)
        state = np.expand_dims(state, axis=0).astype(
            np.float32) if is_single_state else state
        action = self._get_action_body(
            tf.constant(state), self.sigma * (1. - test),
            tf.constant(self.actor.max_action, dtype=tf.float32))
        if tensor:
            return action
        else:
            return action.numpy()[0] if is_single_state else action.numpy()

    @tf.function
    def _get_action_body(self, state, sigma, max_action):
        """Compute policy actions for a batch on the configured TensorFlow device."""
        print("[DEBUG] initializing {_get_action_body DDPG}.")
        with tf.device(self.device):
            action = self.actor(state)
            if sigma > 0.:
                action += tf.random.normal(shape=action.shape,
                                           mean=0., stddev=sigma, dtype=tf.float32)
            return tf.clip_by_value(action, -max_action, max_action)

    def train(self, states, actions, next_states, rewards, done, weights=None):
        """
        Update the actor and critics from a replay minibatch and return temporal-difference errors.
        """
        if weights is None:
            weights = np.ones_like(rewards)
        actor_loss, critic_loss, td_errors = self._train_body(
            states, actions, next_states, rewards, done, weights)

        if actor_loss is not None:
            tf.summary.scalar(name=self.policy_name + "/actor_loss",
                              data=actor_loss)
        tf.summary.scalar(name=self.policy_name + "/critic_loss",
                          data=critic_loss)
        # tf.summary.scalar(name=self.policy_name + "/q_value",
        #                   data=q_value)

        return td_errors

    @tf.function
    def _train_body(self, states, actions, next_states, rewards, dones, weights):
        """Apply actor and critic gradient updates, update target networks, and return diagnostics.
        """
        print("[DEBUG] initializing {_train_body}.")
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                td_errors = self._compute_td_error_body(
                    states, actions, next_states, rewards, dones)
                # Flatten (B,1) replay weights to avoid broadcasting to (B,B).
                critic_loss = tf.reduce_mean(td_errors ** 2 * tf.reshape(weights, (-1,)))

            critic_grad = tape.gradient(
                critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(
                zip(critic_grad, self.critic.trainable_variables))

            with tf.GradientTape() as tape:
                sample_actions = self.actor(states)
                actor_loss = self._replay.actor_mean(-self.critic(states, sample_actions), dones)

            if self._replay.has_actor_samples(dones):
                actor_grad = tape.gradient(
                    actor_loss, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(
                    zip(actor_grad, self.actor.trainable_variables))

            # Update target networks
            update_target_variables(
                self.critic_target.weights, self.critic.weights, self.tau)
            update_target_variables(
                self.actor_target.weights, self.actor.weights, self.tau)

            return actor_loss, critic_loss, td_errors

    def compute_td_error(self, states, actions, next_states, rewards, dones):
        """Return temporal-difference errors for the supplied transition batch."""
        if isinstance(actions, tf.Tensor):
            rewards = tf.expand_dims(rewards, axis=1)
            dones = tf.expand_dims(dones, 1)
        td_errors = self._compute_td_error_body(
            states, actions, next_states, rewards, dones)
        return np.abs(np.ravel(td_errors.numpy()))

    @tf.function
    def _compute_td_error_body(self, states, actions, next_states, rewards, dones):
        """Compute batched temporal-difference errors using the target networks."""
        print("[DEBUG] initializing {_compute_td_error_body}.")
        rewards = tf.reshape(rewards, (-1,))
        dones = tf.reshape(dones, (-1,))

        with tf.device(self.device):
            next_act_target = self._replay.next_actions(self.actor_target(next_states), dones)
            next_q_target = self.critic_target(next_states, next_act_target)
            target_q = self._replay.target(rewards, next_q_target, dones, self.discount)
            current_q = self.critic(states, actions)
            td_errors = tf.stop_gradient(target_q) - current_q
        return td_errors

    @staticmethod
    def get_argument(parser=None):
        """Add this component's command-line options and return the argument parser."""
        parser = OffPolicyAgent.get_argument(parser)
        parser.add_argument('--lr', type=float, default=3e-4, help=" <default: %(default)s>")
        parser.add_argument('--draw-reward-function', action="store_true", help=" <default: %(default)s>")
        parser.add_argument('--units', type=str, default="128,128",
                            help='Network architectures. use case: 128,128 or 400,300. <default: %(default)s>')
        parser.add_argument('--sigma', type=float, default=0.1,
                            help=" DDPG/TD3 Action noise coe <default: %(default)s>")
        parser.add_argument('--tau', type=float, default=0.005,
                            help=" DDPG/TD3 Weight average update coe <default: %(default)s>")
        return parser
