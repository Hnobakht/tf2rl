import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

from tf2rl.algos.ddpg import DDPG
from tf2rl.algos.policy_base import OffPolicyAgent
from tf2rl.misc.target_update_ops import update_target_variables

class Actor(tf.keras.Model):
    def __init__(self, state_shape, action_dim, max_action, units=(400, 300), name="Actor"):
        super().__init__(name=name)

        initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=2e-3)

        self.l1 = Dense(units[0], name="L1", kernel_initializer=initializer)
        self.l2 = Dense(units[1], name="L2", kernel_initializer=initializer)
        self.l3 = Dense(action_dim, name="L3", kernel_initializer=initializer)

        self.max_action = max_action

        # with tf.device("/cpu:0"):
        #     self(tf.constant(np.zeros(shape=(1,) + state_shape, dtype=np.float32)))

    @tf.function
    def call(self, inputs):
        features = tf.nn.relu(self.l1(inputs))
        features = tf.nn.relu(self.l2(features))
        features = self.l3(features)
        action = self.max_action * tf.nn.tanh(features)
        return action

class Critic(tf.keras.Model):
    def __init__(self, state_shape, action_dim, units=(400, 300), name="Critic"):
        super().__init__(name=name)

        # initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=2e-3)
        initializer = tf.keras.initializers.glorot_uniform()

        self.l1 = Dense(units[0], name="L1", kernel_initializer=initializer)
        self.l2 = Dense(units[1], name="L2", kernel_initializer=initializer)
        self.l3 = Dense(1, name="L3", kernel_initializer=initializer)

        self.l4 = Dense(units[0], name="L4", kernel_initializer=initializer)
        self.l5 = Dense(units[1], name="L5", kernel_initializer=initializer)
        self.l6 = Dense(1, name="L6", kernel_initializer=initializer)

        # dummy_state = tf.constant(
        #     np.zeros(shape=(1,)+state_shape, dtype=np.float32))
        # dummy_action = tf.constant(
        #     np.zeros(shape=(1, action_dim), dtype=np.float32))
        # with tf.device("/gpu:0"):
        #     self(dummy_state, dummy_action)

    def call(self, states, actions):
        xu = tf.concat((states, actions), axis=1)

        x1 = tf.nn.relu(self.l1(xu))
        x1 = tf.nn.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = tf.nn.relu(self.l4(xu))
        x2 = tf.nn.relu(self.l5(x2))
        x2 = self.l6(x2)

        return tf.squeeze(x1, axis=1), tf.squeeze(x2, axis=1)

class Actor_pixels(tf.keras.Model):
    def __init__(self, state_shape, action_dim, max_action, units=(400, 300), name="Actor"):
        super().__init__(name=name)

        initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=1e-4)

        self.cnn_layers = []
        self.cnn_layers.append(Conv2D(16, kernel_size=(3, 3), activation='relu'))
        # self.cnn_layers.append(MaxPool2D(2, 2))
        self.cnn_layers.append(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        # self.cnn_layers.append(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        self.cnn_layers.append(Flatten())

        self.l1 = Dense(200, name="L1", kernel_initializer=initializer)
        self.l2 = Dense(200, name="L2", kernel_initializer=initializer)
        self.l3 = Dense(action_dim, name="L3", kernel_initializer=initializer)

        self.max_action = max_action

        # with tf.device("/gpu:0"):
        #     self(tf.constant(np.zeros(shape=(1,) + state_shape, dtype=np.uint8)))
    @tf.function
    def call(self, inputs):
        print("[DEBUG] Input: " + str(inputs.shape) + " Dtype: " + str(inputs.dtype) + "[call Actor_pixels TD3]")
        features = tf.cast(inputs/255, dtype=tf.float32)

        for layer in self.cnn_layers:
            features = layer(features)

        features = tf.nn.relu(self.l1(features))
        features = tf.nn.relu(self.l2(features))
        features = self.l3(features)
        action = self.max_action * tf.nn.tanh(features)
        return action

class Critic_pixels(tf.keras.Model):
    def __init__(self, state_shape, action_dim, units=(400, 300), name="Critic"):
        super().__init__(name=name)

        # initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=2e-3)
        initializer = tf.keras.initializers.glorot_uniform()

        units = (200,200)

        # First Critic Network
        self.cnn_layers_1 = []
        self.cnn_layers_1.append(Conv2D(16, kernel_size=(3, 3), activation='relu', name="cnn_1_1"))
        # self.cnn_layers_1.append(MaxPool2D(2,2))
        self.cnn_layers_1.append(Conv2D(32, kernel_size=(3, 3), activation='relu', name="cnn_1_2"))
        # self.cnn_layers_1.append(Conv2D(32, kernel_size=(3, 3), activation='relu', name="cnn_1_3"))
        self.cnn_layers_1.append(Flatten())

        self.l1 = Dense(units[0], name="L1", kernel_initializer=initializer, activation='relu')
        self.l2 = Dense(units[1], name="L2", kernel_initializer=initializer, activation='relu')
        self.l3 = Dense(1, name="L3", kernel_initializer=initializer)

        # 2nd Critic Network
        self.cnn_layers_2 = []
        self.cnn_layers_2.append(Conv2D(16, kernel_size=(3, 3), activation='relu', name="cnn_2_1"))
        # self.cnn_layers_2.append(MaxPool2D(2, 2))
        self.cnn_layers_2.append(Conv2D(32, kernel_size=(3, 3), activation='relu', name="cnn_2_2"))
        # self.cnn_layers_2.append(Conv2D(32, kernel_size=(3, 3), activation='relu', name="cnn_2_3"))
        self.cnn_layers_2.append(Flatten())

        self.l4 = Dense(units[0], name="L4", kernel_initializer=initializer, activation='relu')
        self.l5 = Dense(units[1], name="L5", kernel_initializer=initializer, activation='relu')
        self.l6 = Dense(1, name="L6", kernel_initializer=initializer)

        # dummy_state = tf.constant(
        #     np.zeros(shape=(1,)+state_shape, dtype=np.uint8))
        # dummy_action = tf.constant(
        #     np.zeros(shape=(1, action_dim), dtype=np.float32))
        # with tf.device("/gpu:0"):
        #     self(dummy_state, dummy_action)

    @tf.function
    def call(self, states, actions):
        print("[DEBUG] Input state: " + str(states.shape) + " Dtype: " + str(states.dtype) + "[call Critic_pixels TD3]")
        features_cnn = tf.cast(states/255, dtype=tf.float32)

        features_cnn_1 = features_cnn
        for layer1 in self.cnn_layers_1:
            features_cnn_1 = layer1(features_cnn_1)

        # x1 = tf.concat((features_cnn_1, actions), axis=1)
        # x1 = self.l1(x1)
        x1 = self.l1(features_cnn_1)
        x1 = tf.concat((x1, actions), axis=1)
        x1 = self.l2(x1)
        x1 = self.l3(x1)

        features_cnn_2 = features_cnn
        for layer2 in self.cnn_layers_2:
            features_cnn_2 = layer2(features_cnn_2)

        # x2 = tf.concat((features_cnn_2, actions), axis=1)
        # x2 = self.l1(x2)
        x2 = self.l4(features_cnn_2)
        x2 = tf.concat((x2, actions), axis=1)
        x2 = self.l5(x2)
        x2 = self.l6(x2)

        return tf.squeeze(x1, axis=1), tf.squeeze(x2, axis=1)

class TD3(DDPG):
    def __init__(
            self,
            state_shape,
            action_dim,
            name="TD3",
            max_action=1.,
            lr_actor=0.001,
            lr_critic=0.001,
            actor_units=(400, 300),
            critic_units=(400, 300),
            sigma=0.1,
            tau=0.005,
            n_warmup=int(1e4),
            memory_capacity=int(1e6),
            actor_update_freq=2,
            policy_noise=0.2,
            noise_clip=0.5,
            is_pixels = False,
            **kwargs):
        # super().__init__(name=name, state_shape=state_shape, action_dim=action_dim, **kwargs)
        OffPolicyAgent.__init__(self,name=name, memory_capacity=memory_capacity, n_warmup=n_warmup, **kwargs)

        # Define and initialize Actor network
        if is_pixels:
            actor_func = Actor_pixels
        else:
            actor_func = Actor
        self.actor = actor_func(state_shape, action_dim, max_action, actor_units)
        self.actor_target = actor_func(state_shape, action_dim, max_action, actor_units)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_actor)
        update_target_variables(self.actor_target.weights,
                                self.actor.weights, tau=1.)

        # Define and initialize Critic network
        if is_pixels:
            critic_func = Critic_pixels
        else:
            critic_func = Critic
        self.critic = critic_func(state_shape, action_dim, critic_units)
        self.critic_target = critic_func(state_shape, action_dim, critic_units)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_critic)
        update_target_variables(
            self.critic_target.weights, self.critic.weights, tau=1.)

        # Set hyperparameters
        self.sigma = sigma
        self.tau = tau

        self._policy_noise = policy_noise
        self._noise_clip = noise_clip

        self._actor_update_freq = actor_update_freq
        self._it = tf.Variable(0, dtype=tf.int32)

        self._is_pixels = is_pixels

        self.state_ndim = len(state_shape)

    def get_action(self, state, test=False, tensor=False):
        is_single_state = len(state.shape) == self.state_ndim

        if not tensor:
            assert isinstance(state, np.ndarray)

        state_dtype = np.uint8 if self._is_pixels else np.float32
        if is_single_state:
            state = np.expand_dims(state, axis=0).astype(state_dtype)

        action = self._get_action_body(
            tf.constant(state), self.sigma * (1. - test),
            tf.constant(self.actor.max_action, dtype=tf.float32))
        if tensor:
            return action
        else:
            return action.numpy()[0] if is_single_state else action.numpy()

    @tf.function
    def _train_body(self, states, actions, next_states, rewards, dones, weights):
        print("[DEBUG] initializing {_train_body TD3}.")
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                td_errors1, td_errors2 = self._compute_td_error_body(states, actions, next_states, rewards, dones)
                critic_loss = (tf.reduce_mean(td_errors1 ** 2 * weights) +
                               tf.reduce_mean(td_errors2 ** 2 * weights))

            critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

            # assert len(dones.shape) == 2
            # assert len(rewards.shape) == 2
            # rewards = tf.squeeze(rewards, axis=1)
            # dones = tf.squeeze(dones, axis=1)
            # not_dones = 1. - tf.cast(dones, dtype=tf.float32)
            # with tf.GradientTape() as tape:
            #     # Get noisy action
            #     next_action = self.actor_target(next_states)
            #     noise = tf.cast(
            #         tf.clip_by_value(tf.random.normal(shape=tf.shape(next_action), stddev=self._policy_noise),
            #                          -self._noise_clip, self._noise_clip), tf.float32)
            #
            #     next_action = tf.clip_by_value(next_action + noise,
            #                                    -self.actor_target.max_action,
            #                                    self.actor_target.max_action)
            #
            #     next_q1_target, next_q2_target = self.critic_target(next_states, next_action)
            #     next_q_target = tf.minimum(next_q1_target, next_q2_target)
            #     q_target = tf.stop_gradient(rewards + not_dones * self.discount * next_q_target)
            #
            #     current_q1, current_q2 = self.critic(states, actions)
            #
            #     critic_loss = (tf.reduce_mean(((q_target - current_q1) ** 2) * weights) +
            #                    tf.reduce_mean(((q_target - current_q2) ** 2) * weights))
            #
            # critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
            # self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

            self._it.assign_add(1)
            with tf.GradientTape() as tape:
                sample_actions = self.actor(states)
                actor_loss = - tf.reduce_mean(self.critic(states, sample_actions))

            remainder = tf.math.mod(self._it, self._actor_update_freq)

            def optimize_actor():
                actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
                return self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))
                # Update target networks. Moved the updates to the delay routine.
            def update_critic():
                return update_target_variables(self.critic_target.weights, self.critic.weights, self.tau)
            def update_actor():
                return update_target_variables(self.actor_target.weights, self.actor.weights, self.tau)

            tf.cond(pred=tf.equal(remainder, 0), true_fn=optimize_actor, false_fn=tf.no_op)
            tf.cond(pred=tf.equal(remainder, 0), true_fn=update_critic, false_fn=tf.no_op)
            tf.cond(pred=tf.equal(remainder, 0), true_fn=update_actor, false_fn=tf.no_op)
            # # Update target networks
            # update_target_variables(
            #     self.critic_target.weights, self.critic.weights, self.tau)
            # update_target_variables(
            #     self.actor_target.weights, self.actor.weights, self.tau)

            return actor_loss, critic_loss, tf.abs(td_errors1) + tf.abs(td_errors2)
            # return actor_loss, critic_loss, 0.0
            # return actor_loss, critic_loss, tf.reduce_mean(current_q1)

    def compute_td_error(self, states, actions, next_states, rewards, dones):
        td_errors1, td_errors2 = self._compute_td_error_body(
            states, actions, next_states, rewards, dones)
        return np.squeeze(np.abs(td_errors1.numpy()) + np.abs(td_errors2.numpy()))

    @tf.function
    def _compute_td_error_body(self, states, actions, next_states, rewards, dones):
        print("[DEBUG] initializing {_compute_td_error_body TD3}.")
        assert len(dones.shape) == 2
        assert len(rewards.shape) == 2
        rewards = tf.squeeze(rewards, axis=1)
        dones = tf.squeeze(dones, axis=1)

        with tf.device(self.device):
            not_dones = 1. - tf.cast(dones, dtype=tf.float32)

            # Get noisy action
            next_action = self.actor_target(next_states)
            noise = tf.cast(tf.clip_by_value(tf.random.normal(shape=tf.shape(next_action),stddev=self._policy_noise),
                                             -self._noise_clip, self._noise_clip), tf.float32)

            next_action = tf.clip_by_value(next_action + noise,
                                           -self.actor_target.max_action,
                                           self.actor_target.max_action)

            next_q1_target, next_q2_target = self.critic_target(next_states, next_action)
            next_q_target = tf.minimum(next_q1_target, next_q2_target)
            q_target = tf.stop_gradient(rewards + not_dones * self.discount * next_q_target)
            current_q1, current_q2 = self.critic(states, actions)

        return q_target - current_q1, q_target - current_q2

    @staticmethod
    def get_argument(parser=None):
        parser = DDPG.get_argument(parser)

        parser.add_argument('--actor-update-freq', type=int, default=2,
                            help=" TD3 actor update freq <default: %(default)s>")
        parser.add_argument('--policy-noise', type=float, default=0.1,
                            help="TD3 policy noise coe <default: %(default)s>")
        parser.add_argument('--noise_clip', type=float, default=0.5,
                            help=" TD3 noise clip coe <default: %(default)s>")
        return parser
