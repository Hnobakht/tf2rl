import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Conv2D, MaxPool2D, Flatten

from tf2rl.algos.policy_base import OffPolicyAgent
from tf2rl.misc.target_update_ops import update_target_variables
from tf2rl.policies.tfp_gaussian_actor import GaussianActor, GaussianActor_pixels


class CriticQ(tf.keras.Model):
    """Neural action-value function over state-action pairs."""
    def __init__(self, state_shape, action_dim, critic_units=(256, 256), name='qf',
                 initializer = tf.keras.initializers.glorot_uniform()):
        """Build the hidden state-action layers and scalar Q-value head."""
        super().__init__(name=name)

        self.base_layers = []
        for unit in critic_units:
            self.base_layers.append(Dense(unit, activation='relu', kernel_initializer=initializer))
            self.base_layers.append(LayerNormalization())  # TODO: For GAIL_EQL
        self.out_layer = Dense(1, name="Q", activation='linear', kernel_initializer=initializer)
        for index, layer in enumerate(self.base_layers):
            setattr(self, f"hidden_{index}", layer)


    def call(self, states, actions, training=None):
        """Return Q-value estimates for a batch of state-action pairs."""
        features = tf.concat((states, actions), axis=1)
        for cur_layer in self.base_layers:
            features = cur_layer(features)
        values = self.out_layer(features)
        return tf.squeeze(values, axis=1)

class CriticQ_pixels(tf.keras.Model):
    """Action-value function using convolutional features from pixel observations."""
    def __init__(self, state_shape, action_dim, critic_units=None, name='qf',
                 initializer = tf.keras.initializers.glorot_uniform()):
        """Build the image encoder and state-action Q-value layers."""
        super().__init__(name=name)

        self.cnn_layers = []
        self.cnn_layers.append(Conv2D(4, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu'))
        self.cnn_layers.append(Conv2D(8, kernel_size=(3, 3), strides=(1,1), padding='same',  activation='relu'))
        self.cnn_layers.append(MaxPool2D(pool_size=(2, 2)))
        self.cnn_layers.append(Conv2D(8, kernel_size=(3, 3), padding='same', activation='relu'))
        self.cnn_layers.append(Flatten())

        self.base_layers = []
        self.base_layers.append(Dense(128, activation='relu', kernel_initializer=initializer))
        self.base_layers.append(LayerNormalization())
        self.base_layers.append(Dense(64, activation='relu', kernel_initializer=initializer))
        self.base_layers.append(LayerNormalization())

        self.out_layer = Dense(1, name="Q", activation='linear', kernel_initializer=initializer)

    def call(self, states, actions, training=None):
        """Return Q-value estimates for a batch of state-action pairs."""
        features_cnn = tf.cast(states/255, dtype=tf.float32)
        print("[DEBUG] input: " + str(states.shape) + " [call CriticQ_pixels]")
        for cnn_layer in self.cnn_layers:
            features_cnn = cnn_layer(features_cnn, training=training)
            print("[DEBUG] features: " + str(features_cnn.shape))

        features = tf.concat((features_cnn, actions), axis=1)
        for cur_layer in self.base_layers:
            features = cur_layer(features)
            print("[DEBUG] features: " + str(features.shape))
        values = self.out_layer(features, training=training)
        return tf.squeeze(values, axis=1)


class SAC(OffPolicyAgent):
    """Soft actor-critic with stochastic actions, twin Q networks, and entropy regularization."""
    def     __init__(
            self,
            state_shape,
            action_dim,
            name="SAC",
            max_action=1.,
            lr=3e-4,
            lr_alpha=3e-4,
            actor_units=(400, 300),
            critic_units=(400, 300),
            is_pixels = False,
            tau=5e-3,
            alpha=.2,
            auto_alpha=False,
            n_warmup=int(1e4),
            memory_capacity=int(2e6),
            is_debug = False,
            is_xla = False,
            **kwargs):
        """Configure the agent's networks, optimizers, and training hyperparameters."""
        super().__init__(
            name=name, memory_capacity=memory_capacity, n_warmup=n_warmup, **kwargs)
        self._is_debug = is_debug
        self._is_xla = is_xla
        self._is_pixels = is_pixels
        self._setup_actor(state_shape, action_dim, actor_units, lr, max_action)
        self._setup_critic_q(state_shape, action_dim, critic_units, lr)

        # Set hyper-parameters
        self.tau = tau
        self.auto_alpha = auto_alpha
        with tf.device(self.device):
            if auto_alpha:
                self.log_alpha = self.add_weight(name="log_alpha", shape=(), initializer="zeros")
                self.alpha = self.add_weight(name="alpha", shape=(), initializer="zeros", trainable=False)
                self.alpha.assign(tf.exp(self.log_alpha))
                self.target_alpha = -action_dim
                self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_alpha)
                self.alpha_optimizer.build([self.log_alpha])
            else:
                self.alpha = alpha

        self.state_ndim = len(state_shape)

        self._init_static_graph(state_shape, action_dim)

    def _setup_actor(self, state_shape, action_dim, actor_units, lr, max_action=1.):
        """Create the action policy and its optimizer."""
        std_dev = 1e-4 if self._is_pixels else 1e-3
        policy_func = GaussianActor_pixels if self._is_pixels else GaussianActor
        with tf.device(self.device):
            self.actor = policy_func(state_shape, action_dim, max_action, squash=True,units=actor_units,
                                       initializer=tf.keras.initializers.TruncatedNormal(stddev=std_dev))
            self.actor(tf.zeros((1,) + tuple(state_shape), dtype=tf.uint8 if self._is_pixels else tf.float32), training=False)
            self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            self.actor_optimizer.build(self.actor.trainable_variables)

    def _setup_critic_q(self, state_shape, action_dim, critic_units, lr):
        """Create Q-value networks, initialize their targets, and configure optimization."""
        Q_func = CriticQ_pixels if self._is_pixels else CriticQ
        with tf.device(self.device):
            q_initializer = tf.keras.initializers.GlorotUniform()
            self.qf1 = Q_func(state_shape, action_dim, critic_units, name="qf1", initializer=q_initializer)
            self.qf2 = Q_func(state_shape, action_dim, critic_units, name="qf2", initializer=q_initializer)
            self.qf1_target = Q_func(state_shape, action_dim, critic_units, name="qf1_target", initializer=q_initializer)
            self.qf2_target = Q_func(state_shape, action_dim, critic_units, name="qf2_target", initializer=q_initializer)

            self.qf_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

            dummy_states = tf.zeros((1,) + tuple(state_shape), dtype=tf.uint8 if self._is_pixels else tf.float32)
            dummy_actions = tf.zeros((1, action_dim), tf.float32)
            for critic in (self.qf1, self.qf2, self.qf1_target, self.qf2_target):
                critic(dummy_states, dummy_actions, training=False)
            self.qf_optimizer.build(self.qf1.trainable_variables + self.qf2.trainable_variables)
            update_target_variables(self.qf1_target.weights, self.qf1.weights, tau=1.)
            update_target_variables(self.qf2_target.weights, self.qf2.weights, tau=1.)

    def get_action(self, state, test=False):
        """Return actions and their log probabilities; test mode selects deterministic actions."""
        assert isinstance(state, np.ndarray)
        is_single_state = len(state.shape) == self.state_ndim

        state_dtype = np.uint8 if self._is_pixels else np.float32
        if is_single_state:
            state = np.expand_dims(state, axis=0).astype(state_dtype)

        with tf.device(self.device):
            action, logps = self._get_action_body_graph(tf.constant(state), tf.constant(test))

        return (action.numpy()[0], logps.numpy()[0]) if is_single_state else (action, logps)

    def _get_action_body(self, state, test):
        """Return batched policy actions and log probabilities on the configured device."""
        print("[DEBUG] initializing {_get_action_body SAC}...")
        with tf.device(self.device):
            actions, log_pis = self.actor(state, training=False, test=test)
        return actions, log_pis

    def train(self, states, actions, next_states, rewards, dones, weights=None):
        """
        Update the actor and critics from a replay minibatch and return temporal-difference errors.
        """
        if weights is None:
            weights = np.ones_like(rewards)

        with tf.device(self.device):
            states, actions, next_states, rewards, dones = \
                tuple([tf.constant(i) for i in (states, actions, next_states, rewards, dones)])

            td_errors, actor_loss, qf1, qf1_loss, logp_min, logp_max, logp_mean = self._train_body_graph(
                states, actions, next_states, rewards, dones)

            tf.summary.scalar(name=self.policy_name + "/actor_loss", data=actor_loss)
            tf.summary.scalar(name=self.policy_name + "/critic_Q1_value", data=qf1.numpy().mean())
            tf.summary.scalar(name=self.policy_name + "/critic_Q1_loss", data=qf1_loss)
            tf.summary.scalar(name=self.policy_name + "/logp_min", data=logp_min)
            tf.summary.scalar(name=self.policy_name + "/logp_max", data=logp_max)
            tf.summary.scalar(name=self.policy_name + "/logp_mean", data=logp_mean)
            if self.auto_alpha:
                tf.summary.scalar(name=self.policy_name + "/log_ent", data=self.log_alpha)
                tf.summary.scalar(name=self.policy_name + "/logp_mean+target", data=logp_mean + self.target_alpha)
            tf.summary.scalar(name=self.policy_name + "/ent", data=self.alpha)

        return td_errors

    def _train_body(self, states, actions, next_states, rewards, dones):#, weights):
        """Apply actor and critic gradient updates, update target networks, and return diagnostics.
        """
        print("[DEBUG] initializing {_train_body SAC}")
        with tf.device(self.device):
            assert len(dones.shape) == 2
            assert len(rewards.shape) == 2
            rewards = tf.squeeze(rewards, axis=1)
            dones = tf.squeeze(dones, axis=1)

            not_dones = 1. - tf.cast(dones, dtype=tf.float32)

            with tf.GradientTape(persistent=True) as tape:
                # Compute loss of critic Q
                current_q1 = self.qf1(states, actions, training=True)
                current_q2 = self.qf2(states, actions, training=True)

                sample_next_actions, next_logp = self.actor(next_states, training=False)
                next_q1_target = self.qf1_target(next_states, sample_next_actions, training=False)
                next_q2_target = self.qf2_target(next_states, sample_next_actions, training=False)
                next_q_min_target = tf.minimum(next_q1_target, next_q2_target)
                soft_next_q = next_q_min_target - self.alpha * next_logp

                target_q = tf.stop_gradient(rewards + not_dones * self.discount * soft_next_q)

                td_loss_q1 = tf.reduce_mean((target_q - current_q1) ** 2)
                td_loss_q2 = tf.reduce_mean((target_q - current_q2) ** 2)  # Eq.(7)
                td_loss_q = td_loss_q1 + td_loss_q2

                sample_actions, logp = self.actor(states, training=True)  # Resample actions to update V
                current_q1_policy = self.qf1(states, sample_actions, training=False)
                current_q2_policy = self.qf2(states, sample_actions, training=False)
                current_min_q_policy = tf.minimum(current_q1_policy, current_q2_policy)

                # Compute loss of policy
                policy_loss = tf.reduce_mean(self.alpha * logp - current_min_q_policy)  # Eq.(12)

                # Compute loss of temperature parameter for entropy
                if self.auto_alpha:
                    alpha_loss = -tf.reduce_mean((self.log_alpha * tf.stop_gradient(logp + self.target_alpha)))

            trainable_var = self.qf1.trainable_variables + self.qf2.trainable_variables
            q_grad = tape.gradient(td_loss_q, trainable_var)
            self.qf_optimizer.apply_gradients(zip(q_grad, trainable_var))

            update_target_variables(self.qf1_target.weights, self.qf1.weights, tau=self.tau)
            update_target_variables(self.qf2_target.weights, self.qf2.weights, tau=self.tau)

            actor_grad = tape.gradient(policy_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

            if self.auto_alpha:
                alpha_grad = tape.gradient(alpha_loss, [self.log_alpha])
                self.alpha_optimizer.apply_gradients(zip(alpha_grad, [self.log_alpha]))
                self.alpha.assign(tf.exp(self.log_alpha))

        return target_q, policy_loss, current_q1, td_loss_q1, tf.reduce_min(logp), tf.reduce_max(logp), tf.reduce_mean(
            logp)

    def compute_td_error(self, states, actions, next_states, rewards, dones):
        """Return temporal-difference errors for the supplied transition batch."""
        if isinstance(actions, tf.Tensor):
            rewards = tf.expand_dims(rewards, axis=1)
            dones = tf.expand_dims(dones, 1)

        with tf.device(self.device):
            states, actions, next_states, rewards, dones = \
                tuple([tf.constant(i) for i in (states, actions, next_states, rewards, dones)])

            td_errors = self._compute_td_error_body_graph(states, actions, next_states, rewards, dones)

        return td_errors.numpy()

    def _compute_td_error_body(self, states, actions, next_states, rewards, dones):
        """Compute batched temporal-difference errors using the target networks."""
        print("[DEBUG] initializing {_compute_td_error_body}...")
        with tf.device(self.device):
            masks = 1. - tf.reshape(tf.cast(dones, tf.float32), (-1,))
            actions_next, logp = self.actor(next_states, training=False)
            q_next = tf.minimum(self.qf1_target(next_states, actions_next, training=False),
                                self.qf2_target(next_states, actions_next, training=False))
            target = tf.reshape(rewards, (-1,)) + masks * self.discount * (q_next - self.alpha * logp)
            return tf.stop_gradient(target) - self.qf1(states, actions, training=False)

    def get_logp(self, state):
        """Return policy log probabilities for the supplied states and actions."""
        assert isinstance(state, np.ndarray)
        is_single_state = len(state.shape) == self.state_ndim

        state = np.expand_dims(state, axis=0).astype(
            np.float32) if is_single_state else state

        with tf.device(self.device):
            logp = self._get_logp_body_graph(tf.constant(state))

        return logp.numpy()[0] if is_single_state else logp.numpy()

    def _get_logp_body(self, state):
        """Evaluate action log probabilities on the configured TensorFlow device."""
        with tf.device(self.device):
            logp = self.actor(state, training=False)[1]
        return logp

    def _init_static_graph(self, state_shape, action_dim):
        """Set up TensorFlow signatures and traced model, training, or inference computations."""
        if self._is_pixels:
            state_spec = tuple([None]+list(state_shape))
            state_dtype = tf.uint8
            xla = self._is_xla
        else:
            state_spec = (None, state_shape[-1])
            state_dtype = tf.float32
            xla = self._is_xla

        with tf.device(self.device):
            self._train_body_graph = tf.function(self._train_body,
                                                 input_signature=[tf.TensorSpec(shape=state_spec, dtype=state_dtype),
                                                                  tf.TensorSpec(shape=(None, action_dim), dtype=tf.float32),
                                                                  tf.TensorSpec(shape=state_spec, dtype=state_dtype),
                                                                  tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                                                  tf.TensorSpec(shape=(None, 1), dtype=tf.float32)],
                                                 jit_compile=xla)

            self._compute_td_error_body_graph = tf.function(self._compute_td_error_body,
                                                            input_signature=[tf.TensorSpec(shape=state_spec, dtype=state_dtype),
                                                                             tf.TensorSpec(shape=(None, action_dim), dtype=tf.float32),
                                                                             tf.TensorSpec(shape=state_spec, dtype=state_dtype),
                                                                             tf.TensorSpec(shape=(None, 1),dtype=tf.float32),
                                                                             tf.TensorSpec(shape=(None, 1),dtype=tf.float32)],
                                                            jit_compile=xla)

            self._get_logp_body_graph = tf.function(self._get_logp_body,
                                                    input_signature=[tf.TensorSpec(shape=state_spec,dtype=state_dtype)],
                                                    jit_compile=xla)

            self._get_action_body_graph = tf.function(self._get_action_body,
                                                      input_signature=[tf.TensorSpec(shape=state_spec, dtype=state_dtype),
                                                                       tf.TensorSpec(shape=(None), dtype=tf.bool)],
                                                      jit_compile=xla)

    @staticmethod
    def get_argument(parser=None):
        """Add this component's command-line options and return the argument parser."""
        parser = OffPolicyAgent.get_argument(parser)
        parser.add_argument('--lr', type=float, default=3e-4, help=" <default: %(default)s>")
        parser.add_argument('--lr-alpha', type=float, default=3e-4, help=" <default: %(default)s>")
        parser.add_argument('--alpha', type=float, default=0.2, help=" <default: %(default)s>")
        parser.add_argument('--auto-alpha-disable', action="store_false", help=" <default: %(default)s>")
        parser.add_argument('--draw-reward-function', action="store_true", help=" <default: %(default)s>")
        parser.add_argument('--units', type=str, default="128,128",
                            help='Network architectures. use case: 128,128 or 400,300. <default: %(default)s>')
        return parser
