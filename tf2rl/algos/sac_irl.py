"""Research SAC generator with the original three-valued DAC replay masks."""
import numpy as np
import tensorflow as tf

from tf2rl.algos.policy_base import OffPolicyAgent
from tf2rl.misc.target_update_ops import update_target_variables
from tf2rl.policies.tfp_gaussian_actor import GaussianActor


def dac_target(rewards, masks, next_q, next_logp, alpha, discount, absorbing):
    continuation = tf.maximum(masks, 0.)
    soft_q = next_q - alpha * next_logp * (continuation if absorbing else 1.)
    return rewards + discount * soft_q * (1. if absorbing else continuation)


def dac_actor_loss(logp, q, masks, alpha, absorbing):
    values = alpha * logp - q
    if not absorbing:
        return tf.reduce_mean(values)
    actor_mask = 1. - tf.maximum(-masks, 0.)
    # A batch consisting solely of self-loops has no actor objective.
    return tf.math.divide_no_nan(tf.reduce_sum(values * actor_mask), tf.reduce_sum(actor_mask))


class CriticQ(tf.keras.Model):
    def __init__(self, state_shape, action_dim, critic_units=(256, 256), name='qf',
                 initializer=None):
        super().__init__(name=name)
        initializer = initializer or tf.keras.initializers.GlorotUniform()
        self.base_layers = [tf.keras.layers.Dense(unit, activation='relu', kernel_initializer=initializer)
                            for unit in critic_units]
        for index, layer in enumerate(self.base_layers):
            setattr(self, f"hidden_{index}", layer)
        self.out_layer = tf.keras.layers.Dense(1, name="Q", kernel_initializer=initializer)
        self(tf.zeros((1,) + tuple(state_shape)), tf.zeros((1, action_dim)))

    def call(self, states, actions):
        features = tf.concat((states, actions), axis=1)
        for layer in self.base_layers:
            features = layer(features)
        return tf.squeeze(self.out_layer(features), axis=1)


class SAC_IRL(OffPolicyAgent):
    def __init__(self, state_shape, action_dim, name="SAC", max_action=1., lr=3e-4,
                 lr_alpha=3e-4, actor_units=(400, 300), critic_units=(400, 300),
                 tau=5e-3, alpha=.2, auto_alpha=False, n_warmup=int(1e4),
                 memory_capacity=int(2e6), is_debug=False, is_absorbed_state=False,
                 is_xla=False, **kwargs):
        super().__init__(name=name, memory_capacity=memory_capacity, n_warmup=n_warmup, **kwargs)
        if len(state_shape) != 1:
            raise ValueError("SAC_IRL supports vector observations")
        self._is_absorbed_state = is_absorbed_state
        self._is_debug = is_debug
        self.state_ndim = len(state_shape)
        self.tau = tau
        self.auto_alpha = auto_alpha
        with tf.device(self.device):
            self.actor = GaussianActor(state_shape, action_dim, max_action, squash=True,
                                       units=actor_units,
                                       initializer=tf.keras.initializers.TruncatedNormal(stddev=1e-3))
            self.actor(tf.zeros((1,) + tuple(state_shape)), test=False)
            self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            q_initializer = tf.keras.initializers.GlorotUniform()
            self.qf1 = CriticQ(state_shape, action_dim, critic_units, name="qf1", initializer=q_initializer)
            self.qf2 = CriticQ(state_shape, action_dim, critic_units, name="qf2", initializer=q_initializer)
            self.qf1_target = CriticQ(state_shape, action_dim, critic_units, name="qf1_target", initializer=q_initializer)
            self.qf2_target = CriticQ(state_shape, action_dim, critic_units, name="qf2_target", initializer=q_initializer)
            update_target_variables(self.qf1_target.weights, self.qf1.weights)
            update_target_variables(self.qf2_target.weights, self.qf2.weights)
            self.qf_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            if auto_alpha:
                # Preserve historical alpha initialization at exp(0), independent
                # of the supplied fixed-temperature alpha argument.
                self.log_alpha = self.add_weight(name="log_alpha", shape=(), initializer="zeros")
                self.alpha = self.add_weight(name="alpha", shape=(), initializer="ones", trainable=False)
                self.target_alpha = -action_dim
                self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_alpha)
            else:
                self.alpha = alpha
        self.actor_optimizer.build(self.actor.trainable_variables)
        self.qf_optimizer.build(self.qf1.trainable_variables + self.qf2.trainable_variables)
        if self.auto_alpha:
            self.alpha_optimizer.build([self.log_alpha])
        self._train_graph = tf.function(self._train_body, reduce_retracing=True, jit_compile=is_xla)
        self._action_graph = tf.function(self._get_action_body, reduce_retracing=True)
        self._td_graph = tf.function(self._compute_td_error_body, reduce_retracing=True)

    def get_action(self, state, test=False):
        state = np.asarray(state, dtype=np.float32)
        single = state.ndim == self.state_ndim
        actions, logps = self._action_graph(state[None] if single else state, tf.constant(test))
        return (actions.numpy()[0], logps.numpy()[0]) if single else (actions, logps)

    def _get_action_body(self, state, test):
        with tf.device(self.device):
            return self.actor(state, training=False, test=test)

    def get_logp(self, state, actions=None):
        """With actions=None, retain the historical fresh-action warmup logp sample."""
        state = np.asarray(state, dtype=np.float32)
        single = state.ndim == self.state_ndim
        batch = state[None] if single else state
        if actions is None:
            logp = self.actor(batch, training=False)[1]
        else:
            actions = np.asarray(actions, dtype=np.float32)
            logp = self.actor.compute_log_probs(batch, actions[None] if single else actions)
        return logp.numpy()[0] if single else logp.numpy()

    def train(self, states, actions, next_states, rewards, dones, weights=None):
        # Historical research objectives use uniform replay. Importance weighting
        # is unsupported rather than silently ignored.
        if weights is not None and not np.allclose(weights, 1):
            raise NotImplementedError("SAC_IRL does not implement importance-weighted losses")
        args = [tf.convert_to_tensor(x, dtype=tf.float32)
                for x in (states, actions, next_states, rewards, dones)]
        result = self._train_graph(*args)
        for label, value in zip(("target_Q", "actor_loss", "critic_Q1_value", "critic_Q1_loss",
                                 "logp_min", "logp_max", "logp_mean"), result):
            tf.summary.scalar(self.policy_name + "/" + label, tf.reduce_mean(value))
        tf.summary.scalar(self.policy_name + "/ent", self.alpha)
        return result[0]

    def _target(self, next_states, rewards, masks):
        next_actions, next_logp = self.actor(next_states)
        if self._is_absorbed_state:
            next_actions *= tf.maximum(masks, 0.)[:, None]
        next_q = tf.minimum(self.qf1_target(next_states, next_actions),
                            self.qf2_target(next_states, next_actions))
        return tf.stop_gradient(dac_target(rewards, masks, next_q, next_logp, self.alpha,
                                           self.discount, self._is_absorbed_state))

    def _train_body(self, states, actions, next_states, rewards, dones):
        with tf.device(self.device):
            rewards, masks = tf.reshape(rewards, (-1,)), tf.reshape(dones, (-1,))
            with tf.GradientTape(persistent=True) as tape:
                current_q1 = self.qf1(states, actions)
                current_q2 = self.qf2(states, actions)
                target_q = self._target(next_states, rewards, masks)
                loss1 = tf.reduce_mean(tf.square(target_q - current_q1))
                loss2 = tf.reduce_mean(tf.square(target_q - current_q2))
                critic_loss = loss1 + loss2
                sample_actions, logp = self.actor(states)
                q_policy = tf.minimum(self.qf1(states, sample_actions), self.qf2(states, sample_actions))
                actor_loss = dac_actor_loss(logp, q_policy, masks, self.alpha, self._is_absorbed_state)
                if self.auto_alpha:
                    alpha_loss = -tf.reduce_mean(self.log_alpha * tf.stop_gradient(logp + self.target_alpha))
            variables = self.qf1.trainable_variables + self.qf2.trainable_variables
            self.qf_optimizer.apply_gradients(zip(tape.gradient(critic_loss, variables), variables))
            update_target_variables(self.qf1_target.weights, self.qf1.weights, tau=self.tau)
            update_target_variables(self.qf2_target.weights, self.qf2.weights, tau=self.tau)
            # Skip Adam's momentum update as well when there is no actor sample.
            if not self._is_absorbed_state or tf.reduce_any(masks >= 0):
                variables = self.actor.trainable_variables
                self.actor_optimizer.apply_gradients(zip(tape.gradient(actor_loss, variables), variables))
            if self.auto_alpha:
                self.alpha_optimizer.apply_gradients([(tape.gradient(alpha_loss, self.log_alpha), self.log_alpha)])
                self.alpha.assign(tf.exp(self.log_alpha))
            del tape
        return (target_q, actor_loss, current_q1, loss1, tf.reduce_min(logp),
                tf.reduce_max(logp), tf.reduce_mean(logp))

    def compute_td_error(self, states, actions, next_states, rewards, dones):
        args = [tf.convert_to_tensor(x, dtype=tf.float32)
                for x in (states, actions, next_states, rewards, dones)]
        return self._td_graph(*args).numpy()

    def _compute_td_error_body(self, states, actions, next_states, rewards, dones):
        target = self._target(next_states, tf.reshape(rewards, (-1,)), tf.reshape(dones, (-1,)))
        return target - self.qf1(states, actions)

    @staticmethod
    def get_argument(parser=None):
        parser = OffPolicyAgent.get_argument(parser)
        parser.add_argument('--lr', type=float, default=3e-4)
        parser.add_argument('--lr-alpha', type=float, default=3e-4)
        parser.add_argument('--alpha', type=float, default=.2)
        parser.add_argument('--auto-alpha-disable', action='store_false')
        parser.add_argument('--draw-reward-function', action='store_true')
        parser.add_argument('--units', default='128,128')
        parser.add_argument('--xla', action='store_true')
        return parser
