"""AIRL discriminator with reward/potential shaping and the original log-sum-exp loss."""
import numpy as np
import tensorflow as tf

from tf2rl.algos.gail import Discriminator
from tf2rl.algos.irl_losses import airl_loss, singular_penalty
from tf2rl.algos.policy_base import IRLPolicy
from tf2rl.networks.spectral_normalization import SpectralNormalization


class StateActionModel(Discriminator):
    def compute_reward(self, inputs):
        return tf.math.log(self(inputs, training=False) + 1e-8)


class StateModel(tf.keras.Model):
    def __init__(self, state_shape, units=(32, 32), enable_sn=False,
                 output_activation="sigmoid", name="StateModel"):
        super().__init__(name=name)
        def dense(size, activation, name):
            layer = tf.keras.layers.Dense(size, activation=activation, name=name)
            return SpectralNormalization(layer) if enable_sn else layer
        self.l1 = dense(units[0], tf.nn.tanh, "L1")
        self.l2 = dense(units[1], tf.nn.tanh, "L2")
        self.l3 = dense(1, output_activation, "L3")
        self(tf.zeros((1,) + tuple(state_shape)), training=False)

    def call(self, inputs, training=None):
        training = True if training is None else training
        return self.l3(self.l2(self.l1(inputs, training=training), training=training),
                       training=training)


class AIRL(IRLPolicy):
    requires_log_probs = True
    equation_reward = False

    def __init__(self, state_shape, action_dim, state_only=True, units=(32, 32),
                 lr=0.001, enable_sn=False, name="AIRL", is_debug=False,
                 update_delay=1, **kwargs):
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
        if self._state_only:
            return StateModel(state_shape, units, enable_sn, "linear", name="reward_net")
        return StateActionModel(state_shape, action_dim, units, enable_sn,
                                "linear", name="reward_net")

    def train(self, agent_states, agent_acts, agent_next_states, agent_logps,
              expert_states, expert_acts, expert_next_states, expert_logps, itr=1):
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
        pass

    def _after_update(self):
        pass

    @tf.function
    def _train_body(self, agent_states, agent_acts, agent_next_states, agent_logps,
                    expert_states, expert_acts, expert_next_states, expert_logps):
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
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if states.ndim == 1:
            states, actions = states[None], actions[None]
        return self._inference_body(states, actions)

    @tf.function
    def _inference_body(self, states, actions):
        with tf.device(self.device):
            inputs = states if self._state_only else tf.concat((states, actions), 1)
            return self.rew_net(inputs, training=False)

    @tf.function
    def _inference_body_disc(self, states, actions, next_states, logps):
        return (self._inference_body(states, actions)
                + self.discount * self.val_net(next_states, training=False)
                - self.val_net(states, training=False) - tf.reshape(logps, (-1, 1)))

    def inference_bellman(self, *args, **kwargs):
        raise NotImplementedError("The experimental AIRL Bellman reward is not implemented")

    @staticmethod
    def get_argument(parser=None):
        parser = IRLPolicy.get_argument(parser)
        parser.add_argument('--state-only', action='store_true')
        return parser
