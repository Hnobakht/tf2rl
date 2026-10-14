"""Dense GAIL with its original log(D) policy reward."""
import numpy as np
import tensorflow as tf

from tf2rl.algos.policy_base import IRLPolicy
from tf2rl.algos.irl_losses import (gail_loss, singular_penalty, agent_gradient_penalty,
                                   js_divergence)


class Discriminator(tf.keras.Model):
    def __init__(self, state_shape, action_dim, units=(32, 32), enable_sn=False,
                 output_activation='sigmoid', name="Discriminator", activation='relu',
                 w_ini='orthogonal'):
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
        return self.l3(self.l2(self.l1(inputs)))

    def compute_reward(self, inputs):
        return tf.math.log(self(inputs, training=False) + 1e-8)


class GAIL(IRLPolicy):
    equation_discriminator = False
    requires_log_probs = False

    def __init__(self, state_shape, action_dim, units=(32, 32), lr=0.001,
                 enable_sn=False, name="GAIL", is_debug=False, n_training=1,
                 grad_penalty_coeff=0, is_absorbed_state=False, update_delay=1, **kwargs):
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

    def _make_discriminator(self, state_shape, action_dim, units, enable_sn):
        return Discriminator(state_shape, action_dim, units, enable_sn)

    def train(self, agent_states, agent_acts, expert_states, expert_acts, itr=1, **kwargs):
        if itr % self.update_delay:
            return None
        result = self._train_body(agent_states, agent_acts, expert_states, expert_acts)
        for label, value in zip(("DiscriminatorLoss", "Accuracy", "JSdivergence"), result):
            tf.summary.scalar(self.policy_name + "/" + label, value)
        if self._is_debug:
            for variable in self.disc.weights:
                tf.summary.histogram(self.policy_name + "/" + variable.name, variable)
        return result

    def _after_update(self):
        pass

    @tf.function
    def _train_body(self, agent_states, agent_acts, expert_states, expert_acts, l1_regularizers=0):
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
        states, actions = np.asarray(states, np.float32), np.asarray(actions, np.float32)
        if states.ndim == 1:
            states, actions = states[None], actions[None]
        return self._inference_body(np.concatenate((states, actions), axis=1))

    @tf.function
    def _inference_body(self, inputs):
        with tf.device(self.device):
            return self.disc.compute_reward(inputs)

    @staticmethod
    def get_argument(parser=None):
        return IRLPolicy.get_argument(parser)
