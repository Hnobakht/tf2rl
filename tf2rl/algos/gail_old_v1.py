import numpy as np
import tensorflow as tf
from tf2rl.networks.spectral_normalization import SpectralNormalization
from tensorflow.keras.layers import Dense
# import tensorflow_gan.python.losses.losses_impl as tfgan_losses

from tf2rl.algos.policy_base import IRLPolicy
# from tf2rl.networks.spectral_norm_dense import SNDense


class Discriminator(tf.keras.Model):
    def __init__(self, state_shape, action_dim, units=(32, 32),
                 enable_sn=False, output_activation='sigmoid',
                 name="Discriminator", activation='relu',
                 w_ini = 'orthogonal'):
        super().__init__(name=name)
        self._enable_sn = enable_sn
        # w_ini = tf.keras.initializers.Ones()

        # DenseClass = SNDense if enable_sn else Dense
        # self.l1 = DenseClass(units[0], name="L1", activation="relu")
        # self.l2 = DenseClass(units[1], name="L2", activation="relu")
        # self.l3 = DenseClass(1, name="L3", activation=output_activation)
        if not enable_sn:
            self.l1 = Dense(units[0], name="L1", activation=activation, kernel_initializer=w_ini)
            self.l2 = Dense(units[1], name="L2", activation=activation, kernel_initializer=w_ini)
            self.l3 = Dense(1, name="L3", activation=output_activation)
        else:
            print("[DEBUG] Spectral Normalization is enabled")
            self.l1 = SpectralNormalization(Dense(units[0], name="L1", activation=activation))#, kernel_initializer=w_ini))
            self.l2 = SpectralNormalization(Dense(units[1], name="L2", activation=activation))# , kernel_initializer=w_ini))
            self.l3 = SpectralNormalization(Dense(1, name="L3", activation=output_activation))

        dummy_state = tf.constant(
            np.zeros(shape=(1,)+state_shape, dtype=np.float32))
        dummy_action = tf.constant(
            np.zeros(shape=(1, action_dim), dtype=np.float32))
        with tf.device("/gpu:0"):
            self(tf.concat((dummy_state, dummy_action), axis=1))

    def call(self, inputs, training=None):
        print("[DEBUG] initializing {call GAIL}")
        if training == None:
            training = True

        if not self._enable_sn:
            features = self.l1(inputs)
            features = self.l2(features)
            return self.l3(features)
        else:
            features = self.l1(inputs, training=training)
            features = self.l2(features, training=training)
            return self.l3(features, training=training)

    def compute_reward(self, inputs):
        print("[DEBUG] initializing {compute_reward GAIL}")
        if not self._enable_sn:
            return tf.math.log(self(inputs) + 1e-8)
            # return tf.math.log(self(inputs) + 1e-8) - \
            #        tf.math.log(1 - self(inputs) + 1e-8)
            # # return self(inputs)
        else:
            return tf.math.log(self(inputs , training=False) + 1e-8)
            # return tf.math.log(self(inputs, training=False) + 1e-8) - \
            #        tf.math.log(1 - self(inputs, training=False) + 1e-8)
            # return self(inputs, training=False)

class GAIL(IRLPolicy):
    def __init__(
            self,
            state_shape,
            action_dim,
            units=[32, 32],
            lr=0.001,
            enable_sn=False,
            name="GAIL",
            is_debug = False,
            n_training = 1,
            grad_penalty_coeff = 0,
            **kwargs):
        super().__init__(name=name, n_training=n_training, **kwargs)
        self._is_debug = is_debug
        self.grad_penalty_coeff = grad_penalty_coeff
        self._is_grad_penalty = grad_penalty_coeff > 0
        self.disc = Discriminator(state_shape=state_shape, action_dim=action_dim,
                                  units=units, enable_sn=enable_sn)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5)
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def train(self, agent_states, agent_acts, expert_states, expert_acts, **kwargs):
        loss, accuracy, js_divergence = self._train_body(agent_states, agent_acts, expert_states, expert_acts)
        tf.summary.scalar(name=self.policy_name+"/DiscriminatorLoss", data=loss)
        tf.summary.scalar(name=self.policy_name+"/Accuracy", data=accuracy)
        tf.summary.scalar(name=self.policy_name+"/JSdivergence", data=js_divergence)
        if self._is_debug:
            [tf.summary.histogram(name=self.policy_name + "/weights_shape:{}".format(i.shape), data=i) for i in self.disc.get_weights()]
    @tf.function
    def _compute_js_divergence(self, fake_logits, real_logits):
        # fake_logits = tf.sigmoid(fake_logits)
        # real_logits = tf.sigmoid(real_logits)
        m = (fake_logits + real_logits) / 2.
        return tf.reduce_mean((
            fake_logits * tf.math.log(fake_logits / m + 1e-8) + real_logits * tf.math.log(real_logits / m + 1e-8)) / 2.)

    @tf.function
    def _train_body(self, agent_states, agent_acts, expert_states, expert_acts):
        print("[DEBUG] initializing {_train_body GAIL}...")
        epsilon = 1e-8
        with tf.device(self.device):
            # grad penalty
            # alpha = tf.random.uniform(shape=(agent_inputs.get_shape()[0], 1))
            # inter = alpha * agent_inputs + (1 - alpha) * expert_inputs
            with tf.GradientTape() as tape:

                real_logits = self.disc(tf.concat((expert_states, expert_acts), axis=1))
                fake_logits = self.disc(tf.concat((agent_states, agent_acts), axis=1))

                if self._is_grad_penalty:
                    inter = tf.concat((agent_states, agent_acts), axis=1)
                    with tf.GradientTape(watch_accessed_variables=False) as tape2:
                        tape2.watch(inter)
                        output = self.disc(inter)
                    grad = tape2.gradient(output, [inter])[0]
                    grad_penalty = tf.reduce_mean(tf.pow(tf.norm(grad, axis=-1), 2))
                    loss = -(tf.reduce_mean(tf.math.log(real_logits + epsilon)) +
                             tf.reduce_mean(tf.math.log(1. - fake_logits + epsilon)) -
                             self.grad_penalty_coeff * grad_penalty)
                else:
                    loss = -(tf.reduce_mean(tf.math.log(real_logits + epsilon)) +
                             tf.reduce_mean(tf.math.log(1. - fake_logits + epsilon)))
                # loss = tfgan_losses.modified_discriminator_loss(real_logits, fake_logits, label_smoothing=0.0)
            grads = tape.gradient(loss, self.disc.trainable_variables)
            self.optimizer.apply_gradients( zip(grads, self.disc.trainable_variables))

        accuracy = (tf.reduce_mean(tf.cast(real_logits >= 0.5, tf.float32)) / 2. +
                    tf.reduce_mean(tf.cast(fake_logits < 0.5, tf.float32)) / 2.)
        js_divergence = self._compute_js_divergence(fake_logits, real_logits)
        return loss, accuracy, js_divergence

    def inference(self, states, actions, next_states=None):
        if states.ndim == actions.ndim == 1:
            states = np.expand_dims(states, axis=0)
            actions = np.expand_dims(actions, axis=0)
        inputs = np.concatenate((states, actions), axis=1)
        return self._inference_body(inputs)

    @tf.function
    def _inference_body(self, inputs):
        print("[DEBUG] initializing {_inference_body}...")
        with tf.device(self.device):
            return self.disc.compute_reward(inputs)

    @staticmethod
    def get_argument(parser=None):
        parser = IRLPolicy.get_argument(parser)
        # parser.add_argument('--enable-sn', action='store_true')
        return parser
