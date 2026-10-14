import numpy as np
import tensorflow as tf
from tf2rl.networks.spectral_normalization import SpectralNormalization
from tensorflow.keras.layers import Dense, Dropout
import tensorflow_probability as tfp

from tf2rl.algos.policy_base import IRLPolicy
# from tf2rl.algos.gail import GAIL, Discriminator as DiscriminatorGAIL
# from tf2rl.networks.spectral_norm_dense import SNDense


class Discriminator(tf.keras.Model):
    LOG_SIG_CAP_MAX = 2  # np.e**2 = 7.389
    LOG_SIG_CAP_MIN = -20  # np.e**-10 = 4.540e-05
    EPS = 1e-6

    def __init__(self, state_shape, action_dim, units=(32, 32), decoder_units = (1,),
                 n_latent_unit=32, enable_sn=False, name="Discriminator", w_ini = 'orthogonal'):
        super().__init__(name=name)
        self._encoder_list = []
        self._decoder_list = []
        self.w_ini =w_ini

        activation = "relu"
        v_activation = "linear"
        output_activation = "linear"
        if not enable_sn:
            self._encoder_model(units, activation)
            self.l_mean = Dense(n_latent_unit, name="L_mean", activation=v_activation, kernel_initializer=w_ini)
            self.l_logstd = Dense(n_latent_unit, name="L_std", activation=v_activation, kernel_initializer=w_ini)
            self._decoder_model(decoder_units, activation, output_activation, state_shape)
        else:
            raise NotImplementedError

        # dummy_state = tf.constant(
        #     np.zeros(shape=(1,)+state_shape, dtype=np.float32))
        # dummy_action = tf.constant(
        #     np.zeros(shape=[1, action_dim], dtype=np.float32))
        # with tf.device("/cpu:0"):
        #     self([dummy_state, dummy_action])
    def _encoder_model(self, units, activation):
        index=0
        for unit in units:
            self._encoder_list.append(Dense(unit, name="L_EN"+str(index),
                                            activation=activation,
                                            kernel_initializer=self.w_ini))
            index += 1

    def _decoder_model(self, units, activation, output_activation, output_shape):
        if  not units[0] == 1:
            index = 0
            for unit in units:
                self._decoder_list.append(Dense(unit, name="L_DE"+str(index),
                                                activation=activation,
                                                kernel_initializer=self.w_ini))
                index += 1
            self._decoder_list.append(Dense(1, name="L_DE_out",
                                            activation=output_activation,
                                            kernel_initializer=self.w_ini))
        else:
            print("[WARNING] Discriminator has one dimention!")
            self._decoder_list.append(Dense(output_shape, name="L_DE_out",
                                            activation=output_activation,
                                            kernel_initializer=self.w_ini))

    # @tf.function
    def call(self, inputs, training):
        # with tf.device("/gpu:0"):
        # Encoder
        # features = tf.concat(inputs, axis=1)
        # if training == None:
        #     raise ValueError

        # features = tf.concat((inputs[0][:,:-1], inputs[1]), axis=1)
        features = inputs
        for layer in self._encoder_list:
            features = layer(features)

        if training == True:
            print("[DEBUG] initializing Training True {call Discriminator-VAIL}")
            means = self.l_mean(features)
            logstds = self.l_logstd(features)
            logstds = tf.clip_by_value(logstds, self.LOG_SIG_CAP_MIN, self.LOG_SIG_CAP_MAX)
            # latents = means + tf.random.normal(shape=means.shape) * tf.math.exp(logstds)
            latents = means + tf.random.normal(shape=(tf.shape(means)[0],tf.shape(means)[1])) * tf.math.exp(logstds)
        else:
            print("[DEBUG] initializing Training False {call Discriminator-VAIL}")
            latents = self.l_mean(features)
            means = latents
            logstds = latents

        latents = tf.concat((latents, tf.expand_dims(inputs[0][:,-1], axis=1)), axis=1)
        # (Not) Binary classifier
        for layer in self._decoder_list:
            latents = layer(latents)
        # out = latents

        return latents, means, logstds

    @tf.function
    def compute_reward(self, inputs):
        print("[DEBUG] initializing {compute_reward GAIfO}")
        # if not self._enable_sn:
        # return tf.math.log(self(inputs) + 1e-8)
        return tf.math.log(self(inputs) + 1e-8) - \
               tf.math.log(1 - self(inputs) + 1e-8)
        # else:
        #     raise NotImplementedError
        #     return tf.math.log(self(inputs, training=False) + 1e-8)

class GAIfO_VAE(IRLPolicy):
    def __init__(
            self,
            state_shape,
            units=(32, 32),
            lr=0.001,
            enable_sn=False,
            dropout_rate=0.,
            name="GAIfO",
            **kwargs):
        super().__init__(name=name, n_training=1, **kwargs)

        self.disc = Discriminator(state_shape=state_shape, units=units, enable_sn=enable_sn, dropout_rate=dropout_rate)

        self.optimizer = tf.keras.optimizers.Adam( learning_rate=lr, beta_1=0.5)

    def train(self, agent_states, agent_next_states,
              expert_states, expert_next_states, **kwargs):
        loss, accuracy, js_divergence = self._train_body(agent_states,
                                                         agent_next_states,
                                                         expert_states,
                                                         expert_next_states)

        tf.summary.scalar(name=self.policy_name+"/DiscriminatorLoss", data=loss)
        tf.summary.scalar(name=self.policy_name+"/Accuracy", data=accuracy)
        tf.summary.scalar(name=self.policy_name+"/JSdivergence", data=js_divergence)

    @tf.function
    def _compute_js_divergence(self, fake_logits, real_logits):
        # fake_logits = tf.sigmoid(fake_logits)
        # real_logits = tf.sigmoid(real_logits)
        m = (fake_logits + real_logits) / 2.
        return tf.reduce_mean((
            fake_logits * tf.math.log(fake_logits / m + 1e-8) + real_logits * tf.math.log(real_logits / m + 1e-8)) / 2.)

    @tf.function
    def _train_body(self, agent_states, agent_next_states, expert_states, expert_next_states):
        epsilon = 1e-8
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                real_logits = self.disc(tf.concat((expert_states, expert_next_states), axis=1))
                fake_logits = self.disc(tf.concat((agent_states, agent_next_states), axis=1))
                loss = -(tf.reduce_mean(tf.math.log(real_logits + epsilon)) +
                         tf.reduce_mean(tf.math.log(1. - fake_logits + epsilon)))
            grads = tape.gradient(loss, self.disc.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.disc.trainable_variables))

        accuracy = (tf.reduce_mean(tf.cast(real_logits >= 0.5, tf.float32)) / 2. +
                    tf.reduce_mean(tf.cast(fake_logits < 0.5, tf.float32)) / 2.)
        js_divergence = self._compute_js_divergence(
            fake_logits, real_logits)
        return loss, accuracy, js_divergence

    def inference(self, states, actions, next_states):
        assert states.shape == next_states.shape
        if states.ndim == 1:
            states = np.expand_dims(states, axis=0)
            next_states = np.expand_dims(next_states, axis=0)
        inputs = np.concatenate((states, next_states), axis=1)
        return self._inference_body(inputs)

    @tf.function
    def _inference_body(self, inputs):
        print("[DEBUG] initializing {_inference_body GAIfO}...")
        with tf.device(self.device):
            return self.disc.compute_reward(inputs)

    @staticmethod
    def get_argument(parser=None):
        parser = IRLPolicy.get_argument(parser)
        # parser.add_argument('--enable-sn', action='store_true')
        return parser
