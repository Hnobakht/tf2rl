import numpy as np
import tensorflow as tf
from tf2rl.networks.spectral_normalization import SpectralNormalization
from tensorflow.keras.layers import Dense, Dropout
import tensorflow_probability as tfp

from tf2rl.algos.policy_base import IRLPolicy
from tf2rl.algos.gail import GAIL, Discriminator as DiscriminatorGAIL
# from tf2rl.networks.spectral_norm_dense import SNDense


class Discriminator(tf.keras.Model):
    def __init__(self, state_shape, units=(32, 32), dropout_rate=0.,
                 enable_sn=False, output_activation="sigmoid",
                 name="Discriminator"):
        tf.keras.Model.__init__(self, name=name)
        self._enable_sn = enable_sn
        self._dropout_rate = dropout_rate
        assert dropout_rate >= 0 and dropout_rate <=1, "The drop out rate should be between 0 and 1. it is {}".format(dropout_rate)

        # initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=2e-3)
        # initializer = tf.keras.initializers.glorot_uniform()
        initializer = 'orthogonal'

        if not enable_sn:
            self.l1 = Dense(units[0], name="L1", activation="relu", kernel_initializer=initializer)
            # self.l1_dout = Dropout(dropout_rate)
            self.l2 = Dense(units[1], name="L2", activation="relu", kernel_initializer=initializer)
            # self.l2_dout = Dropout(dropout_rate)
            self.l3 = Dense(1, name="L3", activation=output_activation, kernel_initializer=initializer)
        else:
            print("[DEBUG] Spectral Normalization is enabled")
            self.l1 = SpectralNormalization(Dense(units[0], name="L1", activation="relu",
                                                             kernel_initializer=initializer))
            # self.l1_dout = Dropout(dropout_rate)
            self.l2 = SpectralNormalization(Dense(units[1], name="L2", activation="relu",
                                                             kernel_initializer=initializer))
            # self.l2_dout = Dropout(dropout_rate)
            self.l3 = SpectralNormalization(Dense(1, name="L3", activation=output_activation,
                                                             kernel_initializer=initializer))

        dummy_state = tf.constant(
            np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        dummy_next_state = tf.constant(
            np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        with tf.device("/gpu:0"):
            self(tf.concat((dummy_state, dummy_next_state), axis=1))

    def call(self, inputs, training=None):

        if training == None:
            training = True

        if not self._enable_sn:
            features = self.l1(inputs)
            features = self.l2(features)
            return self.l3(features)
        else:
            features = self.l1(inputs, training=training)
            # features = self.l1_dout(features, training=training)
            features = self.l2(features, training=training)
            # features = self.l2_dout(features, training=training)
            return self.l3(features, training=training)

    def compute_reward(self, inputs):
        print("[DEBUG] initializing {compute_reward GAIfO}")
        if not self._enable_sn:
            return tf.math.log(self(inputs) + 1e-8)
        else:
            return tf.math.log(self(inputs, training=False) + 1e-8)
'''
class SoftDiscriminator(tf.keras.Model):
    def __init__(self, state_shape, units=(300, 400),
                 enable_sn=True, output_activation="sigmoid",
                 name="Discriminator"):
        super().__init__(name=name)

        if not enable_sn:
            raise NotImplemented

        print("[DEBUG] Spectral Normalization is enabled")
        self.l1 = SpectralNormalization(Dense(units[0], name="L1", activation="relu"))
        self.l2 = SpectralNormalization(Dense(units[1], name="L2", activation="relu"))

        self.out_mean = SpectralNormalization(Dense(1, name="D_mean", activation=None))
        self.out_logstd = SpectralNormalization(Dense(1, name="D_logstd", activation=None))

        dummy_state = tf.constant(
            np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        dummy_next_state = tf.constant(
            np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        with tf.device("/gpu:0"):
            self(tf.concat((dummy_state, dummy_next_state), axis=1))

    def call(self, inputs, training=None):

        if training == None or training == True:
            training = True
        elif training == False:
            training = False
        else:
            raise NotImplemented

        features = self.l1(inputs, training=training)
        features = self.l2(features, training=training)

        mean_feature = self.out_mean(features, training=training)
        logstd_feature = self.out_logstd(features, training=training)
        logstd_feature = tf.clip_by_value(logstd_feature, -20., 2.)

        dist = tfp.distributions.MultivariateNormalDiag(loc=mean_feature, scale_diag=tf.exp(logstd_feature))

        if training:
            out = dist.sample()
        else:
            out = dist.mean()

        return tf.sigmoid(out)

    def compute_reward(self, inputs):
        print("[DEBUG] initializing {compute_reward GAIL}")
        return tf.math.log(self(inputs , training=False) + 1e-8)

class UniformDiscriminator(tf.keras.Model):
    def __init__(self, state_shape, units=(300, 400),
                 enable_sn=True, output_activation="sigmoid",
                 name="Discriminator"):
        super().__init__(name=name)

        if not enable_sn:
            raise NotImplemented

        print("[DEBUG] Spectral Normalization is enabled")
        self.l1 = SpectralNormalization(Dense(units[0], name="L1", activation="relu"))
        self.l2 = SpectralNormalization(Dense(units[1], name="L2", activation="relu"))

        self.out_low = SpectralNormalization(Dense(1, name="D_low", activation=None))
        self.out_delta = SpectralNormalization(Dense(1, name="D_delta", activation=None))

        dummy_state = tf.constant(
            np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        dummy_next_state = tf.constant(
            np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        with tf.device("/gpu:0"):
            self(tf.concat((dummy_state, dummy_next_state), axis=1))

    def call(self, inputs, training=None):

        if training == None or training == True:
            training = True
        elif training == False:
            training = False
        else:
            raise NotImplemented

        features = self.l1(inputs, training=training)
        features = self.l2(features, training=training)

        feature_low = self.out_low(features, training=training)
        feature_delta = self.out_delta(features, training=training)

        dist = tfp.distributions.Uniform(low=feature_low,
                                         high=feature_low + feature_delta)

        if training:
            out = dist.sample()
        else:
            out = dist.mean()

        return tf.sigmoid(out)

    def compute_reward(self, inputs):
        print("[DEBUG] initializing {compute_reward GAIL}")
        return tf.math.log(self(inputs , training=False) + 1e-8)
'''

class GAIfO(GAIL):
    def __init__(
            self,
            state_shape,
            units=(32, 32),
            lr=0.001,
            enable_sn=False,
            dropout_rate=0.,
            name="GAIfO",
            **kwargs):
        IRLPolicy.__init__(self, name=name, n_training=1, **kwargs)

        self.disc = Discriminator(state_shape=state_shape, units=units, enable_sn=enable_sn, dropout_rate=dropout_rate)

        self.optimizer = tf.keras.optimizers.Adam( learning_rate=lr, beta_1=0.5)

    def train(self, agent_states, agent_next_states,
              expert_states, expert_next_states, **kwargs):
        loss, accuracy, js_divergence = self._train_body(agent_states,
                                                         agent_next_states,
                                                         expert_states,
                                                         expert_next_states)

        # tf.summary.scalar(name="DEBUG/b1", data=self.disc.weights[7][0])
        # tf.summary.scalar(name="DEBUG/a1", data=self.disc.weights[8][0][0])
        # tf.summary.scalar(name="DEBUG/b2", data=self.disc.weights[10][0])
        # tf.summary.scalar(name="DEBUG/a2", data=self.disc.weights[11][0][0])

        tf.summary.scalar(name=self.policy_name+"/DiscriminatorLoss", data=loss)
        tf.summary.scalar(name=self.policy_name+"/Accuracy", data=accuracy)
        tf.summary.scalar(name=self.policy_name+"/JSdivergence", data=js_divergence)

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
