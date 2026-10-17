import numpy as np
import tensorflow as tf
from tf2rl.networks.spectral_normalization import SpectralNormalization
from tensorflow.keras.layers import Dense

from tf2rl.algos.policy_base import IRLPolicy


class Discriminator(tf.keras.Model):
    """Distinguish expert and agent state transitions, optionally through a latent bottleneck."""
    def __init__(self, state_shape, units=(32, 32), dropout_rate=0.,
                 enable_sn=False, output_activation="sigmoid",
                 name="Discriminator"):
        """Build the expert-versus-agent discriminator with the requested architecture."""
        tf.keras.Model.__init__(self, name=name)
        self._enable_sn = enable_sn
        self._dropout_rate = dropout_rate
        assert dropout_rate >= 0 and dropout_rate <=1, "The drop out rate should be between 0 and 1. it is {}".format(dropout_rate)

        initializer = 'orthogonal'

        if not enable_sn:
            self.l1 = Dense(units[0], name="L1", activation="relu", kernel_initializer=initializer)
            self.l2 = Dense(units[1], name="L2", activation="relu", kernel_initializer=initializer)
            self.l3 = Dense(1, name="L3", activation=output_activation, kernel_initializer=initializer)
        else:
            print("[DEBUG] Spectral Normalization is enabled")
            raise NotImplementedError
            self.l1 = SpectralNormalization(Dense(units[0], name="L1", activation="relu",
                                                             kernel_initializer=initializer))
            self.l2 = SpectralNormalization(Dense(units[1], name="L2", activation="relu",
                                                             kernel_initializer=initializer))
            self.l3 = SpectralNormalization(Dense(1, name="L3", activation=output_activation,
                                                             kernel_initializer=initializer))


    @tf.function
    def call(self, inputs, training=None):

        """Return discriminator outputs for the supplied batch of inputs."""
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
    @tf.function
    def compute_reward(self, inputs):
        """Return stabilized discriminator log odds for each input sample."""
        print("[DEBUG] initializing {compute_reward GAIfO}")
        return tf.math.log(self(inputs) + 1e-8) - \
               tf.math.log(1 - self(inputs) + 1e-8)

class GAIfO(IRLPolicy):
    """Adversarial imitation from observations using state-transition discrimination."""
    def __init__(
            self,
            state_shape,
            units=(32, 32),
            lr=0.001,
            enable_sn=False,
            dropout_rate=0.,
            name="GAIfO",
            **kwargs):
        """Configure the imitation objective, discriminator networks, and optimizer settings."""
        super().__init__(name=name, n_training=1, **kwargs)

        self.disc = Discriminator(state_shape=state_shape, units=units, enable_sn=enable_sn, dropout_rate=dropout_rate)

        self.optimizer = tf.keras.optimizers.Adam( learning_rate=lr, beta_1=0.5)

    def train(self, agent_states, agent_next_states,
              expert_states, expert_next_states, **kwargs):
        """
        Update the imitation discriminator from agent and expert minibatches and report diagnostics.
        """
        loss, accuracy, js_divergence = self._train_body(agent_states,
                                                         agent_next_states,
                                                         expert_states,
                                                         expert_next_states)


        tf.summary.scalar(name=self.policy_name+"/DiscriminatorLoss", data=loss)
        tf.summary.scalar(name=self.policy_name+"/Accuracy", data=accuracy)
        tf.summary.scalar(name=self.policy_name+"/JSdivergence", data=js_divergence)

    @tf.function
    def _compute_js_divergence(self, fake_logits, real_logits):
        """Return the historical Jensen-Shannon diagnostic for agent and expert outputs."""
        m = (fake_logits + real_logits) / 2.
        return tf.reduce_mean((
            fake_logits * tf.math.log(fake_logits / m + 1e-8) + real_logits * tf.math.log(real_logits / m + 1e-8)) / 2.)

    @tf.function
    def _train_body(self, agent_states, agent_next_states, expert_states, expert_next_states):
        """
        Apply discriminator gradients for the expert and agent batches and return training metrics.
        """
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
        """
        Return learned imitation rewards for the supplied observations and actions or transitions.
        """
        assert states.shape == next_states.shape
        if states.ndim == 1:
            states = np.expand_dims(states, axis=0)
            next_states = np.expand_dims(next_states, axis=0)
        inputs = np.concatenate((states, next_states), axis=1)
        return self._inference_body(inputs)

    @tf.function
    def _inference_body(self, inputs):
        """
        Return learned imitation rewards for the supplied observations and actions or transitions.
        """
        print("[DEBUG] initializing {_inference_body GAIfO}...")
        with tf.device(self.device):
            return self.disc.compute_reward(inputs)

    @staticmethod
    def get_argument(parser=None):
        """Add this component's command-line options and return the argument parser."""
        parser = IRLPolicy.get_argument(parser)
        return parser
