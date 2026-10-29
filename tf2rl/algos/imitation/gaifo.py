import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from tf2rl.algos.base import RewardLearner
from tf2rl.networks.spectral_normalization import SpectralNormalization


class Discriminator(tf.keras.Model):
    """Distinguish expert and agent state transitions, optionally through a latent bottleneck."""
    def __init__(self, state_shape, units=(32, 32), dropout_rate=0.,
                 enable_sn=False, output_activation="sigmoid",
                 name="Discriminator"):
        """Build the expert-versus-agent discriminator with the requested architecture.

        Args:
            state_shape (tuple[int, ...]): Shape of one observation, excluding the batch axis. This
                implementation does not read this argument.
            units (sequence[int]): Hidden-layer widths in forward order.
            dropout_rate (float): Dropout probability used by the discriminator.
            enable_sn (bool): Request spectral normalization where the selected implementation supports
                it.
            output_activation (str or callable or None): Activation used by the scalar or discriminator
                output layer.
            name (str): Name assigned to the model, operation, or registered component.

        Returns:
            None: Initializes the instance.

        Raises:
            NotImplementedError: The requested operation or argument combination is unsupported by this
                component.
        """
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

        """Return discriminator outputs for the supplied batch of inputs.

        Args:
            inputs (tf.Tensor): Batched discriminator inputs with concatenated state/action or
                state/successor coordinates as required by the model.
            training (bool or tf.Tensor or None): Keras training flag controlling stochastic or stateful
                layers; None follows the layer default.

        Returns:
            tf.Tensor: Scalar discriminator or reward outputs shaped (batch, 1), with the configured
                output activation.
        """
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
        """Return stabilized discriminator log odds for each input sample.

        Args:
            inputs (tf.Tensor): Batched discriminator inputs with concatenated state/action or
                state/successor coordinates as required by the model.

        Returns:
            tf.Tensor: State-transition discriminator log-odds rewards shaped (batch, 1), using the
                configured online/target model.
        """
        print("[DEBUG] initializing {compute_reward GAIfO}")
        return tf.math.log(self(inputs) + 1e-8) - \
               tf.math.log(1 - self(inputs) + 1e-8)

class GAIfO(RewardLearner):
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
        """Configure the imitation objective, discriminator networks, and optimizer settings.

        Args:
            state_shape (tuple[int, ...]): Shape of one observation, excluding the batch axis.
            units (sequence[int]): Hidden-layer widths in forward order.
            lr (float): Optimizer learning rate.
            enable_sn (bool): Request spectral normalization where the selected implementation supports
                it.
            dropout_rate (float): Dropout probability used by the discriminator.
            name (str): Name assigned to the model, operation, or registered component.
            **kwargs (dict): Additional keyword arguments forwarded to RewardLearner construction.

        Returns:
            None: Initializes the instance.
        """
        super().__init__(name=name, n_training=1, **kwargs)

        self.disc = Discriminator(state_shape=state_shape, units=units, enable_sn=enable_sn, dropout_rate=dropout_rate)

        self.optimizer = tf.keras.optimizers.Adam( learning_rate=lr, beta_1=0.5)

    def train(self, agent_states, agent_next_states,
              expert_states, expert_next_states, **kwargs):
        """Update the discriminator from agent/expert minibatches and write training
        diagnostics to TensorBoard.

        Args:
            agent_states (array_like): Agent observation batch shaped (agent_batch, *state_shape).
            agent_next_states (array_like): Agent successor observations aligned with agent_states.
            expert_states (array_like): Expert observation batch shaped (expert_batch, *state_shape).
            expert_next_states (array_like): Expert successor observations aligned with expert_states.
            **kwargs (dict): Additional compatibility keyword options, currently unused.

        Returns:
            None.
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
        """Return the historical Jensen-Shannon diagnostic for agent and expert outputs.

        Args:
            fake_logits (tf.Tensor): Agent discriminator outputs used directly in the probability-based
                divergence calculation.
            real_logits (tf.Tensor): Expert discriminator outputs used directly in the probability-based
                divergence calculation.

        Returns:
            tf.Tensor: Scalar stabilized probability-based Jensen-Shannon diagnostic between agent and
                expert outputs.
        """
        m = (fake_logits + real_logits) / 2.
        return tf.reduce_mean((
            fake_logits * tf.math.log(fake_logits / m + 1e-8) + real_logits * tf.math.log(real_logits / m + 1e-8)) / 2.)

    @tf.function
    def _train_body(self, agent_states, agent_next_states, expert_states, expert_next_states):
        """Apply discriminator gradients for the expert and agent batches and return training
        metrics.

        Args:
            agent_states (array_like): Agent observation batch shaped (agent_batch, *state_shape).
            agent_next_states (array_like): Agent successor observations aligned with agent_states.
            expert_states (array_like): Expert observation batch shaped (expert_batch, *state_shape).
            expert_next_states (array_like): Expert successor observations aligned with expert_states.

        Returns:
            tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Scalar discriminator loss, classification accuracy,
                and JS diagnostic after applying gradients.
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
        """Return learned imitation rewards for the supplied observations and actions or
        transitions.

        Args:
            states (array_like): Observation batch shaped (batch, *state_shape).
            actions (array_like or None): Compatibility action argument; rewards depend on states and
                next_states instead. This implementation does not read this argument.
            next_states (array_like): Physical successor observations aligned with the current-state
                batch.

        Returns:
            tf.Tensor: State-transition discriminator log-odds rewards shaped (batch, 1), using the
                configured online/target model.
        """
        assert states.shape == next_states.shape
        if states.ndim == 1:
            states = np.expand_dims(states, axis=0)
            next_states = np.expand_dims(next_states, axis=0)
        inputs = np.concatenate((states, next_states), axis=1)
        return self._inference_body(inputs)

    @tf.function
    def _inference_body(self, inputs):
        """Return learned imitation rewards for the supplied observations and actions or
        transitions.

        Args:
            inputs (tf.Tensor): Batched discriminator inputs with concatenated state/action or
                state/successor coordinates as required by the model.

        Returns:
            tf.Tensor: State-transition discriminator log-odds rewards shaped (batch, 1), using the
                configured online/target model.
        """
        print("[DEBUG] initializing {_inference_body GAIfO}...")
        with tf.device(self.device):
            return self.disc.compute_reward(inputs)

    @staticmethod
    def get_argument(parser=None):
        """Add this component's command-line options and return the argument parser.

        Args:
            parser (argparse.ArgumentParser or None): Parser to extend; None constructs a new parser.

        Returns:
            argparse.ArgumentParser: Parser extended with the supported component options.
        """
        parser = RewardLearner.get_argument(parser)
        return parser
