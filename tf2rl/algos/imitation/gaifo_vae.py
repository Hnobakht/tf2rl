import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from tf2rl.algos.base import RewardLearner


class Discriminator(tf.keras.Model):
    """Distinguish expert and agent state transitions, optionally through a latent bottleneck."""
    LOG_SIG_CAP_MAX = 2  # np.e**2 = 7.389
    LOG_SIG_CAP_MIN = -20  # np.e**-10 = 4.540e-05
    EPS = 1e-6

    def __init__(self, state_shape, action_dim, units=(32, 32), decoder_units = (1,),
                 n_latent_unit=32, enable_sn=False, name="Discriminator", w_ini = 'orthogonal'):
        """Build the expert-versus-agent discriminator with the requested architecture.

        Args:
            state_shape (tuple[int, ...]): Shape of one observation, excluding the batch axis.
            action_dim (int): Number of discrete choices or width of a continuous action vector. This
                implementation does not read this argument.
            units (sequence[int]): Hidden-layer widths in forward order.
            decoder_units (sequence[int] or int): Hidden widths used by the variational discriminator
                decoder.
            n_latent_unit (int): Number of latent coordinates in the variational discriminator.
            enable_sn (bool): Request spectral normalization where the selected implementation supports
                it.
            name (str): Name assigned to the model, operation, or registered component.
            w_ini (str or keras.initializers.Initializer): Initializer for discriminator dense kernels.

        Returns:
            None: Initializes the instance.

        Raises:
            NotImplementedError: The requested operation or argument combination is unsupported by this
                component.
        """
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

    def _encoder_model(self, units, activation):
        """Build the encoder that predicts latent Gaussian parameters.

        Args:
            units (sequence[int]): Hidden-layer widths in forward order.
            activation (str or callable or None): Output activation; None leaves the affine output
                linear.

        Returns:
            None.
        """
        index=0
        for unit in units:
            self._encoder_list.append(Dense(unit, name="L_EN"+str(index),
                                            activation=activation,
                                            kernel_initializer=self.w_ini))
            index += 1

    def _decoder_model(self, units, activation, output_activation, output_shape):
        """Build the discriminator network that maps latent samples to outputs.

        Args:
            units (sequence[int]): Decoder layer widths; (1,) selects a direct scalar output.
            activation (str or callable or None): Output activation; None leaves the affine output
                linear.
            output_activation (str or callable or None): Activation used by the scalar or discriminator
                output layer.
            output_shape (tuple[int, ...]): Output dimensions used to configure the discriminator head.

        Returns:
            None.
        """
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

    def call(self, inputs, training):

        """Encode inputs, sample latent features, and return discriminator outputs and latent
        statistics.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor]): State batch (including its final absorbing indicator)
                and aligned action or successor-state batch.
            training (bool or tf.Tensor or None): Keras training flag controlling stochastic or stateful
                layers; None follows the layer default.

        Returns:
            tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Scalar discriminator outputs, latent means, and
                latent log standard deviations. In evaluation mode some historical variants return the
                mean in both latent-statistic slots.
        """
        features = inputs
        for layer in self._encoder_list:
            features = layer(features)

        if training == True:
            print("[DEBUG] initializing Training True {call Discriminator-VAIL}")
            means = self.l_mean(features)
            logstds = self.l_logstd(features)
            logstds = tf.clip_by_value(logstds, self.LOG_SIG_CAP_MIN, self.LOG_SIG_CAP_MAX)
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

        return latents, means, logstds

    @tf.function
    def compute_reward(self, inputs):
        """Return stabilized discriminator log odds for each input sample.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor]): State batch (including its final absorbing indicator)
                and aligned action or successor-state batch.

        Returns:
            tf.Tensor: State-transition discriminator log-odds rewards shaped (batch, 1), using the
                configured online/target model.
        """
        print("[DEBUG] initializing {compute_reward GAIfO}")
        return tf.math.log(self(inputs) + 1e-8) - \
               tf.math.log(1 - self(inputs) + 1e-8)

class GAIfO_VAE(RewardLearner):
    """Observation-only adversarial imitation with a variational discriminator."""
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
