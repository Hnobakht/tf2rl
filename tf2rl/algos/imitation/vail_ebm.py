import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from tf2rl.algos.base import RewardLearner
from tf2rl.algos.imitation.vail import VAIL


class Discriminator(tf.keras.Model):
    """Variational discriminator with an encoded Gaussian information bottleneck."""
    LOG_SIG_CAP_MAX = 2  # np.e**2 = 7.389
    LOG_SIG_CAP_MIN = -20  # np.e**-10 = 4.540e-05
    EPS = 1e-6

    def __init__(self, state_shape, action_dim, units=(32, 32), decoder_units = (1,),
                 n_latent_unit=32, enable_sn=False, name="Discriminator", w_ini = 'orthogonal'):
        """Build the expert-versus-agent discriminator with the requested architecture.

        Args:
            state_shape (tuple[int, ...]): Shape of one observation, excluding the batch axis. This
                implementation does not read this argument.
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
        output_activation = "sigmoid"
        if not enable_sn:
            self._encoder_model(units, activation)
            self.l_mean = Dense(n_latent_unit, name="L_mean", activation=v_activation, kernel_initializer=w_ini)
            self.l_logstd = Dense(n_latent_unit, name="L_std", activation=v_activation, kernel_initializer=w_ini)
            self._decoder_model(decoder_units, activation, output_activation)
        else:
            raise NotImplementedError

        dummy_state = tf.constant(np.zeros(shape=(1, 64), dtype=np.float32))
        # with tf.device("/cpu:0"):
        self(dummy_state, tf.constant(True))
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

    def _decoder_model(self, units, activation, output_activation):
        """Build the discriminator network that maps latent samples to outputs.

        Args:
            units (sequence[int]): Decoder layer widths; (1,) selects a direct scalar output.
            activation (str or callable or None): Output activation; None leaves the affine output
                linear.
            output_activation (str or callable or None): Activation used by the scalar or discriminator
                output layer.

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
            self._decoder_list.append(Dense(1, name="L_DE_out",
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
            print("[DEBUG] initializing Training True [call {} Discriminator-VAIL_EBM]".format(inputs.shape))
            means = self.l_mean(features)
            logstds = self.l_logstd(features)
            logstds = tf.clip_by_value(logstds, self.LOG_SIG_CAP_MIN, self.LOG_SIG_CAP_MAX)
            latents = means + tf.random.normal(shape=(tf.shape(means)[0],tf.shape(means)[1])) * tf.math.exp(logstds)
        else:
            print("[DEBUG] initializing Training False [call {} Discriminator-VAIL_EBM]".format(inputs.shape))
            latents = self.l_mean(features)
            means = latents
            logstds = latents

        for layer in self._decoder_list:
            latents = layer(latents)

        return latents, means, logstds

class InputModel(tf.keras.Model):
    """Feature model used by the variational energy-based discriminator."""
    def __init__(self, state_shape, action_dim, units, activation='relu', name="InputModel", w_ini='orthogonal', trainable=True):
        """Build the feature layers feeding the variational energy-based discriminator.

        Args:
            state_shape (tuple[int, ...]): Shape of one observation, excluding the batch axis.
            action_dim (int): Number of discrete choices or width of a continuous action vector.
            units (sequence[int]): Hidden-layer widths in forward order.
            activation (str or callable or None): Output activation; None leaves the affine output
                linear.
            name (str): Name assigned to the model, operation, or registered component.
            w_ini (str or keras.initializers.Initializer): Initializer for discriminator dense kernels.
            trainable (bool): Whether Keras should include this layer parameters in optimization.

        Returns:
            None: Initializes the instance.
        """
        super().__init__(name=name)
        self.layer_1 = Dense(units, name="L1_"+str(name),
                             activation=activation if trainable else 'sigmoid',
                             kernel_initializer=w_ini,
                             bias_initializer=tf.keras.initializers.RandomNormal,
                             trainable=trainable)
        self.layer_2 = Dense(int(units/2), name="L2_" + str(name),
                             activation=activation if trainable else 'sigmoid',
                             kernel_initializer=w_ini,
                             bias_initializer=tf.keras.initializers.RandomNormal,
                             trainable=trainable)

        dummy_state = tf.constant(
            np.zeros(shape=(1,)+state_shape, dtype=np.float32))
        dummy_action = tf.constant(
            np.zeros(shape=[1, action_dim], dtype=np.float32))
        # with tf.device("/cpu:0"):
        self((dummy_state, dummy_action))

    def call(self, inputs):
        """Transform inputs into features for the energy-based discriminator.

        Args:
            inputs (tf.Tensor): Batched discriminator inputs with concatenated state/action or
                state/successor coordinates as required by the model.

        Returns:
            tf.Tensor: Encoded input features preserving the leading batch dimension.
        """
        features = tf.concat(inputs, axis=1)
        print("[DEBUG] initializing InputModel {} [call {} Discriminator-VAIL_EBM]"
              .format(str(self.name) ,features.shape))
        features = self.layer_1(features)
        return self.layer_2(features)

class VAIL_EBM(VAIL):
    """Experimental variational imitation learner with an energy-based discriminator."""
    def __init__(
            self,
            state_shape,
            action_dim,
            units=(32, 32),
            decoder_units=(1),
            n_latent_unit=32,
            lr=5e-5,
            kl_target=0.5,
            reg_param=0.,
            enable_sn=False,
            grad_penalty_coeff=0.,
            n_training=1,
            name="VAIL_EBM",
            is_debug=False,
            **kwargs):
        """Configure the imitation objective, discriminator networks, and optimizer settings.

        Args:
            state_shape (tuple[int, ...]): Shape of one observation, excluding the batch axis.
            action_dim (int): Number of discrete choices or width of a continuous action vector.
            units (sequence[int]): Hidden-layer widths in forward order.
            decoder_units (sequence[int] or int): Hidden widths used by the variational discriminator
                decoder.
            n_latent_unit (int): Number of latent coordinates in the variational discriminator.
            lr (float): Optimizer learning rate.
            kl_target (float): Target KL divergence for the variational information bottleneck.
            reg_param (float): Initial multiplier for the variational KL constraint.
            enable_sn (bool): Request spectral normalization where the selected implementation supports
                it.
            grad_penalty_coeff (float): Coefficient multiplying the discriminator gradient penalty.
            n_training (int): Number of discriminator training attempts per scheduled generator update.
                This implementation does not read this argument.
            name (str): Name assigned to the model, operation, or registered component.
            is_debug (bool): Enable additional discriminator/parameter diagnostics where implemented.
            **kwargs (dict): Additional keyword arguments forwarded to VAIL construction.

        Returns:
            None: Initializes the instance.

        Raises:
            NotImplementedError: It is not working!.
        """
        RewardLearner.__init__(self, name=name, n_training=1, **kwargs)
        assert isinstance(decoder_units, tuple) and isinstance(units, tuple), \
            "[ERORR] the units should be tuple[__init__ VAIL_EBM]"
        raise NotImplementedError("It is not working!")

        # with tf.device(self.device):
        self.disc = Discriminator(state_shape=state_shape, action_dim=action_dim,
                                  units=units, n_latent_unit=n_latent_unit, decoder_units=decoder_units,
                                  enable_sn=enable_sn)

        self.expert_model = InputModel(state_shape=state_shape, action_dim=action_dim, units=units[0], name="expert_model", trainable = False)
        self.agent_model =  InputModel(state_shape=state_shape, action_dim=action_dim, units=units[0], name="agent_model", trainable = True)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.optimizer_agent = tf.keras.optimizers.Adam(learning_rate=3e-4)
        self._reg_param = tf.Variable(reg_param, dtype=tf.float32)
        self._step_reg_param = tf.constant(1e-5, dtype=tf.float32)

        self._grad_penalty_coeff = grad_penalty_coeff
        self._enable_gp = grad_penalty_coeff > 0
        self._kl_target = kl_target
        self._is_debug = is_debug

        self._init_static_graph(state_shape, action_dim, n_latent_unit)

    def train(self, agent_states, agent_acts, expert_states, expert_acts, **kwargs):
        # with tf.device(self.device):
        """Update the discriminator from agent/expert minibatches and write training
        diagnostics to TensorBoard.

        Args:
            agent_states (array_like): Agent observation batch shaped (agent_batch, *state_shape).
            agent_acts (array_like): Agent actions aligned with agent_states.
            expert_states (array_like): Expert observation batch shaped (expert_batch, *state_shape).
            expert_acts (array_like): Expert actions aligned with expert_states.
            **kwargs (dict): Additional compatibility keyword options, currently unused.

        Returns:
            None.
        """
        agent_states, agent_acts, expert_states, expert_acts = \
            tuple([tf.constant(i) for i in (agent_states, agent_acts, expert_states, expert_acts)])

        if self._is_debug: tf.summary.trace_on(graph=True)
        loss, accuracy, real_kl, fake_kl, js_divergence, loss_agent = self._train_body_graph(agent_states, agent_acts,
                                                                                 expert_states, expert_acts,
                                                                                 tf.constant(True))
        if self._is_debug: tf.summary.trace_export(name="_train_body_graph")

        tf.summary.scalar(name=self.policy_name+"/DiscriminatorLoss", data=loss)
        tf.summary.scalar(name=self.policy_name+"/Accuracy", data=accuracy)
        tf.summary.scalar(name=self.policy_name+"/RegParam", data=self._reg_param)
        tf.summary.scalar(name=self.policy_name+"/RealLatentKL", data=real_kl)
        tf.summary.scalar(name=self.policy_name+"/FakeLatentKL", data=fake_kl)
        tf.summary.scalar(name=self.policy_name+"/JSdivergence", data=js_divergence)
        tf.summary.scalar(name=self.policy_name + "/Agent's Loss", data=loss_agent)
        if self._is_debug:
            [tf.summary.histogram(name=self.policy_name + "/mean, {}".format(i.shape), data=i) for i in self.disc.l_mean.get_weights()]
            [tf.summary.histogram(name=self.policy_name + "/logstd, {}".format(i.shape), data=i) for i in self.disc.l_logstd.get_weights()]
            [tf.summary.histogram(name=self.policy_name + "/expert, {}".format(i.shape), data=i) for i in self.expert_model.get_weights()]
            [tf.summary.histogram(name=self.policy_name + "/agent, {}".format(i.shape), data=i) for i in self.agent_model.get_weights()]

    def _train_body(self, agent_states, agent_acts, expert_states, expert_acts, training):
        """Apply discriminator gradients for the expert and agent batches and return training
        metrics.

        Args:
            agent_states (array_like): Agent observation batch shaped (agent_batch, *state_shape).
            agent_acts (array_like): Agent actions aligned with agent_states.
            expert_states (array_like): Expert observation batch shaped (expert_batch, *state_shape).
            expert_acts (array_like): Expert actions aligned with expert_states.
            training (bool or tf.Tensor or None): Keras training flag controlling stochastic or stateful
                layers; None follows the layer default.

        Returns:
            tuple[tf.Tensor, ...]: Total loss, accuracy, expert/agent latent KL, JS diagnostic, and
                agent input-model loss after the model updates.
        """
        epsilon = 1e-8
        grad_out = None
        with tf.GradientTape() as tape, \
                tf.GradientTape() as tape2:
            # Compute discriminator loss
            real_features = tf.stop_gradient(self.expert_model((expert_states, expert_acts)))
            fake_fetures = self.agent_model((agent_states, agent_acts))
            expert_features = self.agent_model((agent_states, agent_acts))


            real_logits, real_means, real_logstds = self.disc(real_features, training)
            fake_logits, fake_means, fake_logstds = self.disc(fake_fetures, training)
            print("[DEBUG] initializing {_train_body VAIL_EBM}.")
            # Compute discriminator loss
            disc_loss = -(tf.reduce_mean(tf.math.log(real_logits + epsilon)) +
                          tf.reduce_mean(tf.math.log(1. - fake_logits + epsilon)))
            # Compute KL loss
            real_kl = tf.reduce_mean(self._compute_kl_latent_graph(real_means, real_logstds))
            fake_kl = tf.reduce_mean(self._compute_kl_latent_graph(fake_means, fake_logstds))
            kl_loss = 0.5 * (real_kl - self._kl_target +
                             fake_kl - self._kl_target)

            loss_agent = tf.reduce_mean((real_features-fake_fetures)**2)
            loss = disc_loss + self._reg_param * kl_loss + loss_agent


        grads = tape.gradient(loss, self.disc.trainable_variables+self.agent_model.trainable_variables)




        self.optimizer.apply_gradients(zip(grads, self.disc.trainable_variables+self.agent_model.trainable_variables))

        # Update reguralizer parameter \beta in eq.(9)
        self._reg_param.assign(tf.maximum(tf.constant(0., dtype=tf.float32),
                                          self._reg_param + self._step_reg_param * (kl_loss - self._kl_target)))

        accuracy = (tf.reduce_mean(tf.cast(real_logits >= 0.5, tf.float32)) / 2. +
                    tf.reduce_mean(tf.cast(fake_logits < 0.5, tf.float32)) / 2.)
        js_divergence = self._compute_js_divergence_graph(fake_logits, real_logits)


        return loss, accuracy, real_kl, fake_kl, js_divergence, loss_agent

    def _inference_body(self, states, actions, training):
        """Return learned imitation rewards for the supplied observations and actions or
        transitions.

        Args:
            states (array_like): Observation batch shaped (batch, *state_shape).
            actions (array_like): Actions aligned with the observation batch; continuous actions have a
                final action_dim axis.
            training (bool or tf.Tensor or None): Keras training flag controlling stochastic or stateful
                layers; None follows the layer default.

        Returns:
            tf.Tensor: Stabilized discriminator log-odds rewards shaped (batch, 1). Evaluation uses the
                latent mean rather than a sampled latent.
        """
        print("[DEBUG] initializing {_inference_body VAIL_EBM}")
        false_val = tf.constant(False)
        sig, _, _ = self.disc(self.agent_model((states, actions)), training=training)
        out = tf.math.log(sig + 1e-8) - tf.math.log(1 - sig + 1e-8)
        return out

    def _init_static_graph(self, state_shape, action_dim, n_latent_unit):
        """Set up TensorFlow signatures and traced model, training, or inference computations.

        Args:
            state_shape (tuple[int, ...]): Shape of one observation, excluding the batch axis.
            action_dim (int): Number of discrete choices or width of a continuous action vector.
            n_latent_unit (int): Number of latent coordinates in the variational discriminator.

        Returns:
            None.
        """
        print("[DEBUG] initializing {_init_static_graph VAIL_EBM}")
        # Compiling the static graphs
        assert len(state_shape) == 1, "[ERROR] This SAC only supports RAM env {_init_static_graph VAIL_EBM}"
        # with tf.device(self.device):
        # TODO: the @tf.function on top is better!
        self.disc.call = tf.function(self.disc.call,
                                     input_signature=[
                                         tf.TensorSpec(shape=(None, 64), dtype=tf.float32),
                                         tf.TensorSpec(shape=(None), dtype=tf.bool)]
                                     )

        self.expert_model.call = tf.function(self.expert_model.call,
                                             input_signature=[
                                                 (tf.TensorSpec(shape=(None, state_shape[-1]), dtype=tf.float32),
                                                  tf.TensorSpec(shape=(None, action_dim), dtype=tf.float32))]
                                             )

        self.agent_model.call = tf.function(self.agent_model.call,
                                             input_signature=[
                                                 (tf.TensorSpec(shape=(None, state_shape[-1]), dtype=tf.float32),
                                                  tf.TensorSpec(shape=(None, action_dim), dtype=tf.float32))]
                                             )

        self._train_body_graph = tf.function(self._train_body,
                                             input_signature=[
                                                 tf.TensorSpec(shape=(None, state_shape[-1]), dtype=tf.float32),
                                                 tf.TensorSpec(shape=(None, action_dim), dtype=tf.float32),
                                                 tf.TensorSpec(shape=(None, state_shape[-1]), dtype=tf.float32),
                                                 tf.TensorSpec(shape=(None, action_dim), dtype=tf.float32),
                                                 tf.TensorSpec(shape=(None), dtype=tf.bool)],
                                             )

        self._inference_body_graph = tf.function(self._inference_body,
                                                 input_signature=[tf.TensorSpec(shape=(None, state_shape[-1]), dtype=tf.float32),
                                                                  tf.TensorSpec(shape=(None, action_dim), dtype=tf.float32),
                                                                  tf.TensorSpec(shape=(None), dtype=tf.bool)],
                                                 )

        self._compute_kl_latent_graph = tf.function(self._compute_kl_latent,
                                                    input_signature=[tf.TensorSpec(shape=(None, n_latent_unit), dtype=tf.float32),
                                                                     tf.TensorSpec(shape=(None, n_latent_unit), dtype=tf.float32)],
                                                    )

        self._compute_js_divergence_graph = tf.function(self._compute_js_divergence,
                                                        input_signature=[tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                                                         tf.TensorSpec(shape=(None, 1), dtype=tf.float32)],
                                                        )
