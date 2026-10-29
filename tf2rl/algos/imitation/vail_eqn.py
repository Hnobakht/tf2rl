import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from tf2rl.algos.base import RewardLearner
from tf2rl.networks.layers import DenseLayerv2 as DenseLayer
from tf2rl.networks.layers import EqlLayerv2 as EqlLayer


class Discriminator(tf.keras.Model):
    """Variational discriminator with an encoded Gaussian information bottleneck."""
    LOG_SIG_CAP_MAX = 2  # np.e**2 = 7.389
    LOG_SIG_CAP_MIN = -20  # np.e**-10 = 4.540e-05
    EPS = 1e-6
    def __init__(self, state_shape, action_dim, units=(32, 32), decoder_units = (1,),
                 n_latent_unit=32, enable_sn=False, name="Discriminator", w_ini = 'orthogonal',
                 v=None, exclude=None, drop_out = None):
        """Build the expert-versus-agent discriminator with the requested architecture.

        Args:
            state_shape (tuple[int, ...]): Shape of one observation, excluding the batch axis. This
                implementation does not read this argument.
            action_dim (int): Number of discrete choices or width of a continuous action vector. This
                implementation does not read this argument.
            units (sequence[int]): Hidden-layer widths in forward order.
            decoder_units (sequence[int] or int): Hidden widths used by the variational discriminator
                decoder. This implementation does not read this argument.
            n_latent_unit (int): Number of latent coordinates in the variational discriminator.
            enable_sn (bool): Request spectral normalization where the selected implementation supports
                it.
            name (str): Name assigned to the model, operation, or registered component.
            w_ini (str or keras.initializers.Initializer): Initializer for discriminator dense kernels.
            v (sequence[int] or int or None): Operator repetition counts, one per equation layer; an
                individual layer takes one count.
            exclude (sequence or None): Equation operators to exclude; network constructors use one
                exclusion list per layer.
            drop_out (float or sequence[float] or None): Dropout probability, or one probability per
                equation layer; None disables dropout.

        Returns:
            None: Initializes the instance.

        Raises:
            ValueError: [ERROR] V should be a tuple or list of the EQN network. [ERROR] dropout has too
                be a list.
            NotImplementedError: The requested operation or argument combination is unsupported by this
                component.
        """
        super().__init__(name=name)
        self._encoder_list = []
        self.eql_layers = []

        self._w_ini =w_ini
        self._drop_out = drop_out

        if v:
            if isinstance(v, tuple):
                v = list(v)
            self._v = v
            self._n_eqn_layer = len(v)
        else:
            raise ValueError("[ERROR] V should be a tuple or list of the EQN network")

        if drop_out:
            assert (np.array(drop_out) <= 1).all() and (np.array(drop_out) >= 0).all(), \
                "drop_out is a ratio and it needs to be between 0 and 1"
            if isinstance(drop_out, list):
                assert len(drop_out) == self._n_eqn_layer, \
                    "Lenth of dropout dosent match the number of the layers."
                self._drop_out = drop_out
            else:
                raise ValueError("[ERROR] dropout has too be a list")
        else:
            self._drop_out = None

        if exclude:
            assert len(exclude) == self._n_eqn_layer, "exclude parameter wrong format, len(exclude) must equal " \
                                                    "num_layers, ex.: exclude=[['sin'],[]], num_layers = 2"
            self._exclude = exclude
        else:
            exclude = [[] for i in range(self._n_eqn_layer)]
            self._exclude = exclude

        activation = tf.nn.relu
        v_activation = "linear"
        output_activation = tf.nn.sigmoid
        if not enable_sn:
            self._encoder_model(units, activation)
            self.l_mean = Dense(n_latent_unit, name="L_mean", activation=v_activation, kernel_initializer=w_ini)
            self.l_logstd = Dense(n_latent_unit, name="L_std", activation=v_activation, kernel_initializer=w_ini)
            self._decoder_model(output_activation)
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
                                            kernel_initializer=self._w_ini))
            index += 1

    def _decoder_model(self,output_activation):
        """Build the discriminator network that maps latent samples to outputs.

        Args:
            output_activation (str or callable or None): Activation used by the scalar or discriminator
                output layer.

        Returns:
            None.
        """
        initializer = 'random_normal'
        for index in range(self._n_eqn_layer):
            self.eql_layers.append(EqlLayer(w_initializer=initializer, b_initializer=initializer,
                                            v=self._v[index], exclude=self._exclude[index],
                                            constraint=None, lmbda=0, dynamic_lmbda=False))

            if self._drop_out and not isinstance(self._drop_out, list):
                self.eql_layers.append(tf.keras.layers.Dropout(self._drop_out))
            elif self._drop_out and isinstance(self._drop_out, list):
                if self._drop_out[index] > 0:
                    self.eql_layers.append(tf.keras.layers.Dropout(self._drop_out[index]))

        self.eql_layers.append(DenseLayer(w_initializer=initializer, b_initializer=initializer,
                                    constraint=None, lmbda=0, dynamic_lmbda=False,
                                    activation=output_activation))

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
        features = tf.concat((inputs[0][:, :-1], inputs[1]), axis=1)

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

        out = tf.concat((latents, tf.expand_dims(inputs[0][:, -1], axis=1)), axis=1)
        # (Not) Binary classifier
        for layer in self.eql_layers:
                out = layer(out, training=training)

        return out, means, logstds

class VAIL_EQN(RewardLearner):
    """Variational adversarial imitation learning with an equation-network discriminator."""
    def __init__(
            self,
            state_shape,
            action_dim,
            v=[1,1,1],
            exclude = None,
            drop_out = None,
            units=(32, 32),
            n_latent_unit=32,
            lr=5e-5,
            kl_target=0.5,
            reg_param=0.,
            enable_sn=False,
            grad_penalty_coeff = 0.,
            n_training = 1,
            name="VAIL_EQN",
            is_debug=False,
            **kwargs):
        """Configure the imitation objective, discriminator networks, and optimizer settings.

        Args:
            state_shape (tuple[int, ...]): Shape of one observation, excluding the batch axis.
            action_dim (int): Number of discrete choices or width of a continuous action vector.
            v (sequence[int] or int): Operator repetition counts, one per equation layer; an individual
                layer takes one count.
            exclude (sequence or None): Equation operators to exclude; network constructors use one
                exclusion list per layer.
            drop_out (float or sequence[float] or None): Dropout probability, or one probability per
                equation layer; None disables dropout.
            units (sequence[int]): Hidden-layer widths in forward order.
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
            **kwargs (dict): Additional keyword arguments forwarded to RewardLearner construction.

        Returns:
            None: Initializes the instance.
        """
        RewardLearner.__init__(self, name=name, n_training=1, **kwargs)
        assert isinstance(units, tuple),\
            "[ERORR] the units should be tuple or int [__init__ VAIL_EQN]"

        with tf.device(self.device):
            self.disc = Discriminator(state_shape=state_shape, action_dim=action_dim,
                                      units=units, n_latent_unit=n_latent_unit,
                                      enable_sn=enable_sn, v=v, exclude=exclude, drop_out = drop_out)

            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            self._reg_param = tf.Variable(reg_param, dtype=tf.float32)
            self._step_reg_param = tf.constant(1e-5, dtype=tf.float32)

        self._grad_penalty_coeff = grad_penalty_coeff
        self._enable_gp = grad_penalty_coeff > 0
        self._kl_target = kl_target
        self._is_debug = is_debug

        self._init_static_graph(state_shape, action_dim, n_latent_unit)


    def train(self, agent_states, agent_acts, expert_states, expert_acts, **kwargs):
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
        with tf.device(self.device):
            agent_states, agent_acts, expert_states, expert_acts = \
                tuple([tf.constant(i) for i in (agent_states, agent_acts, expert_states, expert_acts)])

            loss, accuracy, real_kl, fake_kl, js_divergence = self._train_body_graph(agent_states, agent_acts,
                                                                                     expert_states, expert_acts,
                                                                                     tf.constant(True))

            tf.summary.scalar(name=self.policy_name+"/DiscriminatorLoss", data=loss)
            tf.summary.scalar(name=self.policy_name+"/Accuracy", data=accuracy)
            tf.summary.scalar(name=self.policy_name+"/RegParam", data=self._reg_param)
            tf.summary.scalar(name=self.policy_name+"/RealLatentKL", data=real_kl)
            tf.summary.scalar(name=self.policy_name+"/FakeLatentKL", data=fake_kl)
            tf.summary.scalar(name=self.policy_name+"/JSdivergence", data=js_divergence)

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
        print("[DEBUG] initializing {_compute_js_divergence VAIL}")
        with tf.device(self.device):
            m = (fake_logits + real_logits) / 2.
            js = tf.reduce_mean((fake_logits * tf.math.log(fake_logits / m + 1e-8) +
                                 real_logits * tf.math.log(real_logits / m + 1e-8)) / 2.)
        return js

    def _compute_kl_latent(self, means, log_stds):
        """Compute KL divergence of latent spaces over standard Normal
        distribution to compute loss in eq.5.  The formulation of
        KL divergence between two normal distributions is as follows:
            ln(\\sigma_2 / \\sigma_1) + {(\\mu_1 - \\mu_2)^2 + \\sigma_1^2 - \\sigma_2^2} / (2 * \\sigma_2^2)
        Since the target distribution is standard Normal distributions,
        we can assume `\\sigma_2 = 1` and `mean_2 = 0`.
        So, the resulting equation can be computed as:
            ln(1 / \\sigma_1) + (\\mu_1^2 + \\sigma_1^2 - 1) / 2

        Args:
            means (tf.Tensor): Latent Gaussian means, shaped (batch, latent_width).
            log_stds (tf.Tensor): Latent log standard deviations, shaped (batch, latent_width).

        Returns:
            tf.Tensor: Per-sample KL divergence from the diagonal latent Gaussian to the standard
                normal, shaped (batch,).
        """
        print("[DEBUG] initializing {_compute_kl_latent VAIL}")
        with tf.device(self.device):
            kl_dist = tf.reduce_sum(-log_stds + (tf.square(means) + tf.square(tf.exp(log_stds)) - 1.) / 2., axis=-1)
        return kl_dist

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
            tuple[tf.Tensor, ...]: Scalar total loss, accuracy, expert latent KL, agent latent KL, and
                JS diagnostic, after discriminator and KL-multiplier updates.
        """
        print("[DEBUG] initializing {_train_body VAIL}.")
        epsilon = 1e-8
        with tf.device(self.device):
            with tf.GradientTape() as tape:

                real_logits, real_means, real_logstds = self.disc((expert_states, expert_acts), training)
                fake_logits, fake_means, fake_logstds = self.disc((agent_states, agent_acts), training)
                if self._enable_gp:
                    inter = (agent_states, agent_acts)
                    with tf.GradientTape(watch_accessed_variables=False) as tape2:
                        tape2.watch(list(inter))
                        output = self.disc(inter, training)
                    gp_grad = tape2.gradient(output, list(inter))[0]
                    gp_grad_penalty = tf.reduce_mean(tf.pow(tf.norm(gp_grad, axis=-1), 2))

                    # Compute discriminator loss
                    disc_loss = -(tf.reduce_mean(tf.math.log(real_logits + epsilon)) +
                                  tf.reduce_mean(tf.math.log(1. - fake_logits + epsilon)))
                    # Compute KL loss
                    real_kl = tf.reduce_mean(self._compute_kl_latent_graph(real_means, real_logstds))
                    fake_kl = tf.reduce_mean(self._compute_kl_latent_graph(fake_means, fake_logstds))
                    kl_loss = 0.5 * (real_kl - self._kl_target +
                                     fake_kl - self._kl_target)

                    # EQN Log and Div penalty loss
                    eqn_loss = tf.reduce_sum(self.disc.losses)

                    loss = disc_loss + self._reg_param * kl_loss + \
                           self._grad_penalty_coeff * gp_grad_penalty + eqn_loss
                else:
                    # Compute discriminator loss
                    disc_loss = -(tf.reduce_mean(tf.math.log(real_logits + epsilon)) +
                                  tf.reduce_mean(tf.math.log(1. - fake_logits + epsilon)))
                    # Compute KL loss
                    real_kl = tf.reduce_mean(self._compute_kl_latent_graph(real_means, real_logstds))
                    fake_kl = tf.reduce_mean(self._compute_kl_latent_graph(fake_means, fake_logstds))
                    kl_loss = 0.5 * (real_kl - self._kl_target +
                                     fake_kl - self._kl_target)

                    #EQN Log and Div penalty loss
                    eqn_loss = tf.reduce_sum(self.disc.losses)

                    loss = disc_loss + self._reg_param * kl_loss + eqn_loss


            grads = tape.gradient(loss, self.disc.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.disc.trainable_variables))

            # Update reguralizer parameter \beta in eq.(9)
            self._reg_param.assign(tf.maximum(tf.constant(0., dtype=tf.float32),
                                              self._reg_param + self._step_reg_param * (kl_loss - self._kl_target)))

        accuracy = (tf.reduce_mean(tf.cast(real_logits >= 0.5, tf.float32)) / 2. +
                    tf.reduce_mean(tf.cast(fake_logits < 0.5, tf.float32)) / 2.)
        js_divergence = self._compute_js_divergence_graph(fake_logits, real_logits)

        return loss, accuracy, real_kl, fake_kl, js_divergence

    def inference(self, states, actions, next_states=None):
        """Return learned imitation rewards for the supplied observations and actions or
        transitions.

        Args:
            states (array_like): Observation batch shaped (batch, *state_shape).
            actions (array_like): Actions aligned with the observation batch; continuous actions have a
                final action_dim axis.
            next_states (array_like or None): Physical successor observations aligned with the current-
                state batch. This implementation does not read this argument.

        Returns:
            tf.Tensor: Stabilized discriminator log-odds rewards shaped (batch, 1). Evaluation uses the
                latent mean rather than a sampled latent.
        """
        if states.ndim == actions.ndim == 1:
            states = np.expand_dims(states, axis=0)
            actions = np.expand_dims(actions, axis=0)
        with tf.device(self.device):
            out = self._inference_body_graph(tf.constant(states), tf.constant(actions), tf.constant(False))
        return out

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
        print("[DEBUG] initializing {_inference_body VAIL}")
        with tf.device(self.device):
            sig, _, _ = self.disc((states, actions), training=training)
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
        print("[DEBUG] initializing {_init_static_graph VAIL_EQN}")
        # Compiling the static graphs
        assert len(state_shape) == 1, "[ERROR] This SAC only supports RAM env {_init_static_graph SAC}"
        with tf.device(self.device):
            # TODO: the @tf.function on top is better!
            self.disc.call = tf.function(self.disc.call,
                                         input_signature=[
                                             (tf.TensorSpec(shape=(None, state_shape[-1]), dtype=tf.float32),
                                              tf.TensorSpec(shape=(None, action_dim), dtype=tf.float32)),
                                             tf.TensorSpec(shape=(None), dtype=tf.bool)]
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
                                                     input_signature=[
                                                         tf.TensorSpec(shape=(None, state_shape[-1]), dtype=tf.float32),
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

    @staticmethod
    def get_argument(parser=None):
        """Add this component's command-line options and return the argument parser.

        Args:
            parser (argparse.ArgumentParser or None): Parser to extend; None constructs a new parser.

        Returns:
            argparse.ArgumentParser: Parser extended with the supported component options.
        """
        parser = RewardLearner.get_argument(parser)
        parser.add_argument('--latent-units', type=int, default=5,
                            help='Number of latent dimention for VAIL. <default: %(default)s>')
        parser.add_argument('--kl-target', type=float, default=0.1,
                            help='KL Distance target for VAIL. <default: %(default)s>')
        parser.add_argument('--dropout', type=str, default='0,0',
                            help='Discriminator dropout. use case: .75,5 or 0.5  <default: %(default)s>')
        parser.add_argument('-v', '--v', type=str, default="1,1",
                            help='Discriminator architecture. use case: 2,1 or 1. <default: %(default)s>')
        parser.add_argument('--exclude', type=str, default="div-log-mult,id-sin-cos-log-sig-div",
                            help='Discriminator architecture. use case: None or sig,sig or cos.  <default: %(default)s>')
        return parser
