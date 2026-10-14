import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from tf2rl.algos.gail import GAIL
from tf2rl.algos.policy_base import IRLPolicy
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
        activation = "relu"
        v_activation = "linear"
        output_activation = "sigmoid"
        if not enable_sn:
            self._encoder_model(units, activation)
            self.l_mean = Dense(n_latent_unit, name="L_mean", activation=v_activation)
            self.l_logstd = Dense(n_latent_unit, name="L_std", activation=v_activation)
            self._decoder_model(decoder_units, activation, output_activation)
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
            self._encoder_list.append(Dense(unit, name="L_EN"+str(index), activation=activation))
            index += 1

    def _decoder_model(self, units, activation, output_activation):
        if len(units) > 1:
            index = 0
            for unit in units:
                self._decoder_list.append(Dense(unit, name="L_DE"+str(index), activation=activation))
                index += 1
            self._decoder_list.append(Dense(1, name="L_DE_out", activation=output_activation))
        else:
            print("[WARNING] Discriminator has one dimention!")
            self._decoder_list.append(Dense(1, name="L_DE_out", activation=output_activation))

    # @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=None):
        print("[DEBUG] initializing {call VAIL}")
        # Encoder
        # features = tf.concat(inputs, axis=1)
        if training == None:
            training = True

        for layer in self._encoder_list:
            inputs = layer(inputs)

        if training == True:
            means = self.l_mean(inputs)
            logstds = self.l_logstd(inputs)
            logstds = tf.clip_by_value(logstds, self.LOG_SIG_CAP_MIN, self.LOG_SIG_CAP_MAX)
            latents = means + tf.random.normal(shape=means.shape) * tf.math.exp(logstds)
        else:
            latents = self.l_mean(inputs)
            means = None
            logstds = None

        # (Not) Binary classifier
        for layer in self._decoder_list:
            latents = layer(latents)
        # out = latents

        return latents, means, logstds

    @tf.function(experimental_relax_shapes=True)
    def compute_reward(self, inputs):
        print("[DEBUG] initializing {compute_reward VAIL}")
        # features = tf.concat(inputs, axis=1)
        # features = self.l1(inputs)
        # features = self.l2(features)
        # means = self.l_mean(features)
        # sig = self.l3(means)
        sig, _, _ = self(inputs, training = False)
        return tf.math.log(sig + 1e-8) - tf.math.log(1 - sig)


class VAIL(GAIL):
    def __init__(
            self,
            state_shape,
            action_dim,
            units=(32, 32),
            decoder_units = (1),
            n_latent_unit=32,
            lr=5e-5,
            kl_target=0.5,
            reg_param=0.,
            enable_sn=False,
            grad_penalty_coeff = 0.,
            n_training = 1,
            name="VAIL",
            **kwargs):
        IRLPolicy.__init__(self, name=name, n_training=1, **kwargs)
        assert isinstance(decoder_units, tuple) and isinstance(units, tuple),\
            "[ERORR] the units should be tuple or int [__init__ VAIL]"
        self.disc = Discriminator(state_shape=state_shape, action_dim=action_dim,
                                  units=units, n_latent_unit=n_latent_unit, decoder_units=decoder_units,
                                  enable_sn=enable_sn)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self._kl_target = kl_target
        self._reg_param = tf.Variable(reg_param, dtype=tf.float32)
        self._step_reg_param = tf.constant(1e-5, dtype=tf.float32)
        self._grad_penalty_coeff = grad_penalty_coeff
        self._enable_gp = grad_penalty_coeff > 0


    def train(self, agent_states, agent_acts, expert_states, expert_acts, **kwargs):
        loss, accuracy, real_kl, fake_kl, js_divergence = self._train_body(agent_states, agent_acts,
                                                                           expert_states, expert_acts)
        tf.summary.scalar(name=self.policy_name+"/DiscriminatorLoss", data=loss)
        tf.summary.scalar(name=self.policy_name+"/Accuracy", data=accuracy)
        tf.summary.scalar(name=self.policy_name+"/RegParam", data=self._reg_param)
        tf.summary.scalar(name=self.policy_name+"/RealLatentKL", data=real_kl)
        tf.summary.scalar(name=self.policy_name+"/FakeLatentKL", data=fake_kl)
        tf.summary.scalar(name=self.policy_name+"/JSdivergence", data=js_divergence)

    @tf.function(experimental_relax_shapes=True)
    def _compute_kl_latent(self, means, log_stds):
        r"""
        Compute KL divergence of latent spaces over standard Normal
        distribution to compute loss in eq.5.  The formulation of
        KL divergence between two normal distributions is as follows:
            ln(\sigma_2 / \sigma_1) + {(\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2} / (2 * \sigma_2^2)
        Since the target distribution is standard Normal distributions,
        we can assume `\sigma_2 = 1` and `mean_2 = 0`.
        So, the resulting equation can be computed as:
            ln(1 / \sigma_1) + (\mu_1^2 + \sigma_1^2 - 1) / 2
        """
        print("[DEBUG] initializing {_compute_kl_latent VAIL}")
        return tf.reduce_sum(-log_stds + (tf.square(means) + tf.square(tf.exp(log_stds)) - 1.) / 2., axis=-1)

    @tf.function(experimental_relax_shapes=True)
    def _train_body(self, agent_states, agent_acts, expert_states, expert_acts):
        print("[DEBUG] initializing {_train_body VAIL}.")
        epsilon = 1e-8
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                # Compute discriminator loss
                real_logits, real_means, real_logstds = self.disc(tf.concat((expert_states, expert_acts), axis=1))
                fake_logits, fake_means, fake_logstds = self.disc(tf.concat((agent_states, agent_acts), axis=1))
                disc_loss = -(tf.reduce_mean(tf.math.log(real_logits + epsilon)) +
                              tf.reduce_mean(tf.math.log(1. - fake_logits + epsilon)))
                # Compute KL loss
                real_kl = tf.reduce_mean(self._compute_kl_latent(real_means, real_logstds))
                fake_kl = tf.reduce_mean(self._compute_kl_latent(fake_means, fake_logstds))
                kl_loss = 0.5 * (real_kl - self._kl_target +
                                 fake_kl - self._kl_target)
                loss = disc_loss + self._reg_param * kl_loss
                # Gradient penalty
                if self._enable_gp:
                    raise NotImplementedError

            grads = tape.gradient(loss, self.disc.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.disc.trainable_variables))

            # Update reguralizer parameter \beta in eq.(9)
            self._reg_param.assign(tf.maximum(tf.constant(0., dtype=tf.float32),
                                              self._reg_param + self._step_reg_param * (kl_loss - self._kl_target)))

        accuracy = (tf.reduce_mean(tf.cast(real_logits >= 0.5, tf.float32)) / 2. +
                    tf.reduce_mean(tf.cast(fake_logits < 0.5, tf.float32)) / 2.)
        js_divergence = self._compute_js_divergence(fake_logits, real_logits)
        return loss, accuracy, real_kl, fake_kl, js_divergence

    @staticmethod
    def get_argument(parser=None):
        parser = IRLPolicy.get_argument(parser)
        # parser.add_argument('--enable-sn', action='store_true')
        parser.add_argument('--units-decoder', type=str, default="1",
                            help='Decoder architecture for VAIL. use case: 128,128 or 400,300 <default: %(default)s>')
        parser.add_argument('--latent-units', type=int, default=5,
                            help='Number of latent dimention for VAIL. <default: %(default)s>')
        parser.add_argument('--kl-target', type=float, default=0.5,
                            help='KL Distance target for VAIL. <default: %(default)s>')
        return parser
