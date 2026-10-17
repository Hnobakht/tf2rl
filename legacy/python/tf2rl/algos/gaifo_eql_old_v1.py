import numpy as np
import tensorflow as tf

# import tensorflow_addons as tfa
# import tensorflow_gan.python.losses.losses_impl as tfgan_losses
# from tensorflow.keras.layers import Dense

# from tf2rl.networks.layer import EqlLayer, DenseLayer
from tf2rl.networks.layer import DenseLayer, EqlLayerv2 as EqlLayer
from tensorflow.keras import initializers #, regularizers, backend as K

# from tensorflow.keras.constraints import Constraint
# from tensorflow.python.keras.utils.version_utils import training

from tf2rl.algos.policy_base import IRLPolicy
# from tf2rl.misc.normalizer import Normalizer
from tf2rl.misc.target_update_ops import update_target_variables


class Discriminator_EQL(tf.keras.Model):
    """Equation-network discriminator for expert and agent transitions."""
    def __init__(self, state_shape, num_layers=2, drop_out = None,
                 enable_sn=False, output_activation=None, lmbda=0,
                 name="GAIfO_Discriminator", v=None, exclude=None, atol=0, constraint=None,
                 w_initializer='random_normal', b_initializer='random_normal',
                 is_lmbda_dynamic = False):
        """Build the equation discriminator with operator groups, dropout, and regularization."""
        super().__init__(name=name)
        print("[DEBUG] Building [__init__ Discriminator_EQL]")
        self.num_layers = num_layers
        self.w_initializer = initializers.get(w_initializer)
        self.b_initializer = initializers.get(b_initializer)
        # self.w_initializer = initializers.Constant(value=0.1)
        # self.b_initializer = initializers.Constant(value=0.1)
        self.atol= atol
        self.lmbda = lmbda
        self.drop_out = drop_out
        self._is_lmbda_dynamic = is_lmbda_dynamic
        self.output_activation = output_activation

        if drop_out is not None:
            assert np.array(drop_out).any() <= 1 and np.array(drop_out).any() > 0, "drop_out is a ratio and it needs to be between 0 and 1"
            if isinstance(drop_out, list):
                assert len(drop_out) == num_layers, "Lenth of dropout dosent match the number of the layers."

        if exclude:
            assert len(exclude) == self.num_layers, "exclude parameter wrong format, len(exclude) must equal " \
                                                    "num_layers, ex.: exclude=[['sin'],[]], num_layers = 2"
            self.exclude = exclude
        else:
            exclude = [[] for i in range(self.num_layers)]
            self.exclude = exclude

        if v is None:
            self.v = np.ones(self.num_layers, dtype=int)
        else:
            self.v = v

        assert len(self.v) == self.num_layers, 'v array must have same dimensions as number of hidden layers param'

        self.eql_layers = []
        # self.eql_layers.append(tfa.layers.SpectralNormalization(Dense(state_shape[0]+action_dim,
        #                                                          name="spec_layer",activation=tf.tanh,
        #                                                          kernel_initializer=initializers.glorot_uniform())))
        for index in range(self.num_layers):
            self.eql_layers.append(EqlLayer(w_initializer=self.w_initializer, b_initializer=self.b_initializer,
                                            v=self.v[index], exclude=self.exclude[index],
                                            constraint=constraint, lmbda=lmbda, dynamic_lmbda=self._is_lmbda_dynamic))

            if drop_out and not isinstance(drop_out, list):
                self.eql_layers.append(tf.keras.layers.Dropout(drop_out))
            elif drop_out and isinstance(drop_out, list):
                self.eql_layers.append(tf.keras.layers.Dropout(drop_out[index]))

        self.layer_out = DenseLayer(w_initializer=self.w_initializer, b_initializer=self.b_initializer,
                                    constraint=constraint, lmbda=lmbda, dynamic_lmbda=self._is_lmbda_dynamic,
                                    activation=output_activation)

        dummy_state = tf.constant(
            np.zeros(shape=(1,)+state_shape, dtype=np.float32))
        dummy_action = tf.constant(
            np.zeros(shape=(1,)+state_shape, dtype=np.float32))
        with tf.device("/cpu:0"):
            input_tf = tf.concat((dummy_state, dummy_action), axis=1)
            print("[DEBUG] initializing dummy variable. [__init__ Discriminator_EQL] input_shape: {} output_shape: {} name: {}"
                  .format(input_tf.shape, self(input_tf, training=False).shape, self.name))

    def call(self, inputs, training=None, l1_regularizers = 0):
        """Return discriminator outputs for the supplied batch of inputs."""
        print("[DEBUG] initializing [call Discriminator_EQL shapes: {} name: {}".format(inputs.shape, self.name))

        if training is None:
            training = True
        features = tf.cast(inputs, tf.float32)
        for layer in self.eql_layers:
            if "dropout" in layer.name or "spec" in layer.name:
                features = layer(features,
                                 training=training)
            else:
                features = layer(features,
                                 training=training,
                                 l1_regularizers=l1_regularizers)

        return self.layer_out(features, training=training, l1_regularizers=l1_regularizers)

    def compute_reward(self, inputs, l1_regularizers = 0):
        """Return stabilized discriminator log odds for each input sample."""
        print("[DEBUG] initializing [compute_reward Discriminator_EQL] shape: {} name: {}".format(inputs.shape, self.name))
        return tf.math.log(self(inputs, training=False, l1_regularizers=l1_regularizers) + 1e-8) - \
               tf.math.log(1 - self(inputs, training=False, l1_regularizers=l1_regularizers) + 1e-8)
        # return tf.math.log(self(inputs, training=False, l1_regularizers=l1_regularizers) + 1e-8)
        # return -tf.math.log(1 - self(inputs, training=False, l1_regularizers=l1_regularizers) + 1e-8)
        # return self(inputs, training=False, l1_regularizers=l1_regularizers)

class GAIfO_EQL(IRLPolicy):
    """Observation-only adversarial imitation with an equation discriminator."""
    def __init__(
            self,
            state_shape,
            action_dim,
            training_stages=None,
            num_layers=2,
            lr=0.001,
            enable_sn=False,
            name="GAIfO_EQL",
            is_debug = False,
            update_delay = 1,
            atol = 0,
            atol_starting_step = 0,
            v=[2, 1],
            mask_rate = 0,
            masking_starting_step = 0,
            lmbda = 0,
            regularizer_starting_step = 0,
            drop_out = None,
            tau = 0,
            exclude = None,
            grad_penalty_coeff = 0,
            **kwargs):
        """Configure the imitation objective, discriminator networks, and optimizer settings."""
        super().__init__(name=name, n_training=1, **kwargs)
        self._is_debug = is_debug
        self.update_delay = update_delay
        self.atol = atol
        self.atol_starting_step = atol_starting_step if atol > 0 else None
        self.v = v
        self.mask_rate = mask_rate
        self.masking_starting_step = masking_starting_step if mask_rate > 0 else None
        self.lmbda = lmbda
        self.regularizer_starting_step = regularizer_starting_step if lmbda > 0 else None
        self._is_reg_dyna = False # True if self.regularizer_starting_step is not None else False
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.num_layers = num_layers
        self.enable_sn = enable_sn
        self.drop_out = drop_out if drop_out is not None else None
        self.tau = tau
        unit = None #To make sure of EQL independency.
        self.lr = lr
        self.exclude = exclude
        self.grad_penalty_coeff = grad_penalty_coeff
        self._is_grad_penalty = grad_penalty_coeff > 0
        self._init()

    def _init(self):
        """Create the discriminator copies and initialize their parameters."""
        self.disc = Discriminator_EQL(state_shape=self.state_shape, v=self.v, drop_out=self.drop_out,
                                      num_layers=self.num_layers, enable_sn=self.enable_sn, constraint=None,
                                      name="Discriminator", lmbda=self.lmbda, exclude = self.exclude,
                                      is_lmbda_dynamic=self._is_reg_dyna, output_activation=tf.nn.sigmoid)

        if self.atol_starting_step or self.tau > 0:
            self.disc_target = Discriminator_EQL(state_shape=self.state_shape, v=self.v,drop_out=self.drop_out,
                                              num_layers=self.num_layers, enable_sn=self.enable_sn, constraint=None,
                                              name="Discriminator_target", lmbda=self.lmbda,  exclude = self.exclude,
                                              is_lmbda_dynamic=self._is_reg_dyna, output_activation=tf.nn.sigmoid)
        if self.tau > 0:
            with tf.device(self.device):
                update_target_variables(self.disc_target.weights,
                                        self.disc.weights, tau=1.)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=0.5)
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def train(self, agent_states, agent_next_states, expert_states, expert_next_states, itr, **kwargs):
        """
        Update the imitation discriminator from agent and expert minibatches and report diagnostics.
        """
        assert itr , "The interation must be defined [train GAIfO_EQL]"

        if self._is_reg_dyna and self.regularizer_starting_step < itr:
            raise NotImplementedError
            l1_regularizers = self.lmbda
        else:
            l1_regularizers = 0

        if itr % self.update_delay == 0:
            # if self._is_debug: tf.summary.trace_on(graph=True)
            # loss, accuracy, js_divergence, grads = self._train_body(agent_states, agent_next_states,
            loss, accuracy, js_divergence, grads = self._train_body_eql(agent_states, agent_next_states,
                                                                    expert_states, expert_next_states,
                                                                    l1_regularizers)
            # if self._is_debug: tf.summary.trace_export(name="_train_body")
            tf.summary.scalar(name=self.policy_name+"/DiscriminatorLoss", data=loss)
            tf.summary.scalar(name=self.policy_name+"/Accuracy", data=accuracy)
            tf.summary.scalar(name=self.policy_name+"/JSdivergence", data=js_divergence)
            if self._is_debug:
                [tf.summary.histogram(name=self.policy_name + "_Weight/{} - {}".format(i.name, i.shape), data=i) for i  in self.disc.weights]
                [tf.summary.histogram(name=self.policy_name + "_Grad/{}".format(i.shape), data=i) for i in grads]

    @tf.function
    def _compute_js_divergence(self, fake_logits, real_logits):
        """Return the historical Jensen-Shannon diagnostic for agent and expert outputs."""
        print("[DEBUG] initializing [_compute_js_divergence GAIfO_EQL] shape: {TODO}")
        m = (fake_logits + real_logits) / 2.
        return tf.reduce_mean((
            fake_logits * tf.math.log(fake_logits / m + 1e-8) + real_logits * tf.math.log(real_logits / m + 1e-8)) / 2.)

    @tf.function
    def _train_body(self, agent_states, agent_next_states, expert_states, expert_next_states, l1_regularizers):
        """
        Apply discriminator gradients for the expert and agent batches and return training metrics.
        """
        print("[DEBUG] initializing [_train_body GAIfO_EQL] shape: {TODO}")
        epsilon = 1e-8
        with tf.device(self.device):
            with tf.GradientTape() as tape: # watch_accessed_variables=False

                real_logits = self.disc(tf.concat((expert_states, expert_next_states), axis=1), l1_regularizers=l1_regularizers)
                fake_logits = self.disc(tf.concat((agent_states, agent_next_states), axis=1), l1_regularizers=l1_regularizers)

                if self._is_grad_penalty:
                    inter = tf.concat((agent_states, agent_next_states), axis=1)
                    with tf.GradientTape(watch_accessed_variables=False) as tape2:
                        tape2.watch(inter)
                        output = self.disc(inter)
                    grad = tape2.gradient(output, [inter])[0]
                    grad_penalty = tf.reduce_mean(tf.pow(tf.norm(grad, axis=-1), 2))
                    loss = -(tf.reduce_mean(tf.math.log(real_logits + epsilon)) +
                             tf.reduce_mean(tf.math.log(1. - fake_logits + epsilon)) -
                             self.grad_penalty_coeff * grad_penalty)
                else:
                    # GAN like loss
                    loss = -(tf.reduce_mean(tf.math.log(real_logits + epsilon)) +
                             tf.reduce_mean(tf.math.log(1. - fake_logits + epsilon)))

                # loss = tfgan_losses.modified_discriminator_loss(real_logits, fake_logits, label_smoothing=0.0)

            grads = tape.gradient(loss, self.disc.trainable_variables)
            # grads = [tf.clip_by_norm(i, 8, axes=-1) for i in grads]
            self.optimizer.apply_gradients( zip(grads, self.disc.trainable_variables))

            accuracy = (tf.reduce_mean(tf.cast(real_logits >= 0.5, tf.float32)) / 2. +
                        tf.reduce_mean(tf.cast(fake_logits < 0.5, tf.float32)) / 2.)
            js_divergence = self._compute_js_divergence(fake_logits, real_logits)

            if self.tau > 0:
                update_target_variables(self.disc_target.weights,
                                        self.disc.weights, tau=self.tau)
        return loss, accuracy, js_divergence, grads

    @tf.function
    def _train_body_eql(self, agent_states, agent_next_states, expert_states, expert_next_states, l1_regularizers):
        """Update the equation discriminator with its regularization and singularity penalties."""
        print("[DEBUG] initializing [_train_body_eql GAIfO_EQL] shape: {TODO}")
        epsilon = 1e-8
        with tf.device(self.device):
            with tf.GradientTape() as tape:  # watch_accessed_variables=False

                real_logits = self.disc(tf.concat((expert_states, expert_next_states), axis=1),
                                        l1_regularizers=l1_regularizers)
                fake_logits = self.disc(tf.concat((agent_states, agent_next_states), axis=1),
                                        l1_regularizers=l1_regularizers)

                if self._is_grad_penalty:
                    inter = tf.concat((agent_states, agent_next_states), axis=1)
                    with tf.GradientTape(watch_accessed_variables=False) as tape2:
                        tape2.watch(inter)
                        output = self.disc(inter)
                    grad = tape2.gradient(output, [inter])[0]
                    grad_penalty = tf.reduce_mean(tf.pow(tf.norm(grad, axis=-1), 2))
                    loss = -(tf.reduce_mean(tf.math.log(real_logits + epsilon)) +
                             tf.reduce_mean(tf.math.log(1. - fake_logits + epsilon)) -
                             self.grad_penalty_coeff * grad_penalty)
                else:
                    # GAN like loss
                    loss = -(tf.reduce_mean(tf.math.log(real_logits + epsilon)) +
                             tf.reduce_mean(tf.math.log(1. - fake_logits + epsilon)) -
                             1 * tf.reduce_sum(self.disc.losses)) # penalty for log singularities

                # loss = tfgan_losses.modified_discriminator_loss(real_logits, fake_logits, label_smoothing=0.0)

            grads = tape.gradient(loss, self.disc.trainable_variables)
            # grads = [tf.clip_by_norm(i, 8, axes=-1) for i in grads]
            self.optimizer.apply_gradients(zip(grads, self.disc.trainable_variables))

            accuracy = (tf.reduce_mean(tf.cast(real_logits >= 0.5, tf.float32)) / 2. +
                        tf.reduce_mean(tf.cast(fake_logits < 0.5, tf.float32)) / 2.)
            js_divergence = self._compute_js_divergence(fake_logits, real_logits)

            if self.tau > 0:
                update_target_variables(self.disc_target.weights,
                                        self.disc.weights, tau=self.tau)
        return loss, accuracy, js_divergence, grads

    def inference(self, states, actions, next_states, apply_atol=False):
        """
        Return learned imitation rewards for the supplied observations and actions or transitions.
        """
        if states.ndim == 1:
            states = np.expand_dims(states, axis=0)
            next_states = np.expand_dims(next_states, axis=0)
        inputs = np.concatenate((states, next_states), axis=1)
        # if self._is_debug: tf.summary.trace_on(graph=True)
        if apply_atol:
            ret = self._inference_body_atol(inputs)
        else:
            ret = self._inference_body(inputs)
        # if self._is_debug: tf.summary.trace_export(name="_inference_body")
        return ret

    @tf.function
    def _inference_body_atol(self, inputs):
        """Evaluate rewards using a copy with small weights suppressed by the tolerance."""
        print("[DEBUG] initializing [_inference_body_atol GAIfO_EQL] shape: {} name: {}-{}".format(inputs.shape,
                                                                                             self.disc.name,
                                                                                           self.disc_target.name))
        with tf.device(self.device):
            update_target_variables(self.disc_target.weights
                                    ,([tf.where(tf.abs(i) > self.atol, i, 0.)
                                       for i in self.disc.weights])
                                    )
            return  self.disc_target.compute_reward(inputs)
            # return self.disc.compute_reward(inputs)

    @tf.function
    def _inference_body(self, inputs):
        """
        Return learned imitation rewards for the supplied observations and actions or transitions.
        """
        print("[DEBUG] initializing [_inference_body GAIfO_EQL] shape: {}".format(inputs.shape))
        with tf.device(self.device):
            if self.tau > 0:
                print("[DEBUG] inferenceing with the TARGET network.")
                return self.disc_target.compute_reward(inputs)
            else:
                return self.disc.compute_reward(inputs)
    @staticmethod
    def get_argument(parser=None):
        """Add this component's command-line options and return the argument parser."""
        parser = IRLPolicy.get_argument(parser)
        parser.add_argument('--update-delay', type=int, default=3,
                            help='Discriminator training delay.  <default: %(default)s>')
        # parser.add_argument('--dropout', type=float, default=0.5,
        #                     help='Discriminator dropout.  <default: %(default)s>')
        parser.add_argument('--dropout', type=str, default='.5,0.',
                            help='Discriminator dropout. use case: .75,5 or 0.5  <default: %(default)s>')
        parser.add_argument('-v', '--v', type=str, default="2,1",
                            help='Discriminator architecture. use case: 2,1 or 1. <default: %(default)s>')
        parser.add_argument('--exclude', type=str, default=None,
                            help='Discriminator architecture. use case: None or sig,sig or cos.  <default: %(default)s>')
        # parser.add_argument('--enable-sn', action='store_true')

        return parser
