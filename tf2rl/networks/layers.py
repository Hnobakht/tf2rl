"""Compatibility layers used by the retained GAIfO and VAIL experiments."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers

from tf2rl.networks.equation import EquationLayer as EqlLayerv2
from tf2rl.networks.equation import EquationOutput as DenseLayerv2

__all__ = ["L1L2_EQL", "DenseLayer", "EqlLayerv2", "DenseLayerv2", "NoisyDense", "SNDense"]


class L1L2_EQL(regularizers.Regularizer):
    """Regularizer for L1 and L2 regularization.
    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, l1=0.0, l2=0.0):
        """Initialize mutable L1 and L2 regularization coefficients."""
        with K.name_scope(self.__class__.__name__):
            self.l1 = K.variable(l1, name='l1')
            self.l2 = K.variable(l2, name='l2')
            self.val_l1 = l1
            self.val_l2 = l2

    def set_l1_l2(self, l1, l2):
        """Update the regularizer's L1 and L2 coefficients in place."""
        K.set_value(self.l1, l1)
        K.set_value(self.l2, l2)
        self.val_l1 = l1
        self.val_l2 = l2

    def __call__(self, x):
        """Return the weighted L1 and L2 penalties for the input tensor."""
        regularization = 0.
        if not self.val_l1 == 0.:
            regularization += K.sum(self.l1 * K.abs(x))
        if not self.val_l2 == 0.:
            regularization += K.sum(self.l2 * K.square(x))
        return regularization

    def get_config(self):
        """Return the configuration needed to serialize this layer or regularizer."""
        config = {'l1': float(K.get_value(self.l1)),
                  'l2': float(K.get_value(self.l2))}
        return config


class DenseLayer(keras.layers.Layer):
    """Scalar dense output layer with configurable equation-network regularization."""
    def __init__(self, w_initializer, b_initializer, lmbda=0, constraint=None ,mask=None, activation = None, dynamic_lmbda=False):
        """Configure the scalar output activation, initializers, constraints, and L1 regularizer."""
        super(DenseLayer, self).__init__()
        self._is_reg_dyn = dynamic_lmbda
        self.activation = activation
        if not dynamic_lmbda:
            self.regularizer = regularizers.L1(lmbda)
        else:
            self.regularizer = L1L2_EQL()
        self.w_initializer = initializers.get(w_initializer)
        self.b_initializer = initializers.get(b_initializer)
        self.constraint=constraint

    def build(self, input_shape):
        """Create layer weights and biases for the input feature dimension."""
        self.w = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer=self.w_initializer,
            trainable=True, regularizer=self.regularizer,
            constraint=self.constraint, name="w"
        )
        self.b = self.add_weight(
            shape=(1,), initializer=self.b_initializer, trainable=True,
            regularizer=self.regularizer, constraint=self.constraint, name="b"
        )

    def call(self, inputs, training, l1_regularizers, mask = None):
        """Return the scalar affine output with its configured activation and regularization."""
        if training is None: tf.print("The training state of the layer needs to be set. [call DenseLayer]")
        if self._is_reg_dyn and training:
            print("[DEBUG] The regularizer is set to {} [call DenseLayer]".format(l1_regularizers))
            self.regularizer.set_l1_l2(l1_regularizers, 0.)
        elif self._is_reg_dyn and not training:
            self.regularizer.set_l1_l2(0., 0.)

        if mask:
            for i in range(self.w.shape[0]):
                w_mask = tf.matmul([self.w[i]], tf.cast(mask[0][i], tf.float32))[0]
                self.w[i].assign(w_mask)
            b_mask = tf.matmul([self.b], tf.cast(mask[1], tf.float32))[0]
            self.b.assign(b_mask)
        if self.activation is not None:
            out = self.activation(tf.matmul(inputs, self.w) + self.b)
        else:
            out = tf.matmul(inputs, self.w) + self.b
        return out


# This is an implementation of noisy linear layer defined in following paper:
# Noisy Networks for Exploration, https://arxiv.org/abs/1706.10295
# Forked from https://github.com/LuEE-C/Noisy-A3C-Keras/blob/master/NoisyDense.py
# Fixed a bug by Kei Ohta




class NoisyDense(tf.keras.layers.Layer):
    """Dense layer with learned noise scales and freshly perturbed weights on each call."""
    def __init__(
            self,
            units,
            sigma_init=0.017,
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            trainable=True,
            **kwargs):
        """Configure dense units, trainable noise scales, activation, and parameter initialization.
        """
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super().__init__(**kwargs)
        self.units = int(units)
        self.sigma_init = sigma_init
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(
            activity_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.trainable = trainable

    def build(self, input_shape):
        """Create dense weights, biases, and their trainable noise scales for the input shape."""
        assert len(input_shape) >= 2
        self.input_dim = input_shape[-1]
        self.kernel_shape = tf.constant((self.input_dim, self.units))
        self.bias_shape = tf.constant((self.units,))

        self.kernel = self.add_weight(
            shape=[self.input_dim, self.units],
            initializer=tf.keras.initializers.Orthogonal(),
            name='kernel',
            dtype=tf.float32,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=self.trainable)

        self.sigma_kernel = self.add_weight(
            shape=(self.input_dim, self.units),
            initializer=tf.keras.initializers.Constant(value=self.sigma_init),
            name='sigma_kernel',
            trainable=self.trainable)

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=self.trainable)

            self.sigma_bias = self.add_weight(
                shape=(self.units,),
                initializer=tf.keras.initializers.Constant(
                    value=self.sigma_init),
                name='sigma_bias',
                trainable=self.trainable)

        else:
            self.bias = None
            self.epsilon_bias = None

        super().build(input_shape)

    def call(self, inputs):
        # Implement Eq.(9)
        """Return dense activations using freshly sampled uniform parameter noise."""
        perturbed_kernel = (self.kernel
                            + self.sigma_kernel
                            * tf.random.uniform(shape=self.kernel_shape))
        outputs = tf.matmul(inputs, perturbed_kernel)
        if self.use_bias:
            perturbed_bias = (self.bias
                              + self.sigma_bias
                              * tf.random.uniform(shape=self.bias_shape))
            outputs = tf.nn.bias_add(outputs, perturbed_bias)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        """Return the output batch shape with the configured number of units."""
        return (input_shape[0], self.units)


"""The original SNDense rule using only public TensorFlow operations."""
import tensorflow as tf


class SNDense(tf.keras.layers.Dense):
    """Dense layer using a power-iteration estimate to spectrally normalize its kernel."""
    def __init__(self, units, trainable=True, u_kernel_initializer=None, **kwargs):
        """Configure the dense layer and initializer for its power-iteration vector."""
        super().__init__(units, trainable=trainable, **kwargs)
        if u_kernel_initializer is None:
            u_kernel_initializer = tf.keras.initializers.TruncatedNormal()
        elif isinstance(u_kernel_initializer, type):
            u_kernel_initializer = u_kernel_initializer()
        self.u_kernel_initializer = tf.keras.initializers.get(u_kernel_initializer)

    def build(self, input_shape):
        """Build the dense parameters and create the spectral-norm estimation vector."""
        super().build(input_shape)
        self.u_kernel = self.add_weight(name='u_kernel', shape=(1, self.units),
                                        initializer=self.u_kernel_initializer, trainable=False)

    def compute_spectral_norm(self):
        """Return the kernel divided by its estimated largest singular value."""
        def normalize(value):
            """Return a unit-length vector with a small stabilizer in the denominator."""
            return value / (tf.sqrt(tf.reduce_sum(tf.square(value))) + 1e-12)
        v = normalize(tf.matmul(self.u_kernel, self.kernel, transpose_b=True))
        u = normalize(tf.matmul(v, self.kernel))
        sigma = tf.matmul(tf.matmul(v, self.kernel), u, transpose_b=True)
        self.u_kernel.assign(u)
        return self.kernel / sigma

    def call(self, inputs):
        """Apply the spectrally normalized kernel, bias, and activation."""
        weights = self.compute_spectral_norm()
        if isinstance(inputs, tf.SparseTensor):
            output = tf.sparse.sparse_dense_matmul(inputs, weights)
        else:
            output = tf.tensordot(tf.cast(inputs, self.compute_dtype), weights, axes=1)
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        return self.activation(output) if self.activation is not None else output

    def get_config(self):
        """Return the configuration needed to serialize this layer or regularizer."""
        config = super().get_config()
        config['u_kernel_initializer'] = tf.keras.initializers.serialize(self.u_kernel_initializer)
        return config
