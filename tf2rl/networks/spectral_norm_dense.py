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
