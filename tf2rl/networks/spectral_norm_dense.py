"""The original SNDense rule using only public TensorFlow operations."""
import tensorflow as tf


class SNDense(tf.keras.layers.Dense):
    def __init__(self, units, trainable=True, u_kernel_initializer=None, **kwargs):
        super().__init__(units, trainable=trainable, **kwargs)
        if u_kernel_initializer is None:
            u_kernel_initializer = tf.keras.initializers.TruncatedNormal()
        elif isinstance(u_kernel_initializer, type):
            u_kernel_initializer = u_kernel_initializer()
        self.u_kernel_initializer = tf.keras.initializers.get(u_kernel_initializer)

    def build(self, input_shape):
        super().build(input_shape)
        self.u_kernel = self.add_weight(name='u_kernel', shape=(1, self.units),
                                        initializer=self.u_kernel_initializer, trainable=False)

    def compute_spectral_norm(self):
        def normalize(value):
            return value / (tf.sqrt(tf.reduce_sum(tf.square(value))) + 1e-12)
        v = normalize(tf.matmul(self.u_kernel, self.kernel, transpose_b=True))
        u = normalize(tf.matmul(v, self.kernel))
        sigma = tf.matmul(tf.matmul(v, self.kernel), u, transpose_b=True)
        self.u_kernel.assign(u)
        return self.kernel / sigma

    def call(self, inputs):
        weights = self.compute_spectral_norm()
        if isinstance(inputs, tf.SparseTensor):
            output = tf.sparse.sparse_dense_matmul(inputs, weights)
        else:
            output = tf.tensordot(tf.cast(inputs, self.compute_dtype), weights, axes=1)
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        return self.activation(output) if self.activation is not None else output

    def get_config(self):
        config = super().get_config()
        config['u_kernel_initializer'] = tf.keras.initializers.serialize(self.u_kernel_initializer)
        return config
