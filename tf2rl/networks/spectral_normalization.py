"""Dense spectral normalization with the historical Addons power-iteration rule.

Normalization assigns the kernel during training; inference does not normalize it.
This small project-local wrapper removes the dependency on retired TF Addons.
"""
import tensorflow as tf


class SpectralNormalization(tf.keras.layers.Wrapper):
    """Normalize a wrapped layer's kernel by its estimated spectral norm during training."""
    def __init__(self, layer, power_iterations=1, **kwargs):
        """Wrap a layer and require a positive number of power iterations.

        Args:
            layer (keras.layers.Layer): Wrapped layer whose kernel is spectrally normalized.
            power_iterations (int): Number of power iterations used to estimate the largest singular
                value.
            **kwargs (dict): Additional keyword arguments forwarded to tf.keras.layers.Wrapper
                construction.

        Returns:
            None: Initializes the instance.

        Raises:
            ValueError: power_iterations must be positive.
        """
        super().__init__(layer, **kwargs)
        if power_iterations < 1:
            raise ValueError("power_iterations must be positive")
        self.power_iterations = power_iterations

    def build(self, input_shape):
        """Build the wrapped layer and create its non-trainable power-iteration vector.

        Args:
            input_shape (tuple[int, ...] or tf.TensorShape): Incoming tensor dimensions, including the
                batch axis.

        Returns:
            None.
        """
        super().build(input_shape)
        self.w = self.layer.kernel
        self.u = self.add_weight(name="sn_u", shape=(1, self.w.shape[-1]),
                                 initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False)

    def _normalize(self):
        """Update the power-iteration estimate and normalize the wrapped kernel in place.

        Returns:
            tf.Tensor: Scalar zero used as a control dependency after normalizing the wrapped kernel and
                updating its power-iteration state.
        """
        weights = tf.reshape(self.w, (-1, self.w.shape[-1]))
        u = self.u
        for _ in range(self.power_iterations):
            v = tf.math.l2_normalize(tf.matmul(u, weights, transpose_b=True))
            u = tf.math.l2_normalize(tf.matmul(v, weights))
        u, v = tf.stop_gradient(u), tf.stop_gradient(v)
        sigma = tf.matmul(tf.matmul(v, weights), u, transpose_b=True)
        self.u.assign(u)
        self.w.assign(tf.reshape(weights / sigma, self.w.shape))
        return tf.constant(0)

    def call(self, inputs, training=None):
        """Normalize the wrapped kernel during training and return the layer output.

        Args:
            inputs (tf.Tensor): Batched input features; the final axis contains the model input
                coordinates.
            training (bool or tf.Tensor or None): Keras training flag controlling stochastic or stateful
                layers; None follows the layer default.

        Returns:
            tf.Tensor: Wrapped-layer output; training calls normalize the kernel before evaluation.
        """
        if training is True:
            self._normalize()
        elif training is not None and not isinstance(training, bool):
            tf.cond(training, self._normalize, lambda: tf.constant(0))
        return self.layer(inputs)
