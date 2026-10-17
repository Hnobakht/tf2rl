"""Compatibility layers used by the retained GAIfO and VAIL experiments."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers, initializers, backend as K

from tf2rl.networks.eqn import EquationLayer as EqlLayerv2, EquationOutput as DenseLayerv2

__all__ = ["L1L2_EQL", "DenseLayer", "EqlLayerv2", "DenseLayerv2"]


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
