import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers, initializers, backend as K

@tf.function
def identity(out, index):
    return tf.identity(tf.gather(out, [index], axis=1), name='identity_output')

@tf.function
def sin(out, index):
    return tf.sin(tf.gather(out, [index], axis=1), name='sin_output')

@tf.function
def cos(out, index):
    return tf.cos(tf.gather(out, [index], axis=1), name='cos_output')


@tf.function
def sigmoid(out, index):
    return tf.sigmoid(tf.gather(out, [index], axis=1), name='sig_output')


@tf.function
def mult(out, index):
    sum1 = tf.gather(out, [index], axis=1)
    sum2 = tf.gather(out, [index + 1], axis=1)
    sum_input = tf.add(sum1, sum2)
    return tf.multiply(sum_input, sum_input, name='mult_output')

@tf.function
def log(out, index):
    relax = 0.1
    z = tf.gather(out, [index], axis=1)
    return tf.where(z >= 0,
                     tf.math.log(tf.where(z >= 0 ,z + relax , 0) + 1e-8, name='log_output'),
                     0)
@tf.function
def div(out, index):
    relax = 0.1
    n = tf.gather(out, [index], axis=1)
    d = tf.gather(out, [index + 1], axis=1)
    return tf.where(d > 0,
                    tf.divide(n + relax, tf.where(d > 0, d + relax, 1), name='div_output'),
                    0)

# def exp(out, index):
#     return tf.exp(tf.gather(out, [index], axis=1), name='exp_output')


class L1L2_EQL(regularizers.Regularizer):
    """Regularizer for L1 and L2 regularization.
    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, l1=0.0, l2=0.0):
        with K.name_scope(self.__class__.__name__):
            self.l1 = K.variable(l1, name='l1')
            self.l2 = K.variable(l2, name='l2')
            self.val_l1 = l1
            self.val_l2 = l2

    def set_l1_l2(self, l1, l2):
        K.set_value(self.l1, l1)
        K.set_value(self.l2, l2)
        self.val_l1 = l1
        self.val_l2 = l2

    def __call__(self, x):
        regularization = 0.
        if not self.val_l1 == 0.:
            regularization += K.sum(self.l1 * K.abs(x))
        if not self.val_l2 == 0.:
            regularization += K.sum(self.l2 * K.square(x))
        return regularization

    def get_config(self):
        config = {'l1': float(K.get_value(self.l1)),
                  'l2': float(K.get_value(self.l2))}
        return config


class EqlLayer(keras.layers.Layer):
    def __init__(self, w_initializer, b_initializer, v, lmbda=0,
                 constraint=None, mask=None, exclude=None, dynamic_lmbda = False):
        super(EqlLayer, self).__init__()
        self._is_reg_dyn = dynamic_lmbda
        if exclude is None:
            exclude = []

        if not dynamic_lmbda:
            self.regularizer = regularizers.L1(lmbda)
        else:
            self.regularizer = L1L2_EQL()

        self.w_initializer = initializers.get(w_initializer)
        self.b_initializer = initializers.get(b_initializer)
        self.mask = mask
        self.v = v
        self.activations = [identity, sin, cos, sigmoid, mult]
        self.constraint=constraint

        self.exclusion = 0
        if 'id' in exclude:
            self.exclusion += 1
            self.activations.remove(identity)
        if 'sin' in exclude:
            self.exclusion += 1
            self.activations.remove(sin)
        if 'cos' in exclude:
            self.exclusion += 1
            self.activations.remove(cos)
        if 'sig' in exclude:
            self.exclusion += 1
            self.activations.remove(sigmoid)
        if 'mult' in exclude:
            self.exclusion += 2
            self.activations.remove(mult)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], 6 * self.v - self.v * self.exclusion),
            initializer=self.w_initializer,
            trainable=True, regularizer=self.regularizer,
            constraint=self.constraint, name="w"
        )
        self.b = self.add_weight(
            shape=(6 * self.v - self.v * self.exclusion,), initializer=self.b_initializer,
            trainable=True, regularizer=self.regularizer,
            constraint=self.constraint, name="b"
        )

    def call(self, inputs, training, l1_regularizers = 0):
        if training is None: tf.print("[WARNNING]The training state of the layer needs to be set. [call EqlLayer]")
        if self._is_reg_dyn and training:
            print("[DEBUG] The regularizer is set to {} [call EqlLayer]".format(l1_regularizers))
            self.regularizer.set_l1_l2(l1_regularizers, 0.)
        elif self._is_reg_dyn and not training:
            print("[DEBUG] The regularizer is set to {} [call EqlLayer]".format(l1_regularizers))
            self.regularizer.set_l1_l2(0., 0.)

        if self.mask:
            for i in range(self.w.shape[0]):
                #TODO: Check this DEBUG!
                w_mask = tf.matmul([self.w[i]], tf.cast(self.mask[0][i], tf.float32))[0]
                self.w[i].assign(w_mask)
            b_mask = tf.matmul([self.b], tf.cast(self.mask[1], tf.float32))[0]
            self.b.assign(b_mask)

        out = tf.matmul(inputs, self.w) + self.b
        output_batches = []
        for i in range(self.v):
            v = (6 - self.exclusion) * i
            for a in range(len(self.activations)):
                activation = self.activations[a](out, a + v)
                output_batches.append(activation)
        output = tf.concat(output_batches, axis=1)
        return output


class DenseLayer(keras.layers.Layer):
    def __init__(self, w_initializer, b_initializer, lmbda=0, constraint=None ,
                 mask=None, activation = None, dynamic_lmbda=False):
        super(DenseLayer, self).__init__()
        self._is_reg_dyn = dynamic_lmbda
        self.activation = activation
        if not dynamic_lmbda:
            self.regularizer = regularizers.L1(lmbda)
        else:
            self.regularizer = L1L2_EQL()
        self.w_initializer = initializers.get(w_initializer)
        self.b_initializer = initializers.get(b_initializer)
        # self.mask = mask
        self.constraint=constraint

    def build(self, input_shape):
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

class EqlLayerv2(keras.layers.Layer):
    def __init__(self, w_initializer, b_initializer, v, lmbda=0,
                 constraint=None, mask=None, exclude=None, dynamic_lmbda=False):
        super(EqlLayerv2, self).__init__()
        self._is_reg_dyn = dynamic_lmbda
        if exclude is None:
            exclude = []

        self.regularizer = None
        self.w_initializer = initializers.get(w_initializer)
        self.b_initializer = initializers.get(b_initializer)
        self.mask = mask
        self.v = v
        # self.activations = [identity, sin, cos, sigmoid, log, mult, div]
        self.activations_prop = {"identity":identity, "sin":sin, "cos":cos,
                                 "sigmoid":sigmoid, "log":log, "mult":mult, "div":div}
        self.constraint = constraint

        self.exclusion = 0
        if 'id' in exclude:
            self.exclusion += 1
            # self.activations.remove(identity)
            self.activations_prop.pop("identity")
        if 'sin' in exclude:
            self.exclusion += 1
            # self.activations.remove(sin)
            self.activations_prop.pop("sin")
        if 'cos' in exclude:
            self.exclusion += 1
            # self.activations.remove(cos)
            self.activations_prop.pop("cos")
        if 'sig' in exclude:
            self.exclusion += 1
            # self.activations.remove(sigmoid)
            self.activations_prop.pop("sigmoid")
        if 'log' in exclude:
            self.exclusion += 1
            # self.activations.remove(log)
            self.activations_prop.pop("log")
        if 'mult' in exclude:
            self.exclusion += 2
            # self.activations.remove(mult)
            self.activations_prop.pop("mult")
        if 'div' in exclude:
            self.exclusion += 2
            # self.activations.remove(div)
            self.activations_prop.pop("div")


    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], 9 * self.v - self.v * self.exclusion),
            initializer=self.w_initializer,
            trainable=True, regularizer=self.regularizer,
            constraint=self.constraint, name="w"
        )
        self.b = self.add_weight(
            shape=(9 * self.v - self.v * self.exclusion,), initializer=self.b_initializer,
            trainable=True, regularizer=self.regularizer,
            constraint=self.constraint, name="b"
        )

        # self.alpha = tf.Variable(tf.constant(10., shape=(self.v,)), trainable=True)

    # def call(self, inputs, training, l1_regularizers = 0):
    @tf.function
    def call(self, inputs, training):
        print("[DEBUG] initializing [call EqlLayerv2]")
        loss = 0.
        sigma = 1e-3
        out = tf.matmul(inputs, self.w) + self.b
        output_batches = []
        for i in range(self.v):
            v = (9 - self.exclusion) * i
            index = 0
            for key in self.activations_prop:
                activation = self.activations_prop[key](out, index + v)
                output_batches.append(activation)
                if key == "log":
                    loss += tf.reduce_sum(
                        tf.where(
                            tf.gather(out, [index + v], axis=1) <= sigma,
                            tf.abs(sigma - tf.where(tf.gather(out, [index + v], axis=1) <= sigma,
                                                    tf.gather(out, [index + v], axis=1),
                                                    0)),
                            0)
                    )
                if key == "div":
                    loss += tf.reduce_sum(
                        tf.where(
                            tf.gather(out, [index + 1 + v], axis=1) <= sigma,
                            tf.abs(sigma - tf.where(tf.gather(out, [index + 1 + v], axis=1) <= sigma,
                                                    tf.gather(out, [index + 1 + v], axis=1),
                                                    0)),
                            0)
                    )
                index += 2 if key == "mult" or key == "div" else 1
        output = tf.concat(output_batches, axis=1)

        self.add_loss(tf.divide(loss, tf.cast(tf.shape(out)[0], dtype=tf.float32)))

        return output

class DenseLayerv2(keras.layers.Layer):
    def __init__(self, w_initializer, b_initializer, lmbda=0, constraint=None ,mask=None, activation = None, dynamic_lmbda=False):
        super(DenseLayerv2, self).__init__()
        # self._is_reg_dyn = dynamic_lmbda
        if activation:
            self.activation = activation
        else:
            self.activation = tf.identity

        # if not dynamic_lmbda:
        #     self.regularizer = regularizers.L1(lmbda)
        # else:
        #     self.regularizer = L1L2_EQL()
        self.w_initializer = initializers.get(w_initializer)
        self.b_initializer = initializers.get(b_initializer)
        # self.mask = mask
        # self.constraint=constraint

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer=self.w_initializer,
            trainable=True, name="w",
            # regularizer=self.regularizer, constraint=self.constraint,
            )
        self.b = self.add_weight(
            shape=(1,), initializer=self.b_initializer, trainable=True, name="b",
            # regularizer=self.regularizer, constraint=self.constraint
        )
    @tf.function
    def call(self, inputs, training):
        print("[DEBUG] initializing [call DenseLayerv2]")
        return self.activation(tf.matmul(inputs, self.w) + self.b)
