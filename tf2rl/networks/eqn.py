"""Equation networks used by EQN-AIRL and EQN-GAIL.

These operators reproduce the repository formulas, including the squared-binomial
operator and the 0.1 relaxation. See docs/modernization.md for paper differences.
"""
import numpy as np
import tensorflow as tf


class EquationLayer(tf.keras.layers.Layer):
    OPERATORS = (("id", 1), ("sin", 1), ("cos", 1), ("sig", 1),
                 ("log", 1), ("mult", 2), ("div", 2))

    def __init__(self, w_initializer="random_normal", b_initializer="random_normal", v=1,
                 lmbda=0, constraint=None, mask=None, exclude=None, dynamic_lmbda=False,
                 **kwargs):
        super().__init__(**kwargs)
        if int(v) != v or v < 1:
            raise ValueError("Each equation layer needs a positive integer v")
        exclude = set(exclude or [])
        if exclude - {key for key, _ in self.OPERATORS}:
            raise ValueError(f"Unknown equation operators: {exclude}")
        self.operators = [(key, width) for key, width in self.OPERATORS if key not in exclude]
        if not self.operators:
            raise ValueError("Cannot exclude every equation operator")
        self.v = int(v)
        self.input_width = sum(width for _, width in self.operators)
        self.w_initializer = tf.keras.initializers.get(w_initializer)
        self.b_initializer = tf.keras.initializers.get(b_initializer)
        self.constraint = constraint
        # The historical v2 equation layer does not apply L1 or masks. Its loss is
        # the singular-unit penalty below, with coefficient one in the objectives.

    def build(self, input_shape):
        width = self.input_width * self.v
        self.w = self.add_weight(name="w", shape=(input_shape[-1], width),
                                 initializer=self.w_initializer, constraint=self.constraint)
        self.b = self.add_weight(name="b", shape=(width,), initializer=self.b_initializer,
                                 constraint=self.constraint)
        super().build(input_shape)

    def call(self, inputs, training=None, l1_regularizers=0):
        affine = tf.matmul(inputs, self.w) + self.b
        outputs = []
        penalties = []
        for group in range(self.v):
            index = group * self.input_width
            for operator, width in self.operators:
                a = affine[:, index:index + 1]
                if operator == "id":
                    value = a
                elif operator == "sin":
                    value = tf.sin(a)
                elif operator == "cos":
                    value = tf.cos(a)
                elif operator == "sig":
                    value = tf.sigmoid(a)
                elif operator == "log":
                    value = tf.math.log(tf.maximum(a, 0.0) + 0.1)
                    penalties.append(tf.reduce_sum(tf.nn.relu(0.001 - a)))
                else:
                    b = affine[:, index + 1:index + 2]
                    if operator == "mult":
                        value = tf.square(a + b)
                    else:
                        value = (a + 0.1) / (tf.maximum(b, 0.0) + 0.1)
                        penalties.append(tf.reduce_sum(tf.nn.relu(0.001 - b)))
                outputs.append(value)
                index += width
        loss = tf.add_n(penalties) if penalties else tf.zeros((), dtype=affine.dtype)
        self.add_loss(loss / tf.cast(tf.shape(affine)[0], affine.dtype))
        return tf.concat(outputs, axis=-1)


class EquationOutput(tf.keras.layers.Layer):
    def __init__(self, w_initializer="random_normal", b_initializer="random_normal",
                 activation=None, lmbda=0, constraint=None, mask=None, dynamic_lmbda=False,
                 regularize=False, **kwargs):
        super().__init__(**kwargs)
        self.w_initializer = tf.keras.initializers.get(w_initializer)
        self.b_initializer = tf.keras.initializers.get(b_initializer)
        self.activation = tf.keras.activations.get(activation)
        self.regularizer = tf.keras.regularizers.L1(lmbda) if regularize else None
        self.constraint = constraint if regularize else None

    def build(self, input_shape):
        self.w = self.add_weight(name="w", shape=(input_shape[-1], 1),
                                 initializer=self.w_initializer, regularizer=self.regularizer,
                                 constraint=self.constraint)
        self.b = self.add_weight(name="b", shape=(1,), initializer=self.b_initializer,
                                 regularizer=self.regularizer, constraint=self.constraint)
        super().build(input_shape)

    def call(self, inputs, training=None, l1_regularizers=0):
        return self.activation(tf.matmul(inputs, self.w) + self.b)


class EquationNetwork(tf.keras.Model):
    """Shared scalar EQN model; parameter names and ordering match the old networks."""
    def __init__(self, state_shape, action_dim=0, num_layers=2, drop_out=None,
                 enable_sn=False, output_activation=None, lmbda=0, name="Discriminator",
                 v=None, exclude=None, atol=0, constraint=None,
                 w_initializer="random_normal", b_initializer="random_normal",
                 is_lmbda_dynamic=False, regularize_output=False):
        super().__init__(name=name)
        self.num_layers = num_layers
        self.v = list(v) if v is not None else [1] * num_layers
        self.exclude = exclude or [[] for _ in range(num_layers)]
        rates = [0.] * num_layers if drop_out is None else (
            list(drop_out) if isinstance(drop_out, (list, tuple)) else [drop_out] * num_layers)
        if len(self.v) != num_layers or len(self.exclude) != num_layers or len(rates) != num_layers:
            raise ValueError("v, exclude and dropout lengths must equal num_layers")
        if not all(0 <= rate < 1 for rate in rates):
            raise ValueError("Dropout rates must lie in [0, 1)")
        self.eql_layers = []
        w_init = tf.keras.initializers.get(w_initializer)
        b_init = tf.keras.initializers.get(b_initializer)
        for index, (width, exclusions, rate) in enumerate(zip(self.v, self.exclude, rates)):
            layer = EquationLayer(w_init, b_init, width, exclude=exclusions,
                                  constraint=constraint, name=f"equation_{index}")
            self.eql_layers.append(layer)
            if rate:
                self.eql_layers.append(tf.keras.layers.Dropout(rate))
        self.layer_out = EquationOutput(w_init, b_init, activation=output_activation,
                                        lmbda=lmbda, regularize=regularize_output,
                                        constraint=constraint, name="equation_output")
        self(tf.zeros((1, state_shape[0] + action_dim), dtype=tf.float32), training=False)

    def call(self, inputs, training=None, l1_regularizers=0):
        # Historically an omitted training flag enabled EQN dropout.
        training = True if training is None else training
        features = tf.cast(inputs, tf.float32)
        for layer in self.eql_layers:
            features = layer(features, training=training)
        return self.layer_out(features, training=training)

    def compute_reward(self, inputs, l1_regularizers=0):
        probability = self(inputs, training=False)
        return tf.math.log(probability + 1e-8) - tf.math.log(1 - probability + 1e-8)
