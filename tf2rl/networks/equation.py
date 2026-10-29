"""Equation networks used by EQN-AIRL and EQN-GAIL.

These operators reproduce the repository formulas, including the squared-binomial
operator and the 0.1 relaxation. See docs/modernization.md for paper differences.
"""
import math

import tensorflow as tf


class _EquationParameters(tf.keras.layers.Layer):
    """Shared equation-layer weight masking and permanent pruning operations."""
    def _build_masks(self):
        """Create non-trainable weight and bias masks when sparsity is enabled.

        Returns:
            None.
        """
        if self.sparse:
            self.w_mask = self.add_weight(name="w_mask", shape=self.w.shape,
                                          initializer="ones", trainable=False)
            self.b_mask = self.add_weight(name="b_mask", shape=self.b.shape,
                                          initializer="ones", trainable=False)

    def effective_parameters(self):
        """Return weights and biases with sparsity masks applied when enabled.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Weight matrix and bias vector after applying permanent masks,
                or the underlying variables if sparsity is disabled.
        """
        if self.sparse:
            return self.w * self.w_mask, self.b * self.b_mask
        return self.w, self.b

    def project_masks(self):
        """Force pruned equation weights and biases back to zero in place.

        Returns:
            None.
        """
        if self.sparse:
            self.w.assign(self.w * self.w_mask)
            self.b.assign(self.b * self.b_mask)

    def prune(self, threshold):
        """Permanently mask parameters below the threshold; require sparsity to be enabled.

        Args:
            threshold (float): Absolute parameter magnitude below which pruning permanently clears mask
                entries.

        Returns:
            None.

        Raises:
            ValueError: Construct the network with sparsity enabled before pruning.
        """
        if not self.sparse:
            raise ValueError("Construct the network with sparsity enabled before pruning")
        self.w_mask.assign(self.w_mask * tf.cast(tf.abs(self.w) >= threshold, self.w.dtype))
        self.b_mask.assign(self.b_mask * tf.cast(tf.abs(self.b) >= threshold, self.b.dtype))
        self.project_masks()


class EquationLayer(_EquationParameters):
    """Apply an operator dictionary to affine features ``z = xW+b``.

    With all operators enabled, each repetition consumes nine affine columns
    and emits seven features. ``mult(a,b)=(a+b)^2`` is the historical squared
    binomial, not ordinary multiplication. Log/division use a 0.1 relaxation;
    their domain penalty instead uses a 0.001 threshold. Neither constant is
    a physical unit conversion, so input scaling affects the learned equation.

    Related equation-learning methodology: Sahoo et al. (2018),
    https://proceedings.mlr.press/v80/sahoo18a.html. This operator dictionary
    is repository-specific and is not an exact EQL-division implementation.
    """
    OPERATORS = (("id", 1), ("sin", 1), ("cos", 1), ("sig", 1),
                 ("log", 1), ("mult", 2), ("div", 2))

    def __init__(self, w_initializer="random_normal", b_initializer="random_normal", v=1,
                 lmbda=0, constraint=None, mask=None, exclude=None, dynamic_lmbda=False,
                 sparse=False, **kwargs):
        """Configure operator groups, exclusions, initializers, and optional sparsity masks.

        Args:
            w_initializer (str or keras.initializers.Initializer): Initializer for equation-layer
                weights.
            b_initializer (str or keras.initializers.Initializer): Initializer for equation-layer
                biases.
            v (int): Positive number of repetitions of the enabled operator dictionary.
            lmbda (float): L1 regularization coefficient for equation parameters.
            constraint (callable or None): Keras weight constraint applied when parameters are updated.
            mask (array_like or None): Unused legacy mask argument; sparse=True creates persistent
                internal masks instead.
            exclude (sequence[str] or None): Operators to remove from id, sin, cos, sig, log, mult, div;
                at least one must remain.
            dynamic_lmbda (bool): Enable the historical mutable regularization-coefficient path where
                supported. This implementation does not read this argument.
            sparse (bool): Allocate persistent masks used for staged sparsity and pruning.
            **kwargs (dict): Additional keyword arguments forwarded to _EquationParameters construction.

        Returns:
            None: Initializes the instance.

        Raises:
            ValueError: Each equation layer needs a positive integer v. Cannot exclude every equation
                operator. lmbda must be finite and nonnegative.
        """
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
        self.sparse = sparse
        self.lmbda = float(lmbda)
        if not math.isfinite(self.lmbda) or self.lmbda < 0:
            raise ValueError("lmbda must be finite and nonnegative")

    def build(self, input_shape):
        """Create layer weights and biases for the input feature dimension.

        Args:
            input_shape (tuple[int, ...] or tf.TensorShape): Incoming tensor dimensions, including the
                batch axis.

        Returns:
            None.
        """
        width = self.input_width * self.v
        self.w = self.add_weight(name="w", shape=(input_shape[-1], width),
                                 initializer=self.w_initializer, constraint=self.constraint)
        self.b = self.add_weight(name="b", shape=(width,), initializer=self.b_initializer,
                                 constraint=self.constraint)
        self._build_masks()
        super().build(input_shape)

    def call(self, inputs, training=None, l1_regularizers=0):
        """Apply affine equation operators and add L1 and log/division domain penalties.

        Args:
            inputs (tf.Tensor): Input feature matrix shaped (batch, input_width); network input width is
                observation width plus action_dim.
            training (bool or tf.Tensor or None): Keras training flag controlling stochastic or stateful
                layers; None follows the layer default. This implementation does not read this argument.
            l1_regularizers (float): Historical per-call L1 coefficient; some modern equation paths
                retain but ignore it. This implementation does not read this argument.

        Returns:
            tf.Tensor: Operator features shaped (batch, v * number_of_enabled_operators). L1 and
                log/division domain penalties are registered in layer.losses.
        """
        w, b = self.effective_parameters()
        affine = tf.matmul(inputs, w) + b
        if self.lmbda:
            self.add_loss(self.lmbda * (tf.reduce_sum(tf.abs(w)) + tf.reduce_sum(tf.abs(b))))
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
                    # ReLU makes evaluation finite; the separate penalty supplies
                    # a restoring gradient for negative, otherwise flat inputs.
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


class EquationOutput(_EquationParameters):
    """Scalar affine equation output with optional activation, masking, and L1 loss."""
    def __init__(self, w_initializer="random_normal", b_initializer="random_normal",
                 activation=None, lmbda=0, constraint=None, mask=None, dynamic_lmbda=False,
                 regularize=False, sparse=False, **kwargs):
        """Configure the scalar output activation, initializers, and optional regularization.

        Args:
            w_initializer (str or keras.initializers.Initializer): Initializer for equation-layer
                weights.
            b_initializer (str or keras.initializers.Initializer): Initializer for equation-layer
                biases.
            activation (str or callable or None): Output activation; None leaves the affine output
                linear.
            lmbda (float): L1 regularization coefficient for equation parameters.
            constraint (callable or None): Keras weight constraint applied when parameters are updated.
            mask (array_like or None): Unused legacy mask argument; sparse=True creates persistent
                internal masks instead.
            dynamic_lmbda (bool): Enable the historical mutable regularization-coefficient path where
                supported. This implementation does not read this argument.
            regularize (bool): Enable output-weight L1 and constraint settings; False forces its L1
                coefficient to zero.
            sparse (bool): Allocate persistent masks used for staged sparsity and pruning.
            **kwargs (dict): Additional keyword arguments forwarded to _EquationParameters construction.

        Returns:
            None: Initializes the instance.
        """
        super().__init__(**kwargs)
        self.w_initializer = tf.keras.initializers.get(w_initializer)
        self.b_initializer = tf.keras.initializers.get(b_initializer)
        self.activation = tf.keras.activations.get(activation)
        # Make the coefficient floating point even for integer defaults and keep
        # the same L1 expression in the traced graph with the singular penalties.
        self.lmbda = float(lmbda) if regularize else 0.
        self.constraint = constraint if regularize else None
        self.sparse = sparse

    def build(self, input_shape):
        """Create layer weights and biases for the input feature dimension.

        Args:
            input_shape (tuple[int, ...] or tf.TensorShape): Incoming tensor dimensions, including the
                batch axis.

        Returns:
            None.
        """
        self.w = self.add_weight(name="w", shape=(input_shape[-1], 1),
                                 initializer=self.w_initializer, constraint=self.constraint)
        self.b = self.add_weight(name="b", shape=(1,), initializer=self.b_initializer,
                                 constraint=self.constraint)
        self._build_masks()
        super().build(input_shape)

    def call(self, inputs, training=None, l1_regularizers=0):
        """Return the scalar affine output with its configured activation and regularization.

        Args:
            inputs (tf.Tensor): Input feature matrix shaped (batch, input_width); network input width is
                observation width plus action_dim.
            training (bool or tf.Tensor or None): Keras training flag controlling stochastic or stateful
                layers; None follows the layer default. This implementation does not read this argument.
            l1_regularizers (float): Historical per-call L1 coefficient; some modern equation paths
                retain but ignore it. This implementation does not read this argument.

        Returns:
            tf.Tensor: Activated scalar outputs shaped (batch, 1), with configured parameter penalties
                registered in layer.losses.
        """
        w, b = self.effective_parameters()
        if self.lmbda:
            self.add_loss(tf.cast(self.lmbda, self.w.dtype)
                          * (tf.reduce_sum(tf.abs(w)) + tf.reduce_sum(tf.abs(b))))
        return self.activation(tf.matmul(inputs, w) + b)


class EquationNetwork(tf.keras.Model):
    """Shared scalar EQN model; parameter names and ordering match the old networks."""
    def __init__(self, state_shape, action_dim=0, num_layers=2, drop_out=None,
                 enable_sn=False, output_activation=None, lmbda=0, name="Discriminator",
                 v=None, exclude=None, atol=0, constraint=None,
                 w_initializer="random_normal", b_initializer="random_normal",
                 is_lmbda_dynamic=False, regularize_output=False, training_stages=None,
                 prune_threshold=0.001):
        """Build the equation layers and configure optional warmup, sparsity, and fixed-mask
        stages.

        Args:
            state_shape (tuple[int, ...]): Shape of one observation, excluding the batch axis.
            action_dim (int): Number of discrete choices or width of a continuous action vector.
            num_layers (int): Number of equation layers.
            drop_out (float or sequence[float] or None): Dropout probability, or one probability per
                equation layer; None disables dropout.
            enable_sn (bool): Request spectral normalization where the selected implementation supports
                it. This implementation does not read this argument.
            output_activation (str or callable or None): Activation used by the scalar or discriminator
                output layer.
            lmbda (float): Nonnegative L1 coefficient; a positive value enables persistent sparsity
                state.
            name (str): Name assigned to the model, operation, or registered component.
            v (sequence[int] or None): Positive operator repetition count per equation layer; None uses
                one per layer.
            exclude (sequence or None): Equation operators to exclude; network constructors use one
                exclusion list per layer.
            atol (float): Absolute weight threshold used by legacy tolerance-based inference copies.
                This implementation does not read this argument.
            constraint (callable or None): Keras weight constraint applied when parameters are updated.
            w_initializer (str or keras.initializers.Initializer): Initializer for equation-layer
                weights.
            b_initializer (str or keras.initializers.Initializer): Initializer for equation-layer
                biases.
            is_lmbda_dynamic (bool): Historical dynamic-regularization option; unsupported
                configurations may be rejected. This implementation does not read this argument.
            regularize_output (bool): Enable the equation output layer regularization used by state-only
                AIRL.
            training_stages (sequence[int] or None): Pair (warmup_updates, sparse_updates) followed by
                fixed-mask refitting. Counts are actual discriminator optimizer updates, not environment
                steps.
            prune_threshold (float): Prune parameters whose absolute values are below this threshold at
                the refitting boundary.

        Returns:
            None: Initializes the instance.

        Raises:
            ValueError: lmbda must be finite and nonnegative. prune_threshold must be finite and
                nonnegative. training_stages requires (warmup >= 0, sparse > 0) updates and lmbda > 0.
                v, exclude and dropout lengths must equal num_layers. Dropout rates must lie in [0, 1).
        """
        super().__init__(name=name)
        if not math.isfinite(lmbda) or lmbda < 0:
            raise ValueError("lmbda must be finite and nonnegative")
        if not math.isfinite(prune_threshold) or prune_threshold < 0:
            raise ValueError("prune_threshold must be finite and nonnegative")
        if training_stages is not None and (
                len(training_stages) != 2 or any(int(n) != n for n in training_stages)
                or training_stages[0] < 0 or training_stages[1] <= 0 or lmbda <= 0):
            raise ValueError("training_stages requires (warmup >= 0, sparse > 0) updates and lmbda > 0")
        self.sparse = lmbda > 0 or training_stages is not None
        if self.sparse:
            warmup, sparse_updates = training_stages or (0, 0)
            boundaries = [warmup, warmup + sparse_updates] if training_stages else [-1, -1]
            self.sparsity_boundaries = self.add_weight(name="sparsity_boundaries", shape=(2,),
                dtype="int64", initializer=tf.keras.initializers.Constant(boundaries), trainable=False)
            self.sparsity_updates = self.add_weight(name="sparsity_updates", shape=(),
                dtype="int64", initializer="zeros", trainable=False)
            self.sparsity_phase = self.add_weight(name="sparsity_phase", shape=(), dtype="int64",
                initializer=tf.keras.initializers.Constant(0 if warmup else 1), trainable=False)
            self.sparsity_lambda = self.add_weight(name="sparsity_lambda", shape=(),
                initializer=tf.keras.initializers.Constant(lmbda), trainable=False)
            self.l1_coefficient = self.add_weight(name="l1_coefficient", shape=(),
                initializer=tf.keras.initializers.Constant(0 if warmup else lmbda), trainable=False)
            self.prune_threshold = self.add_weight(name="prune_threshold", shape=(),
                initializer=tf.keras.initializers.Constant(prune_threshold), trainable=False)
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
                                  constraint=constraint, sparse=self.sparse, name=f"equation_{index}")
            # Keras 3 tracks the variables for optimization when a layer is in a
            # plain list, but TensorFlow checkpoints only discover attribute
            # dependencies.  Give every equation layer a stable trackable path.
            setattr(self, f"equation_{index}", layer)
            self.eql_layers.append(layer)
            if rate:
                dropout = tf.keras.layers.Dropout(rate, name=f"dropout_{index}")
                setattr(self, f"dropout_{index}", dropout)
                self.eql_layers.append(dropout)
        self.layer_out = EquationOutput(w_init, b_init, activation=output_activation,
                                        lmbda=0, regularize=regularize_output, sparse=self.sparse,
                                        constraint=constraint, name="equation_output")
        self(tf.zeros((1, state_shape[0] + action_dim), dtype=tf.float32), training=False)

    def call(self, inputs, training=None, l1_regularizers=0):
        # Historically an omitted training flag enabled EQN dropout.
        """Evaluate the equation network and add active L1 losses; omitted training enables
        dropout.

        Args:
            inputs (tf.Tensor): Input feature matrix shaped (batch, input_width); network input width is
                observation width plus action_dim.
            training (bool or tf.Tensor or None): Keras training flag controlling stochastic or stateful
                layers; None follows the layer default.
            l1_regularizers (float): Historical per-call L1 coefficient; some modern equation paths
                retain but ignore it. This implementation does not read this argument.

        Returns:
            tf.Tensor: Scalar network predictions shaped (batch, 1); current equation-domain and
                sparsity penalties are retained in model.losses.
        """
        training = True if training is None else training
        features = tf.cast(inputs, tf.float32)
        for layer in self.eql_layers:
            features = layer(features, training=training)
        output = self.layer_out(features, training=training)
        if self.sparse:
            self.add_loss(self.l1_coefficient * self.l1_norm())
        return output

    @property
    def parameter_layers(self):
        """Return equation layers and the output layer, excluding dropout layers.

        Returns:
            list: Equation hidden layers and scalar output layer in forward order; dropout layers are
                excluded.
        """
        return [layer for layer in self.eql_layers if isinstance(layer, EquationLayer)] + [self.layer_out]

    def l1_norm(self):
        """Return the sum of absolute effective equation weights and biases.

        Returns:
            tf.Tensor: Scalar sum of absolute effective weights and biases across parameter layers.
        """
        return tf.add_n([tf.reduce_sum(tf.abs(value)) for layer in self.parameter_layers
                         for value in layer.effective_parameters()])

    def prepare_update(self):
        """Select warmup, L1 fitting, or fixed-support fitting before an update.

        Boundaries count completed discriminator optimizer steps, not physical
        environment steps. The final phase prunes once, removes L1 shrinkage,
        and refits surviving coefficients; masks and counters are checkpointed.

        Returns:
            None.
        """
        if not self.sparse:
            return
        start, end = self.sparsity_boundaries.numpy()
        updates = int(self.sparsity_updates.numpy())
        if end < 0:
            return
        phase = 0 if updates < start else (1 if updates < end else 2)
        if phase == 2 and int(self.sparsity_phase.numpy()) != 2:
            for layer in self.parameter_layers:
                layer.prune(self.prune_threshold)
        self.sparsity_phase.assign(phase)
        self.l1_coefficient.assign(self.sparsity_lambda if phase == 1 else 0.)

    def finish_update(self):
        """Reapply pruning masks and count one completed discriminator update.

        Returns:
            None.
        """
        if self.sparse:
            # Adam momentum can move a masked weight despite a zero gradient.
            for layer in self.parameter_layers:
                layer.project_masks()
            self.sparsity_updates.assign_add(1)

    def sparsity_metrics(self):
        """Return the sparsity phase, active parameter count, L1 norm, and update diagnostics.

        Returns:
            dict: Phase, active L1 coefficient/norm, active-parameter count, pruned fraction, and
                discriminator optimizer-update count.
        """
        total = sum(math.prod(value.shape) for layer in self.parameter_layers
                    for value in (layer.w, layer.b))
        active = (tf.add_n([tf.reduce_sum(value) for layer in self.parameter_layers
                           for value in (layer.w_mask, layer.b_mask)])
                  if self.sparse else tf.cast(total, tf.float32))
        return {"phase": self.sparsity_phase if self.sparse else -1,
                "l1_coefficient": self.l1_coefficient if self.sparse else 0.,
                "l1_norm": self.l1_norm(), "active_parameters": active,
                "pruned_fraction": 1. - active / total,
                "discriminator_updates": self.sparsity_updates if self.sparse else 0}

    def write_sparsity_summaries(self):
        """Write equation sparsity diagnostics to TensorBoard when sparsity is enabled.

        Returns:
            None.
        """
        if self.sparse:
            for name, value in self.sparsity_metrics().items():
                tf.summary.scalar("EQN/" + name, value)

    def compute_reward(self, inputs, l1_regularizers=0):
        """Return stabilized discriminator log odds for each input sample.

        Args:
            inputs (tf.Tensor): Input feature matrix shaped (batch, input_width); network input width is
                observation width plus action_dim.
            l1_regularizers (float): Historical per-call L1 coefficient; some modern equation paths
                retain but ignore it. This implementation does not read this argument.

        Returns:
            tf.Tensor: Log-odds rewards log(p+epsilon)-log(1-p+epsilon) shaped (batch, 1). The network
                output must represent a probability.
        """
        probability = self(inputs, training=False)
        return tf.math.log(probability + 1e-8) - tf.math.log(1 - probability + 1e-8)
