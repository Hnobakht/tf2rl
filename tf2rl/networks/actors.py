import tensorflow as tf
import tensorflow_probability as tfp


class GaussianActor(tf.keras.Model):
    """Diagonal Gaussian with optional ``a = max_action * tanh(u)`` transform.

    ``u = mean(s) + exp(log_std(s))*epsilon`` is reparameterized for SAC.
    Returned log densities include the tanh Jacobian but historically omit
    ``sum(log(abs(max_action)))``. They are densities in normalized action
    coordinates, not environment coordinates when the scale differs from one.
    Retain this convention when comparing entropy targets and AIRL datasets.
    See Haarnoja et al. (2018), https://arxiv.org/abs/1801.01290, Appendix C.
    """
    LOG_STD_CAP_MAX = 2  # np.e**2 = 7.389
    LOG_STD_CAP_MIN = -20  # exp(-20) = 2.061e-9
    EPS = 1e-6

    def __init__(self, state_shape, action_dim, max_action, units=(256, 256),
                 hidden_activation="relu", state_independent_std=False,
                 squash=False, name='gaussian_policy',
                 initializer = tf.keras.initializers.glorot_uniform()):
        """Build Gaussian action heads and configure action scaling, standard deviations, and
        squashing.
        """
        super().__init__(name=name)

        layers = tf.keras.layers

        self._state_independent_std = state_independent_std
        self._squash = squash

        self.base_layers = []
        # self.base_layers.append(tfa.layers.SpectralNormalization(Dense(, name="L1", activation=activation, kernel_initializer=w_ini)))
        for unit in units:
            self.base_layers.append(layers.Dense(unit, activation=hidden_activation, kernel_initializer=initializer))
            self.base_layers.append(layers.LayerNormalization()) #TODO: For GAIL_EQL

        self.out_mean = layers.Dense(action_dim, name="L_mean", kernel_initializer=initializer)
        for index, layer in enumerate(self.base_layers):
            setattr(self, f"hidden_{index}", layer)
        if self._state_independent_std:
            self.out_logstd = self.add_weight(
                name="L_logstd", shape=(action_dim,),
                initializer=tf.keras.initializers.Constant(-0.5))
        else:
            self.out_logstd = layers.Dense(action_dim, name="L_logstd", kernel_initializer=initializer)

        self._max_action = max_action

    def _compute_dist(self, states):
        """

        Args:
            states: np.ndarray or tf.Tensor
                Inputs to neural network.

        Returns:
            tfp.distributions.MultivariateNormalDiag
                Multivariate normal distribution object whose mean and
                standard deviation is output of a neural network
        """
        features = states

        for cur_layer in self.base_layers:
            features = cur_layer(features)

        mean = self.out_mean(features)

        if self._state_independent_std:
            log_std = tf.tile(
                input=tf.expand_dims(self.out_logstd, axis=0),
                multiples=[tf.shape(mean)[0], 1])
        else:
            log_std = self.out_logstd(features)
            log_std = tf.clip_by_value(log_std, self.LOG_STD_CAP_MIN, self.LOG_STD_CAP_MAX)

        return tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=tf.exp(log_std))

    def call(self, states, training=None, test=False):
        """
        Compute actions and log probabilities of the selected action
        """
        dist = self._compute_dist(states)
        # Training passes a Python bool: avoid differentiating an unused tf.cond
        # branch, whose TFP broadcast gradients are incompatible with XLA.
        if isinstance(test, bool):
            raw_actions = dist.mean() if test else dist.sample()
        else:
            raw_actions = tf.cond(tf.cast(test, tf.bool), dist.mean, dist.sample)
        log_pis = dist.log_prob(raw_actions)

        if self._squash:
            actions = tf.tanh(raw_actions)
            # log pi(tanh u|s) = log N(u|s) - sum log(1 - tanh(u)^2).
            diff = tf.reduce_sum(tf.math.log(1 - actions ** 2 + self.EPS), axis=1)
            log_pis -= diff
        else:
            actions = raw_actions

        actions = actions * self._max_action
        return actions, log_pis

    def compute_log_probs(self, states, actions):
        """Evaluate supplied actions using the same density convention as ``call``.

        Divide out the action scale and invert tanh when enabled. Clipping
        makes the inverse finite at numerical saturation; the density there is
        an approximation. The constant scale Jacobian remains omitted.
        """
        raw_actions = actions / self._max_action
        dist = self._compute_dist(states)
        if self._squash:
            squashed = tf.clip_by_value(raw_actions, -1 + self.EPS, 1 - self.EPS)
            raw_actions = tf.atanh(squashed)
            correction = tf.reduce_sum(tf.math.log(1 - squashed ** 2 + self.EPS), axis=-1)
            return dist.log_prob(raw_actions) - correction
        return dist.log_prob(raw_actions)

    def compute_entropy(self, states):
        """Return latent Gaussian entropy, before tanh or action scaling.

        This analytic quantity is not the entropy of a squashed policy; SAC
        instead uses sampled, Jacobian-corrected log densities in its loss.
        """
        return self._compute_dist(states).entropy()

class GaussianActor_pixels(tf.keras.Model):
    """Gaussian continuous-action policy with convolutional image features."""
    LOG_STD_CAP_MAX = 2  # np.e**2 = 7.389
    LOG_STD_CAP_MIN = -20  # exp(-20) = 2.061e-9
    EPS = 1e-6

    def __init__(self, state_shape, action_dim, max_action, units=(256, 256),
                 hidden_activation="relu", state_independent_std=False,
                 squash=False, name='gaussian_policy',
                 initializer=tf.keras.initializers.glorot_uniform(),
                 is_pixel=False):
        """Build image features and Gaussian action heads with optional action squashing."""
        super().__init__(name=name)

        layers = tf.keras.layers

        self._state_independent_std = state_independent_std
        self._squash = squash

        self.base_layers = []
        self.base_layers.append(layers.Conv2D(4, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu'))
        self.base_layers.append(layers.Conv2D(8, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu'))
        self.base_layers.append(layers.MaxPool2D(pool_size=(2, 2)))
        self.base_layers.append(layers.Conv2D(8, kernel_size=(3, 3), padding='same', activation='relu'))
        self.base_layers.append(layers.Flatten())

        self.base_layers.append(layers.Dense(128, activation=hidden_activation, kernel_initializer=initializer))
        self.base_layers.append(layers.LayerNormalization())
        self.base_layers.append(layers.Dense(64, activation=hidden_activation, kernel_initializer=initializer))
        self.base_layers.append(layers.LayerNormalization())

        self.out_mean = layers.Dense(action_dim, name="L_mean", kernel_initializer=initializer)
        if self._state_independent_std:
            self.out_logstd = self.add_weight(
                name="L_logstd", shape=(action_dim,),
                initializer=tf.keras.initializers.Constant(-0.5))
        else:
            self.out_logstd = layers.Dense(action_dim, name="L_logstd", kernel_initializer=initializer)

        self._max_action = max_action

    def _compute_dist(self, states, training):
        """

        Args:
            states: np.ndarray or tf.Tensor
                Inputs to neural network.

        Returns:
            tfp.distributions.MultivariateNormalDiag
                Multivariate normal distribution object whose mean and
                standard deviation is output of a neural network
        """
        features = tf.cast(states/255, dtype=tf.float32)
        print("[DEBUG] input: " + str(features.shape) + " [_compute_dist call GaussianActor_pixels]")
        for cur_layer in self.base_layers:
            features = cur_layer(features, training=training)
            print("[DEBUG] features: " + str(features.shape))

        mean = self.out_mean(features, training=training)

        if self._state_independent_std:
            log_std = tf.tile(
                input=tf.expand_dims(self.out_logstd, axis=0),
                multiples=[tf.shape(mean)[0], 1])
        else:
            log_std = self.out_logstd(features, training=training)
            log_std = tf.clip_by_value(log_std, self.LOG_STD_CAP_MIN, self.LOG_STD_CAP_MAX)

        print("[DEBUG] mean: " + str(mean.shape) +  " log_std: " + str(log_std.shape))
        return tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=tf.exp(log_std))

    def call(self, states, training=None, test=False):
        """
        Compute actions and log probabilities of the selected action
        """
        dist = self._compute_dist(states, training)
        raw_actions = tf.cond(tf.cast(test, tf.bool), dist.mean, dist.sample)
        log_pis = dist.log_prob(raw_actions)

        if self._squash:
            actions = tf.tanh(raw_actions)
            diff = tf.reduce_sum(tf.math.log(1 - actions ** 2 + self.EPS), axis=1)
            log_pis -= diff
        else:
            actions = raw_actions

        actions = actions * self._max_action
        return actions, log_pis


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense


class CategoricalActor(tf.keras.Model):
    """Neural policy representing a categorical distribution over discrete actions."""
    def __init__(self, state_shape, action_dim, units=(256, 256),
                 hidden_activation="relu", name="CategoricalActor"):
        """Build hidden features and the categorical action-probability head."""
        super().__init__(name=name)
        self.action_dim = action_dim

        base_layers = []
        for cur_layer_size in units:
            cur_layer = tf.keras.layers.Dense(cur_layer_size, activation=hidden_activation)
            base_layers.append(cur_layer)

        self.base_layers = base_layers
        self.out_prob = Dense(action_dim, activation='softmax')

        self(tf.constant(
            np.zeros(shape=(1,)+state_shape, dtype=np.float32)))

    def _compute_features(self, states):
        """Transform a batch of observations into policy features."""
        features = states

        for cur_layer in self.base_layers:
            features = cur_layer(features)

        return features

    def _compute_dist(self, states):
        """

        Args:
            states: np.ndarray or tf.Tensor
                Inputs to neural network.

        Returns:
            tfp.distributions.Categorical
                Categorical distribution whose probabilities are
                computed using softmax activation of a neural network
        """
        features = self._compute_features(states)

        probs = self.out_prob(features)
        dist = tfp.distributions.Categorical(probs)

        return dist

    def compute_prob(self, states):
        """Return the categorical distribution logits for a batch of states."""
        dist = self._compute_dist(states)
        return dist.logits

    def call(self, states, test=False):
        """Return actions and log probabilities; use greedy actions in test mode."""
        dist = self._compute_dist(states)

        if test:
            action = tf.argmax(dist.logits, axis=1)  # (size,)
        else:
            action = dist.sample()  # (size,)
        log_prob = dist.log_prob(action)

        return action, log_prob

    def compute_entropy(self, states):
        """Return the action distribution entropy for each observation."""
        dist = self._compute_dist(states)
        return dist.entropy()

    def compute_log_probs(self, states, actions):
        """Compute log probabilities of state-action pairs

        Args:
            states: tf.Tensor
                Tensors of inputs to NN
            actions: tf.Tensor
                Tensors of NOT one-hot vector.
                They will be converted to one-hot vector inside this function.

        Returns:
            Log probabilities
        """
        dist = self._compute_dist(states)
        return dist.log_prob(actions)


class CategoricalActorCritic(CategoricalActor):
    """Categorical policy with a state-value head for actor-critic training."""
    def __init__(self, *args, **kwargs):
        """Create the categorical policy and its additional state-value head."""
        tf.keras.Model.__init__(self)
        self.v = Dense(1, activation="linear")
        super().__init__(*args, **kwargs)

    def call(self, states, test=False):
        """Return actions and log probabilities, plus state values; use greedy actions in test mode.
        """
        features = self._compute_feature(states)
        probs = self.out_prob(features)
        dist = tfp.distributions.Categorical(probs)

        if test:
            action = tf.argmax(dist.logits, axis=1)  # (size,)
        else:
            action = dist.sample()  # (size,)
        log_prob = dist.prob(action)

        v = self.v(features)

        return action, log_prob, v
