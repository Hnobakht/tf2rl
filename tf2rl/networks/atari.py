import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten

from tf2rl.networks.layers import NoisyDense
from tf2rl.networks.actors import CategoricalActorCritic, CategoricalActor


class AtariBaseModel(tf.keras.Model):
    """Convolutional Atari feature extractor with optional noisy dense layers."""
    def __init__(self, name, enable_noisy_dqn=False):
        """Build the Atari convolutional layers and choose standard or noisy dense features."""
        super().__init__(name=name)

        DenseLayer = NoisyDense if enable_noisy_dqn else Dense
        base_layers = []

        base_layers.append(Conv2D(32, kernel_size=(8, 8), strides=(4, 4),
                                  padding='valid', activation='relu'))
        base_layers.append(Conv2D(64, kernel_size=(4, 4), strides=(2, 2),
                                  padding='valid', activation='relu'))
        base_layers.append(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                                  padding='valid', activation='relu'))
        base_layers.append(Flatten())
        base_layers.append(DenseLayer(512, activation='relu'))
        self.base_layers = base_layers

    def call(self, inputs):
        # TODO: This type conversion seems to be bottle neck
        """Scale byte images to floats and extract convolutional features."""
        features = tf.divide(tf.cast(inputs, tf.float32),
                             tf.constant(255.))

        for cur_layer in self.base_layers:
            features = cur_layer(features)

        return features


class AtariQFunc(AtariBaseModel):
    """Atari action-value network with optional dueling, noisy, and categorical heads."""
    def __init__(self, state_shape, action_dim, units=None,
                 name="QFunc", enable_dueling_dqn=False,
                 enable_noisy_dqn=False, enable_categorical_dqn=False,
                 n_atoms=51):
        """
        Configure the Atari action-value head and optional dueling, noisy, or categorical variants.
        """
        self._enable_dueling_dqn = enable_dueling_dqn
        self._enable_noisy_dqn = enable_noisy_dqn
        self._enable_categorical_dqn = enable_categorical_dqn
        if enable_categorical_dqn:
            self._action_dim = action_dim
            self._n_atoms = n_atoms
            action_dim = (action_dim + int(enable_dueling_dqn)) * n_atoms

        # Build base layers
        super().__init__(name, enable_noisy_dqn)
        DenseLayer = NoisyDense if enable_noisy_dqn else Dense

        self.fc2 = DenseLayer(action_dim, activation='linear')

        if self._enable_dueling_dqn and not enable_categorical_dqn:
            self.fc3 = DenseLayer(1, activation='linear')

        input_shape = (1,) + state_shape
        with tf.device("/cpu:0"):
            self(inputs=tf.constant(np.zeros(shape=input_shape,
                                             dtype=np.float32)))

    def call(self, inputs):
        # Extract features on base layers
        """Return action values or categorical value distributions from Atari images."""
        features = super().call(inputs)

        if self._enable_categorical_dqn:
            features = self.fc2(features)
            if self._enable_dueling_dqn:
                features = tf.reshape(
                    features, (-1, self._action_dim + 1, self._n_atoms))  # [batch_size, action_dim, n_atoms]
                v_values = tf.reshape(
                    features[:, 0], (-1, 1, self._n_atoms))
                advantages = tf.reshape(
                    features[:, 1:], [-1, self._action_dim, self._n_atoms])
                features = v_values + (advantages - tf.expand_dims(
                    tf.reduce_mean(advantages, axis=1), axis=1))
            else:
                features = tf.reshape(
                    features, (-1, self._action_dim, self._n_atoms))  # [batch_size, action_dim, n_atoms]
            # [batch_size, action_dim, n_atoms]
            q_dist = tf.keras.activations.softmax(features, axis=2)
            return tf.clip_by_value(q_dist, 1e-8, 1.0 - 1e-8)
        else:
            if self._enable_dueling_dqn:
                advantages = self.fc2(features)
                v_values = self.fc3(features)
                q_values = (v_values
                            + (advantages
                               - tf.reduce_mean(advantages, axis=1, keepdims=True)))
            else:
                q_values = self.fc2(features)
            return q_values


class AtariCategoricalActor(CategoricalActor, AtariBaseModel):
    """Categorical action policy built on Atari image features."""
    def __init__(self, state_shape, action_dim, units=None,
                 name="AtariCategoricalActor"):
        """Build the Atari feature extractor and categorical action head."""
        self.action_dim = action_dim

        # Build base layers
        AtariBaseModel.__init__(self, name, state_shape)

        # Build top layer
        self.out_prob = Dense(action_dim, activation='softmax')

        CategoricalActor.call(self, tf.constant(
            np.zeros(shape=(1,) + state_shape, dtype=np.uint8)))

    def _compute_features(self, states):
        # Extract features on base layers
        """Transform a batch of observations into policy features."""
        return AtariBaseModel.call(self, states)


class AtariCategoricalActorCritic(CategoricalActorCritic, AtariBaseModel):
    """Atari policy with shared image features and a state-value head."""
    def __init__(self, state_shape, action_dim, units=None,
                 name="AtariCategoricalActorCritic"):
        """Build Atari image features with categorical action and state-value heads."""
        AtariBaseModel.__init__(self, name=name)
        self.action_dim = action_dim

        self.out_prob = Dense(action_dim, activation='softmax')
        self.v = Dense(1, activation="linear")

        self(tf.constant(
            np.zeros(shape=(1,) + state_shape, dtype=np.float32)))

    def _compute_feature(self, states):
        """Extract convolutional features from a batch of Atari observations."""
        return AtariBaseModel.call(self, states)
