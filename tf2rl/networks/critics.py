"""Shared SAC action-value and policy-gradient value networks."""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, LayerNormalization, MaxPool2D


class CriticQ(tf.keras.Model):
    """Neural action-value function over state-action pairs."""
    def __init__(self, state_shape, action_dim, critic_units=(256, 256), name='qf',
                 initializer=None, layer_norm=True):
        """Build the hidden state-action layers and scalar Q-value head."""
        super().__init__(name=name)
        initializer = initializer or tf.keras.initializers.GlorotUniform()

        self.base_layers = []
        for unit in critic_units:
            self.base_layers.append(Dense(unit, activation='relu', kernel_initializer=initializer))
            if layer_norm:
                self.base_layers.append(LayerNormalization())
        self.out_layer = Dense(1, name="Q", activation='linear', kernel_initializer=initializer)
        for index, layer in enumerate(self.base_layers):
            setattr(self, f"hidden_{index}", layer)


    def call(self, states, actions, training=None):
        """Return Q-value estimates for a batch of state-action pairs."""
        features = tf.concat((states, actions), axis=1)
        for cur_layer in self.base_layers:
            features = cur_layer(features)
        values = self.out_layer(features)
        return tf.squeeze(values, axis=1)


class CriticQ_pixels(tf.keras.Model):
    """Action-value function using convolutional features from pixel observations."""
    def __init__(self, state_shape, action_dim, critic_units=None, name='qf',
                 initializer=None, layer_norm=True):
        """Build the image encoder and state-action Q-value layers."""
        super().__init__(name=name)
        initializer = initializer or tf.keras.initializers.GlorotUniform()

        self.cnn_layers = []
        self.cnn_layers.append(Conv2D(4, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu'))
        self.cnn_layers.append(Conv2D(8, kernel_size=(3, 3), strides=(1,1), padding='same',  activation='relu'))
        self.cnn_layers.append(MaxPool2D(pool_size=(2, 2)))
        self.cnn_layers.append(Conv2D(8, kernel_size=(3, 3), padding='same', activation='relu'))
        self.cnn_layers.append(Flatten())

        self.base_layers = []
        self.base_layers.append(Dense(128, activation='relu', kernel_initializer=initializer))
        if layer_norm:
            self.base_layers.append(LayerNormalization())
        self.base_layers.append(Dense(64, activation='relu', kernel_initializer=initializer))
        if layer_norm:
            self.base_layers.append(LayerNormalization())

        self.out_layer = Dense(1, name="Q", activation='linear', kernel_initializer=initializer)

    def call(self, states, actions, training=None):
        """Return Q-value estimates for a batch of state-action pairs."""
        features_cnn = tf.cast(states/255, dtype=tf.float32)
        print("[DEBUG] input: " + str(states.shape) + " [call CriticQ_pixels]")
        for cnn_layer in self.cnn_layers:
            features_cnn = cnn_layer(features_cnn, training=training)
            print("[DEBUG] features: " + str(features_cnn.shape))

        features = tf.concat((features_cnn, actions), axis=1)
        for cur_layer in self.base_layers:
            features = cur_layer(features)
            print("[DEBUG] features: " + str(features.shape))
        values = self.out_layer(features, training=training)
        return tf.squeeze(values, axis=1)


class CriticV(tf.keras.Model):
    """Neural state-value function for policy-gradient agents."""
    def __init__(self, state_shape, units, name='critic_v', hidden_activation='relu',
                 device='/cpu:0'):
        """Build the hidden state layers and scalar value head."""
        super().__init__(name=name)

        self.l1 = Dense(units[0], name="L1", activation=hidden_activation)
        self.l2 = Dense(units[1], name="L2", activation=hidden_activation)
        self.l3 = Dense(1, name="L2", activation='linear')

        with tf.device(device):
            self(tf.constant(
                np.zeros(shape=(1,)+state_shape, dtype=np.float32)))

    def call(self, inputs):
        """Return a scalar state-value estimate for each observation."""
        features = self.l1(inputs)
        features = self.l2(features)
        values = self.l3(features)

        return tf.squeeze(values, axis=1)

