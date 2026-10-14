import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers

def CNN_model():
    base_layers = []
    # base_layers.append(layers.experimental.preprocessing.Rescaling(scale=1. / 255))
    # base_layers.append(layers.experimental.preprocessing.Resizing(32, 32))
    base_layers.append(layers.Conv2D(32, kernel_size=(8, 8), strides=(4,4), padding='same', activation='relu'))
    # base_layers.append(layers.MaxPool2D(pool_size=(2,2)))
    base_layers.append(layers.Conv2D(64, kernel_size=(4, 4), strides=(2,2), padding='same', activation='relu'))
    # base_layers.append(layers.Dropout(0.2))
    # base_layers.append(layers.MaxPool2D(pool_size=(2, 2)))
    base_layers.append(layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    # base_layers.append(layers.Dropout(0.2))
    base_layers.append(layers.Flatten())

    fnn_layer = []
    fnn_layer.append(layers.Dense(1024, activation=hidden_activation, kernel_initializer=initializer))
    # fnn_layer.append(layers.Dropout(0.2))
    # fnn_layer.append(layers.LayerNormalization())
    fnn_layer.append(layers.Dense(512, activation=hidden_activation, kernel_initializer=initializer))
    # fnn_layer.append(layers.Dropout(0.2))
    # fnn_layer.append(layers.LayerNormalization())
    return b
