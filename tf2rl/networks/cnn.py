from tensorflow.keras import layers

def CNN_model(hidden_activation='relu', initializer='glorot_uniform'):
    """Return ordered convolutional and dense feature layers for NHWC images.

    Wrap the returned list in ``tf.keras.Sequential`` or apply it in order.
    The final feature width is 512; callers supply any task-specific head and
    image normalization. This helper does not construct a policy by itself.
    """
    base_layers = []
    base_layers.append(layers.Conv2D(32, kernel_size=(8, 8), strides=(4,4), padding='same', activation='relu'))
    base_layers.append(layers.Conv2D(64, kernel_size=(4, 4), strides=(2,2), padding='same', activation='relu'))
    base_layers.append(layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    base_layers.append(layers.Flatten())

    fnn_layer = []
    fnn_layer.append(layers.Dense(1024, activation=hidden_activation, kernel_initializer=initializer))
    fnn_layer.append(layers.Dense(512, activation=hidden_activation, kernel_initializer=initializer))
    return base_layers + fnn_layer
