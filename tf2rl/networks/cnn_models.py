from tensorflow.keras import layers

def CNN_model():
    base_layers = []
    base_layers.append(layers.Conv2D(32, kernel_size=(8, 8), strides=(4,4), padding='same', activation='relu'))
    base_layers.append(layers.Conv2D(64, kernel_size=(4, 4), strides=(2,2), padding='same', activation='relu'))
    base_layers.append(layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    base_layers.append(layers.Flatten())

    fnn_layer = []
    fnn_layer.append(layers.Dense(1024, activation=hidden_activation, kernel_initializer=initializer))
    fnn_layer.append(layers.Dense(512, activation=hidden_activation, kernel_initializer=initializer))
    return b
