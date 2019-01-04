from keras.models import Model
from keras.layers import Input, Dense, Conv2D
from keras.layers import Flatten, MaxPool2D
from keras.layers import BatchNormalization


def alexnet(input_shape, classes):
    input = Input(input_shape)

    x = Conv2D(96, 11, strides=4, padding="same", activation="relu")(input)
    x = BatchNormalization()(x)
    x = MaxPool2D(3, strides=2)(x)

    x = Conv2D(256, 5, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(3, strides=2)(x)

    x = Conv2D(384, 3, padding="same", activation="relu")(x)
    x = Conv2D(384, 3, padding="same", activation="relu")(x)
    x = Conv2D(256, 3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(3, strides=2)(x)

    x = Flatten()
    x = Dense(4096, activation="relu")(x)
    x = Dense(4096, activation="relu")(x)

    output = Dense(classes, activation="softmax")(x)

    model = Model(input, output)

    return model
