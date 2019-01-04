from keras.models import Model
from keras.layers import Input, Dense, Conv2D
from keras.layers import Flatten, MaxPool2D
from keras.layers import BatchNormalization


def vggnet(input_shape, classes):
    filters = 64, 128, 256, 512, 512,
    repetitions = 2, 2, 3, 3, 3

    def vgg_block(x, f, r):
        for i in range(r):
            x = Conv2D(f, 3, padding="same", activation="relu")
        x = MaxPool2D(2, strides=2)(x)

        return x

    input = Input(input_shape)

    x = input
    for f, r in zip(filters, repetitions):
        x = vgg_block(x, f, r)

    x = Flatten()(x)
    x = Dense(4096, activation="relu")(x)
    x = Dense(4096, activation="relu")(x)
    output = Dense(classes, activation="softmax")(x)

    model = Model(input, output)

    return model
