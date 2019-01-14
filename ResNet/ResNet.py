from keras.models import Model
from keras.layers import Input, Dense, Conv2D
from keras.layers import Flatten, MaxPool2D, AvgPool2D
from keras.layers import BatchNormalization, add, ReLU


def resnet(input_shape, n_classes):
    def conv_bn_rl(x, f, k=1, s=1, p='same'):
        x = Conv2D(f, k, strides=s, padding=p)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    def conv_block(tensor, f1, f2, s):
        x = conv_bn_rl(tensor, f1)
        x = conv_bn_rl(x, f1, 3, s=s)
        x = Conv2D(f2, 1)(x)
        x = BatchNormalization()(x)

        shortcut = Conv2D(f2, 1, strides=s, padding='same')(tensor)
        shortcut = BatchNormalization()(shortcut)

        x = add([shortcut, x])
        output = ReLU()(x)

        return output

    def identity_block(tensor, f1, f2):
        x = conv_bn_rl(tensor, f1)
        x = conv_bn_rl(x, f1, 3)
        x = Conv2D(f2, 1)(x)
        x = BatchNormalization()(x)

        x = add([tensor, x])
        output = ReLU()(x)

        return output

    def resnet_block(x, f1, f2, r, s=2):
        x = conv_block(x, f1, f2, s)

        for _ in range(r - 1):
            x = identity_block(x, f1, f2)

        return x

    input = Input(input_shape)

    x = conv_bn_rl(input, 64, 7, 2)
    x = MaxPool2D(3, strides=2, padding='same')(x)

    x = resnet_block(x, 64, 256, 3, 1)
    x = resnet_block(x, 128, 512, 4)
    x = resnet_block(x, 256, 1024, 6)
    x = resnet_block(x, 512, 2048, 3)

    x = AvgPool2D(7, strides=1)(x)
    x = Flatten()(x)

    output = Dense(n_classes, activation='softmax')(x)
    model = Model(input, output)

    return model
