import tensorflow as tf

from tensorflow.keras.layers import AveragePooling2D, Conv2D, MaxPooling2D, concatenate

# Inception v4 Modules, from https://arxiv.org/pdf/1602.07261.pdf

# Parameters
# previous: Previous layer of the net.
# activation: activation function for Convolutional layers (in tensorflow format)(optional)
def inceptionv4_A(previous, name='inception_A', activation=None):
    with tf.name_scope(name):
        b1 = AveragePooling2D(pool_size=(1, 1), padding='same')(previous)
        b1 = Conv2D(filters=96, kernel_size=(1, 1), activation=activation, padding='same')(b1)

        b2 = Conv2D(filters=96, kernel_size=(1, 1), activation=activation, padding='same')(previous)

        b3 = Conv2D(filters=64, kernel_size=(1, 1), activation=activation, padding='same')(previous)
        b3 = Conv2D(filters=96, kernel_size=(3, 3), activation=activation, padding='same')(b3)

        b4 = Conv2D(filters=64, kernel_size=(1, 1), activation=activation, padding='same')(previous)
        b4 = Conv2D(filters=96, kernel_size=(3, 3), activation=activation, padding='same')(b4)
        b4 = Conv2D(filters=96, kernel_size=(3, 3), activation=activation, padding='same')(b4)

        concat = concatenate([b1, b2, b3, b4], axis=3)

    return concat


# Parameters
# previous: Previous layer of the net.
# activation: activation function for Convolutional layers (in tensorflow format)(optional)
def inceptionv4_B(previous, name='inception_B', activation=None):
    with tf.name_scope(name):
        b1 = AveragePooling2D(pool_size=(1, 1), padding='same')(previous)
        b1 = Conv2D(filters=128, kernel_size=(1, 1), activation=activation, padding='same')(b1)

        b2 = Conv2D(filters=384, kernel_size=(1, 1), activation=activation, padding='same')(previous)

        b3 = Conv2D(filters=192, kernel_size=(1, 1), activation=activation, padding='same')(previous)
        b3 = Conv2D(filters=224, kernel_size=(1, 7), activation=activation, padding='same')(b3)
        b3 = Conv2D(filters=256, kernel_size=(1, 7), activation=activation, padding='same')(b3)

        b4 = Conv2D(filters=192, kernel_size=(1, 1), activation=activation, padding='same')(previous)
        b4 = Conv2D(filters=192, kernel_size=(1, 7), activation=activation, padding='same')(b4)
        b4 = Conv2D(filters=224, kernel_size=(7, 1), activation=activation, padding='same')(b4)
        b4 = Conv2D(filters=224, kernel_size=(1, 7), activation=activation, padding='same')(b4)
        b4 = Conv2D(filters=256, kernel_size=(7, 1), activation=activation, padding='same')(b4)

        concat = concatenate([b1, b2, b3, b4], axis=3)

    return concat


# Parameters
# previous: Previous layer of the net.
# activation: activation function for Convolutional layers (in tensorflow format)(optional)
def inceptionv4_C(previous, name='inception_C', activation=None):
    with tf.name_scope(name):
        b1 = AveragePooling2D(pool_size=(1, 1), padding='same')(previous)
        b1 = Conv2D(filters=256, kernel_size=(1, 1), activation=activation, padding='same')(b1)

        b2 = Conv2D(filters=256, kernel_size=(1, 1), activation=activation, padding='same')(previous)

        b3 = Conv2D(filters=384, kernel_size=(1, 1), activation=activation, padding='same')(previous)
        b3_1 = Conv2D(filters=256, kernel_size=(1, 3), activation=activation, padding='same')(b3)
        b3_2 = Conv2D(filters=256, kernel_size=(3, 1), activation=activation, padding='same')(b3)

        b4 = Conv2D(filters=384, kernel_size=(1, 1), activation=activation, padding='same')(previous)
        b4 = Conv2D(filters=448, kernel_size=(1, 3), activation=activation, padding='same')(b4)
        b4 = Conv2D(filters=512, kernel_size=(3, 1), activation=activation, padding='same')(b4)
        b4_1 = Conv2D(filters=256, kernel_size=(3, 1), activation=activation, padding='same')(b4)
        b4_2 = Conv2D(filters=256, kernel_size=(1, 3), activation=activation, padding='same')(b4)

        concat = concatenate([b1, b2, b3_1, b3_2, b4_1, b4_2], axis=3)

    return concat


# Parameters
# previous: Previous layer of the net.
# activation: activation function for Convolutional layers (in tensorflow format)(optional)
def inceptionv4_stem(previous, name='inception_stem', activation=None):
    with tf.name_scope(name):
        c1 = Conv2D(filters=32, kernel_size=(3, 3), activation=activation, strides=2, padding='valid')(previous)
        c1 = Conv2D(filters=32, kernel_size=(3, 3), activation=activation, padding='valid')(c1)
        c1 = Conv2D(filters=64, kernel_size=(3, 3), activation=activation, padding='same')(c1)

        b1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(c1)
        b2 = Conv2D(filters=96, kernel_size=(3, 3), activation=activation, strides=2, padding='valid')(c1)

        c1 = concatenate([b1, b2], axis=3)

        b1 = Conv2D(filters=64, kernel_size=(1, 1), activation=activation, padding='same')(c1)
        b1 = Conv2D(filters=96, kernel_size=(3, 3), activation=activation, padding='valid')(b1)
        b2 = Conv2D(filters=64, kernel_size=(1, 1), activation=activation, padding='same')(c1)
        b2 = Conv2D(filters=64, kernel_size=(7, 1), activation=activation, padding='same')(b2)
        b2 = Conv2D(filters=64, kernel_size=(1, 7), activation=activation, padding='same')(b2)
        b2 = Conv2D(filters=96, kernel_size=(3, 3), activation=activation, padding='valid')(b2)

        c2 = concatenate([b1, b2], axis=3)

        b1 = Conv2D(filters=192, kernel_size=(3, 3), activation=activation, padding='valid')(c2)
        b2 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='valid')(
            c2)  # 'b2 = MaxPooling2D(pool_size=(2,2), strides=2, padding='valid')(c2)' according to the paper

        concat = concatenate([b1, b2], axis=3)

    return concat


# Parameters
# previous: Previous layer of the net.
# activation: activation function for Convolutional layers (in tensorflow format)(optional)
def inceptionv4_reductionA(previous, name='reduction_A', activation=None, n=384, k=192, l=224, m=256):
    with tf.name_scope(name):
        b1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(previous)

        b2 = Conv2D(filters=n, kernel_size=(3, 3), activation=activation, strides=2, padding='valid')(previous)

        b3 = Conv2D(filters=k, kernel_size=(1, 1), activation=activation, padding='same')(previous)
        b3 = Conv2D(filters=l, kernel_size=(3, 3), activation=activation, padding='same')(b3)
        b3 = Conv2D(filters=m, kernel_size=(3, 3), activation=activation, strides=2, padding='valid')(b3)

        concat = concatenate([b1, b2, b3], axis=3)

    return concat


# Parameters
# previous: Previous layer of the net.
# activation: activation function for Convolutional layers (in tensorflow format)(optional)
def inceptionv4_reductionB(previous, name='reduction_B', activation=None):
    with tf.name_scope(name):
        b1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(previous)

        b2 = Conv2D(filters=192, kernel_size=(1, 1), activation=activation, padding='same')(previous)
        b2 = Conv2D(filters=192, kernel_size=(3, 3), activation=activation, strides=2, padding='valid')(b2)

        b3 = Conv2D(filters=256, kernel_size=(1, 1), activation=activation, padding='same')(previous)
        b3 = Conv2D(filters=256, kernel_size=(1, 7), activation=activation, padding='same')(b3)
        b3 = Conv2D(filters=320, kernel_size=(7, 1), activation=activation, padding='same')(b3)
        b3 = Conv2D(filters=320, kernel_size=(3, 3), activation=activation, strides=2, padding='valid')(b3)

        concat = concatenate([b1, b2, b3], axis=3)

    return concat