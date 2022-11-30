import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Conv2DTranspose

class ConvLayer(tf.Module):
    def __init__(self, out_channels, batchNorm=True, name="ConvLayer"):
        super(ConvLayer, self).__init__(name)
        self.conv = Conv2D(out_channels, 3, padding="same")
        self.batchNorm = batchNorm
        if batchNorm:
            self.batchnorm = BatchNormalization()
        self.relu = ReLU()

    def __call__(self, x):
        x = self.conv(x)
        if self.batchNorm:
            x = self.batchnorm(x)
        return self.relu(x)


class ConvTransposeLayer(tf.Module):
    def __init__(self, out_channels, name="ConvLayer"):
        super(ConvTransposeLayer, self).__init__(name)
        self.conv = Conv2DTranspose(out_channels, 3, padding="same")
        self.batchnorm = BatchNormalization()
        self.relu = ReLU()

    def __call__(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        return self.relu(x)

class PatchInterface:
    def save_weights(self, filename):
        pass

    def load_weights(self, filename):
        pass