import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Conv2DTranspose

class ConvLayer(tf.Module):
    def __init__(self, out_channels, name="ConvLayer"):
        super(ConvLayer, self).__init__(name)
        self.conv = Conv2D(out_channels, 3, padding="same")
        self.batchnorm = BatchNormalization()
        self.relu = ReLU()

    def __call__(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        return self.relu(x)


class ConvTranposeLayer(tf.Module):
    def __init__(self, out_channels, name="ConvLayer"):
        super(ConvTranposeLayer, self).__init__(name)
        self.conv = Conv2DTranspose(out_channels, 3, padding="same")
        self.batchnorm = BatchNormalization()
        self.relu = ReLU()

    def __call__(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        return self.relu(x)

class PatchInterface:
    def getNets(self):
        return self.depthNet, self.normalNet