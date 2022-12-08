import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Conv2DTranspose

# Auxiliary layers used in PatchNets and an PatchInterface class. All PatchNets should extend this class

class ConvLayer(tf.Module):
    """
    Extended 3x3 convolution layer, including an optional BatchNormalization layer and a 'relu' activation layer afterwards.
    Can be implemented as a regular layer in a tensorflow model.
    """
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
    """
    Extended 3x3 transposed convolution layer, including an optional BatchNormalization layer and a 
    'relu' activation layer afterwards.
    Can be implemented as a regular layer in a tensorflow model.
    """
    def __init__(self, out_channels, name="TransConvLayer"):
        super(ConvTransposeLayer, self).__init__(name)
        self.conv = Conv2DTranspose(out_channels, 3, padding="same")
        self.batchnorm = BatchNormalization()
        self.relu = ReLU()

    def __call__(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        return self.relu(x)

class PatchInterface:
    """
    Skeleton class for PatchNets. All variations of PatchNets shall extend the class and code the actual implementations of
    the following methods:
    """
    def __call__(self, x):
        pass

    def save_weights(self, filename):
        pass

    def load_weights(self, filename):
        pass