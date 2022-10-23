# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 19:19:22 2022

@author: Marc Johler
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPool2D, UpSampling2D

batch_size = 32
patch_size = 128
min_channels = 32

class ConvLayer(tf.Module):
    def __init__(self, out_channels, name = "ConvLayer"):
        super(ConvLayer, self).__init__(name)
        self.conv = Conv2D(out_channels, 3, padding = "same")
        self.batchnorm = BatchNormalization()
        self.relu = ReLU()  
    
    def __call__(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        return self.relu(x)
    
class Encoder_common(tf.Module):
    def __init__(self, input_size, min_channels, name = "Encoder_common"):
        super(Encoder_common, self).__init__(name)
        self.layers = tf.keras.Sequential([
            ConvLayer(min_channels),
            ConvLayer(min_channels),
            MaxPool2D(2),
            ConvLayer(min_channels * 2),
            ConvLayer(min_channels * 2),
            MaxPool2D(2),
            ConvLayer(min_channels * 4),
            ConvLayer(min_channels * 4),
            ConvLayer(min_channels * 4),
            MaxPool2D(2),
            ConvLayer(min_channels * 8),
            ConvLayer(min_channels * 8),
            ConvLayer(min_channels * 8),
            MaxPool2D(2)
            ])
        
        self.layers.build(input_size)
    
    def __call__(self, x):
        self.layers(x)
        
class Decoder(tf.Module):
    def __init__(self, input_size, min_channels, out_channels, name = "decoder"):
        super(Decoder, self).__init__(name)
        self.layers = tf.keras.Sequential([
            UpSampling2D(),
            ConvLayer(min_channels * 8),
            ConvLayer(min_channels * 8),
            ConvLayer(min_channels * 8),
            UpSampling2D(),
            ConvLayer(min_channels * 8),
            ConvLayer(min_channels * 8),
            ConvLayer(min_channels * 8),
            UpSampling2D(),
            ConvLayer(min_channels * 4),
            ConvLayer(min_channels * 4),
            ConvLayer(min_channels * 4),
            UpSampling2D(),
            ConvLayer(min_channels * 2),
            ConvLayer(min_channels * 2),
            UpSampling2D(),
            ConvLayer(min_channels),
            ConvLayer(min_channels),
            Conv2D(out_channels, 1)
            ])
        
        self.layers.build(input_size)
    
    def __call__(self, x):
        self.layers(x)
        

class PatchNet(tf.Module):
    def __init__(self, batch_size, patch_size, min_channels, name = "patchnet"):
        super(PatchNet, self).__init__(name)
        input_size = (batch_size, patch_size, patch_size, 3)
        encoded_size = (batch_size, int(patch_size / 32), int(patch_size / 32), min_channels * 8)
        self.encoder = Encoder_common(input_size, min_channels)
        self.depth_decoder = Decoder(encoded_size, min_channels, 1, "depth_decoder")
        self.normals_decoder = Decoder(encoded_size, min_channels, 3, "normals_decoder")
    
    def __call__(self, x):
        encoded = self.encoder(x)
        depth_map = self.depth_decoder(encoded)
        normals_map = self.normals_decoder(encoded)
        return depth_map, normals_map


input_size = (batch_size, patch_size, patch_size, 3)
encoder_common = Encoder_common(input_size, min_channels)

encoded_size = (batch_size, int(patch_size / 32), int(patch_size / 32), min_channels * 8)
depth_decoder = Decoder(encoded_size, min_channels, 1, "depth_decoder")
normals_decoder = Decoder(encoded_size, min_channels, 3, "normals_decoder")

patch_net = PatchNet(batch_size, patch_size, min_channels)
