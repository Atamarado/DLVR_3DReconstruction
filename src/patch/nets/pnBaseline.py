# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 19:19:22 2022

@author: Marc Johler
"""
import tensorflow as tf
from tensorflow.keras.layers import MaxPool2D, UpSampling2D, Conv2D, Input
from patch.nets.PatchInterface import ConvLayer, PatchInterface
    
class Encoder_common(tf.Module):
    def __init__(self, input_size, min_channels, name = "Encoder_common"):
        super(Encoder_common, self).__init__(name)
        self.layers = tf.keras.Sequential([
            Input(input_size),
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
            MaxPool2D(2),
            ConvLayer(min_channels * 8),
            ConvLayer(min_channels * 8),
            ConvLayer(min_channels * 8),
            MaxPool2D(2)
            ])
        
        self.layers.build(input_size)
    
    def __call__(self, x):
        return self.layers(x)
    
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
        return self.layers(x)
        

class TfNetwork(PatchInterface):
    def __init__(self, patch_size, min_channels):
        input_size = (patch_size, patch_size, 3)
        encoded_size = (3, int(patch_size / 32), int(patch_size / 32), min_channels * 8)
        self.encoder = Encoder_common(input_size, min_channels)
        self.depth_decoder = Decoder(encoded_size, min_channels, 1, "depth_decoder")
        self.normals_decoder = Decoder(encoded_size, min_channels, 3, "normals_decoder")
               