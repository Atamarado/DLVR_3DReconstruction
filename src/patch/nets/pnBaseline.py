# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 19:19:22 2022

@author: Marc Johler
"""
import tensorflow as tf
from tensorflow.keras.layers import MaxPool2D, UpSampling2D, Conv2D, Input
from patch.nets.PatchInterface import ConvLayer, PatchInterface
    
class Encoder_common:
    def __init__(self, input_size, min_channels):
        i = Input(input_size)
        l = ConvLayer(min_channels)(i)
        l = ConvLayer(min_channels)(l)
        l = MaxPool2D(2)(l)
        l = ConvLayer(min_channels * 2)(l)
        l = ConvLayer(min_channels * 2)(l)
        l = MaxPool2D(2)(l)
        l = ConvLayer(min_channels * 4)(l)
        l = ConvLayer(min_channels * 4)(l)
        l = ConvLayer(min_channels * 4)(l)
        l = MaxPool2D(2)(l)
        l = ConvLayer(min_channels * 8)(l)
        l = ConvLayer(min_channels * 8)(l)
        l = ConvLayer(min_channels * 8)(l)
        l = MaxPool2D(2)(l)
        l = ConvLayer(min_channels * 8)(l)
        l = ConvLayer(min_channels * 8)(l)
        l = ConvLayer(min_channels * 8)(l)
        l = MaxPool2D(2)(l)

        self.input = i
        self.output = l
    
    def __call__(self):
        return self.input, self.output
    
class Decoder:
    def __init__(self, min_channels, out_channels, input_layer):
        i = UpSampling2D()(input_layer)
        l = ConvLayer(min_channels * 8)(i)
        l = ConvLayer(min_channels * 8)(l)
        l = ConvLayer(min_channels * 8)(l)
        l = UpSampling2D()(l)
        l = ConvLayer(min_channels * 8)(l)
        l = ConvLayer(min_channels * 8)(l)
        l = ConvLayer(min_channels * 8)(l)
        l = UpSampling2D()(l)
        l = ConvLayer(min_channels * 4)(l)
        l = ConvLayer(min_channels * 4)(l)
        l = ConvLayer(min_channels * 4)(l)
        l = UpSampling2D()(l)
        l = ConvLayer(min_channels * 2)(l)
        l = ConvLayer(min_channels * 2)(l)
        l = UpSampling2D()(l)
        l = ConvLayer(min_channels)(l)
        l = ConvLayer(min_channels)(l)
        l = Conv2D(out_channels, 1)(l)

        self.input = i
        self.output = l
    def __call__(self):
        return self.input, self.output
        

class TfNetwork(PatchInterface):
    def __init__(self, patch_size, min_channels):
        input_size = (patch_size, patch_size, 3)
        encoder = Encoder_common(input_size, min_channels)
        input_decoders = encoder.output

        depthDecoder = Decoder(min_channels, 1, input_decoders)
        normalDecoder = Decoder(min_channels, 3, input_decoders)

        self.depthNet = tf.keras.Model(inputs=encoder.input, outputs=depthDecoder.output)
        self.normalNet = tf.keras.Model(inputs=encoder.input, outputs=normalDecoder.output)
