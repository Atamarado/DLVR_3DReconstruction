# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 19:19:22 2022

@author: Krisztián Bokor, Ginés Carreto Picón, Marc Johler
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, Input
from patch.nets.PatchInterface import ConvLayer, ConvTransposeLayer, PatchInterface
from patch.nets.InceptionModules import *

class Decoder():
    def __init__(self, min_channels, out_channels, input_layer):
        l = UpSampling2D()(input_layer)
        l = ConvLayer(min_channels * 8)(l)
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

        self.out = l

class TfNetwork(PatchInterface, tf.Module):
    """
    Modification of the baseline model that changes changes the encoder, adding inception modules into it
    (see InceptionModules.py for more info)
    """
    def __init__(self, patch_size, min_channels):
        input_size = (patch_size, patch_size, 3)

        # Encoder
        i = Input(input_size)
        stem = inceptionv4_stem(i)
        p1 = MaxPool2D(2, data_format='channels_last')(stem)
        a1 = inceptionv4_A(p1)
        r1 = inceptionv4_reductionA(a1)
        b1 = inceptionv4_B(r1)
        conv_1 = Conv2D(filters=256, kernel_size=(1, 1), padding='same')(b1)
        c1 = inceptionv4_C(conv_1)
        conv_2 = Conv2D(filters=64, kernel_size=(3, 3), padding='valid')(c1)

        depth_layers = Decoder(min_channels, 1, conv_2)
        depth_layers = depth_layers.out

        normal_layers = Decoder(min_channels, 3, conv_2)
        normal_layers = normal_layers.out

        self.network = tf.keras.Model(i, outputs=(depth_layers, normal_layers))
        self.trainableVariables = self.network.trainable_weights

    def __call__(self, x):
        return self.network.call(x)

    def save_weights(self, filename):
        self.network.save_weights(filename)

    def load_weights(self, filename):
        self.network.load_weights(filename)