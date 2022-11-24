# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 19:19:22 2022

@author: Marc Johler
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPool2D, UpSampling2D, Flatten, Dense, Input, Conv2DTranspose, Concatenate
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
from patch.Stitching import depth_map_stitching, normals_map_stitching
from patch.Losses import prediction_loss, prediction_loss_separate_losses
from patch.Patching import tensor_patching
from patch.nets.PatchInterface import ConvLayer, ConvTransposeLayer, PatchInterface

class Decoder():
    def __init__(self, min_channels, out_channels, input_layer, conv_connections):
        (c1, c2, c3, c4, c5) = conv_connections
        up5 = Concatenate()([c5, input_layer])
        ct5 = ConvTranposeLayer(min_channels * 8)(up5)
        ct5 = ConvTranposeLayer(min_channels * 8)(ct5)
        ct5 = ConvTranposeLayer(min_channels * 8)(ct5)
        up4 = UpSampling2D()(ct5)
        u4 = Concatenate()([c4, up4])
        ct4 = ConvTranposeLayer(min_channels * 8)(u4)
        ct4 = ConvTranposeLayer(min_channels * 8)(ct4)
        ct4 = ConvTranposeLayer(min_channels * 8)(ct4)
        up3 = UpSampling2D()(ct4)
        u3 = Concatenate()([c3, up3])
        ct3 = ConvTranposeLayer(min_channels * 4)(u3)
        ct3 = ConvTranposeLayer(min_channels * 4)(ct3)
        ct3 = ConvTranposeLayer(min_channels * 4)(ct3)
        up2 = UpSampling2D()(ct3)
        u2 = Concatenate()([c2, up2])
        ct2 = ConvTranposeLayer(min_channels * 2)(u2)
        ct2 = ConvTranposeLayer(min_channels * 2)(ct2)
        up1 = UpSampling2D()(ct2)
        u1 = Concatenate()([c1, up1])
        ct1 = ConvTranposeLayer(min_channels)(u1)
        ct1 = ConvTranposeLayer(min_channels)(ct1)
        out = Conv2D(out_channels, 1)(ct1)
        self.out = out
    
    def __call__(self, x):
        return self.out

class TfNetwork(PatchInterface):
    def __init__(self, patch_size, min_channels):
        input_size = (patch_size, patch_size, 3)

        # Encoder
        i = Input(input_size)
        c1 = ConvLayer(min_channels)(i)
        c1 = ConvLayer(min_channels)(c1)
        p1 = MaxPool2D(2, data_format='channels_last')(c1)
        c2 = ConvLayer(min_channels*2)(p1)
        c2 = ConvLayer(min_channels*2)(c2)
        p2 = MaxPool2D(2, data_format='channels_last')(c2)
        c3 = ConvLayer(min_channels * 4)(p2)
        c3 = ConvLayer(min_channels * 4)(c3)
        c3 = ConvLayer(min_channels * 4)(c3)
        p3 = MaxPool2D(2, data_format='channels_last')(c3)
        c4 = ConvLayer(min_channels * 8)(p3)
        c4 = ConvLayer(min_channels * 8)(c4)
        c4 = ConvLayer(min_channels * 8)(c4)
        p4 = MaxPool2D(2, data_format='channels_last')(c4)
        c5 = ConvLayer(min_channels * 8)(p4)
        c5 = ConvLayer(min_channels * 8)(c5)
        c5 = ConvLayer(min_channels * 8)(c5)
        p5 = MaxPool2D(2, data_format='channels_last')(c5)

        f1 = Flatten()(p5)
        d1 = Dense(6400)(f1)

        input_layer_decoder = tf.reshape(d1, [-1]+(c5.get_shape()[1:].as_list()))

        self.encoder = tf.keras.Model()

        conv_connections = [c1, c2, c3, c4, c5]

        depth_layers = Decoder(min_channels, 1, input_layer_decoder, conv_connections, "depth_decoder")
        depth_layers = depth_layers.out
        self.depth_decoder = tf.keras.Model(i, depth_layers)

        normal_layers = Decoder(min_channels, 3, input_layer_decoder, conv_connections, "normals_decoder")
        normal_layers = normal_layers.out
        self.normals_decoder = tf.keras.Model(i, normal_layers)