# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 19:19:22 2022

@author: Marc Johler
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPool2D, UpSampling2D
from Patching import patching
from Stitching import feature_map_stitching

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
    
# Implement decoder
class VANet_adapted(tf.Module):
    def __init__(self, batch_size, patch_size, min_channels, decoder_dim, name = "vanet"):
        super(VANet_adapted, self).__init__(name)
        input_size = (batch_size, patch_size, patch_size, 3)
        encoded_size = (batch_size, int(patch_size / 32), int(patch_size / 32), min_channels * 8)
        self.patch_size = patch_size
        self.encoder = Encoder_common(input_size, min_channels)
        self.decoder_dim = decoder_dim
        self.decoder = None # add the decoder part here 
    
    def __call__(self, x):
        patches, n_height, n_width = patching(x, self.patch_size, return_intervals = False)
        input_shape = x.shape
        # watch out that channel is last dimension
        encoded_patches = np.repeat(None, n_height * n_width)
        with tf.GradientTape(persistent = True) as self.tape:
            # encode all the patches
            for i in range(n_height * n_width):
                patch_i = []
                for j in range(input_shape[-1]):
                    patch_i.append(patches[j][i])
                patch_i = tf.stack(patch_i, 2)
                # add batch dimension
                patch_i = tf.reshape(patch_i, (1, self.patch_size, self.patch_size, input_shape[-1]))
                encoded_patches[i] = self.encoder(patch_i)[0]
            # stitch them back together
            stitched_map = feature_map_stitching(encoded_patches, n_height, n_width, self.decoder_dim)
            # decoder 
            # self.decoder(stitched_map)
            return stitched_map
    
    def backprop(self, loss):
        # first compute the gradient for the decoder
        dloss_dD = self.tape.gradient(loss, self.decoder.layers.trainable_variables)
        dD_dE = self.tape.gradient(dloss_dD, self.encoder.layer.trainable_variables)