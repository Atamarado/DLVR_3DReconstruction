# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 19:19:22 2022

@author: Marc Johler
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPool2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from Patching import patching, tensor_patching
from Stitching import feature_map_stitching, depth_map_stitching, normals_map_stitching
from Losses import prediction_loss, mean_squared_error
from matplotlib import pyplot as plt

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
            MaxPool2D(2),
            ConvLayer(min_channels * 8),
            ConvLayer(min_channels * 8),
            ConvLayer(min_channels * 8),
            MaxPool2D(2)
            ])
        
        self.layers.build(input_size)
    
    def __call__(self, x):
        return self.layers(x)
    
class Encoder_adapted(tf.Module):
    def __init__(self, input_size, min_channels, name = "Encoder_common"):
        super(Encoder_adapted, self).__init__(name)
        self.layers = tf.keras.Sequential([
            ConvLayer(min_channels),
            ConvLayer(min_channels),
            ConvLayer(min_channels),
            MaxPool2D(2),
            ConvLayer(min_channels * 2),
            ConvLayer(min_channels * 2),
            ConvLayer(min_channels * 2),
            MaxPool2D(2),
            ConvLayer(min_channels * 4),
            ConvLayer(min_channels * 4),
            ConvLayer(min_channels * 4)
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
    def __init__(self, patch_size, min_channels, name = "patchnet"):
        super(PatchNet, self).__init__(name)
        input_size = (3, patch_size, patch_size, 3)
        encoded_size = (3, int(patch_size / 32), int(patch_size / 32), min_channels * 8)
        self.encoder = Encoder_common(input_size, min_channels)
        self.depth_decoder = Decoder(encoded_size, min_channels, 1, "depth_decoder")
        self.normals_decoder = Decoder(encoded_size, min_channels, 3, "normals_decoder")
        # initialize optimizer
        self.opt = Adam()
        # save patch size for later usage
        self.patch_size = patch_size
    
    def __call__(self, x):
        encoded = self.encoder(x)
        depth_map = self.depth_decoder(encoded)
        normals_map = self.normals_decoder(encoded)
        
        return depth_map, normals_map
    
    def training_step(self, x, foreground_map, depth_map, normals_map):
        with tf.GradientTape(persistent = True) as tape:
            pred_depth_map, pred_normals_map = self(x)
            #loss = mean_squared_error(depth_map, pred_depth_map) + mean_squared_error(normals_map, pred_normals_map)
            loss = prediction_loss(pred_depth_map, depth_map, pred_normals_map, normals_map, foreground_map)
        parameters = self.encoder.trainable_variables + self.depth_decoder.trainable_variables + self.normals_decoder.trainable_variables
        grads = tape.gradient(loss, parameters)
        
        grad_max = 0
        grad_min = 0
        
        tmp = prediction_loss(pred_depth_map, depth_map, pred_normals_map, normals_map, foreground_map)
        
        self.opt.apply_gradients(zip(grads, parameters))
        return loss
        
    def forward_image(self, img, print_maps = True):
        patches, height_intervals, width_intervals = tensor_patching(img, self.patch_size)
        n_patches = len(patches)
        # forward pass
        depth_maps, normals_maps = self(patches)
        # stitch the maps together
        pred_depth_map = depth_map_stitching(img.shape, depth_maps, height_intervals, width_intervals)
        pred_normals_map = normals_map_stitching(img.shape, normals_maps, height_intervals, width_intervals)
        if print_maps:
            plt.imshow(pred_depth_map)
            #plt.imshow(normals_maps)
        return pred_depth_map, pred_normals_map
        
    # method for feeding a whole picture and 
    def evaluate_on_image(self, img, foreground_map, depth_map, normals_map, print_maps = True):
        pred_depth_map, pred_normals_map = self.forward_image(img, print_maps)
        # cast to correct float format
        pred_depth_map = tf.cast(pred_depth_map, dtype = "float32")
        pred_normals_map = tf.cast(pred_normals_map, dtype = "float32")
        # compute the loss 
        return prediction_loss(pred_depth_map, depth_map, pred_normals_map, normals_map, foreground_map)
        
        
# Implement decoder
class DLVR_net(tf.Module):
    def __init__(self, batch_size, patch_size, min_channels, decoder_dim, name = "vanet"):
        super(DLVR_net, self).__init__(name)
        input_size = (batch_size, patch_size, patch_size, 3)
        encoded_size = (batch_size, int(patch_size / 32), int(patch_size / 32), min_channels * 8)
        self.patch_size = patch_size
        self.encoder = Encoder_adapted(input_size, min_channels)
        self.decoder_dim = decoder_dim
        self.decoder = None # add the decoder part here 
        # delete this after decoder has been implemented
        self.min_channels = min_channels
        # optimizer
        self.opt = Adam()
    
    def __call__(self, x):
        patches, n_height, n_width = patching(x, self.patch_size, return_intervals = False)
        input_shape = x.shape
        # watch out that channel is last dimension
        # batch functionality is not provided here
        encoded_patches = tf.zeros(tuple([0]) + self.encoder.layers.output_shape[1:])
        # encode all the patches
        for i in range(n_height * n_width):
            patch_i = []
            for j in range(input_shape[-1]):
                patch_i.append(patches[j][i])
            patch_i = tf.stack(patch_i, 2)
            # add batch dimension
            patch_i = tf.reshape(patch_i, (1, self.patch_size, self.patch_size, input_shape[-1]))
            # WATCH OUT: this dimensions are only correct if batch_size = 1 
            encoded_patches = tf.concat([encoded_patches, self.encoder(patch_i)], axis = 0)
        # stitch them back together
        stitched_map = feature_map_stitching(encoded_patches, n_height, n_width, self.decoder_dim)
        # decoder 
        # self.decoder(stitched_map)
        return stitched_map
    
    # implement this after decoder has been implemented
    def step(self, x):
        with tf.GradientTape(persistent = True) as tape:
            stitched_map = self(x)
            random_gradients = tf.convert_to_tensor(np.random.rand(self.decoder_dim[0], self.decoder_dim[1], self.min_channels * 4))
            loss = mean_squared_error(random_gradients, stitched_map)
            avg_loss = tf.reduce_mean(loss)
        # first compute the gradient for the decoder
        #dloss_dD = self.tape.gradient(loss, self.decoder.layers.trainable_variables)
        # just compute the loss kinda randomly as long as there is no decoder
        grads = tape.gradient(avg_loss, self.encoder.layers.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.encoder.layers.trainable_variables))