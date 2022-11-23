# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 19:19:22 2022

@author: Marc Johler
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPool2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
from patch.Stitching import depth_map_stitching, normals_map_stitching
from patch.Losses import prediction_loss, prediction_loss_separate_losses
from patch.Patching import tensor_patching

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
        # seed = 758
        # random.seed(seed)
        # np.random.seed(seed)
        # tf.random.set_seed(seed)
        # tf.experimental.numpy.random.seed(seed)
        # print("Tensorflow seed", seed)
        super(PatchNet, self).__init__(name)
        input_size = (3, patch_size, patch_size, 3)
        encoded_size = (3, int(patch_size / 32), int(patch_size / 32), min_channels * 8)
        self.encoder = Encoder_common(input_size, min_channels)
        self.depth_decoder = Decoder(encoded_size, min_channels, 1, "depth_decoder")
        self.normals_decoder = Decoder(encoded_size, min_channels, 3, "normals_decoder")
        # initialize optimizer
        self.opt = Adam(learning_rate = 0.0001)
        # save patch size for later usage
        self.patch_size = patch_size
    
    def __call__(self, x):
        encoded = self.encoder(x)
        depth_map = self.depth_decoder(encoded)
        normals_map = self.normals_decoder(encoded)
        
        return depth_map, normals_map
    
    def training_step(self, x, foreground_map, depth_map, normals_map):
        with tf.GradientTape(persistent = False) as tape:
            pred_depth_map, pred_normals_map = self(x)
            if np.isnan(pred_depth_map).any():
                print("Problem detected")
            loss = prediction_loss(pred_depth_map, depth_map, pred_normals_map, normals_map, foreground_map)
    
        parameters = self.encoder.trainable_variables + self.depth_decoder.trainable_variables + self.normals_decoder.trainable_variables
        grads = tape.gradient(loss, parameters)
        
        loss = prediction_loss(pred_depth_map, depth_map, pred_normals_map, normals_map, foreground_map)
        
        self.opt.apply_gradients(zip(grads, parameters))
        return loss
    
    def training_step_separate_loss(self, x, foreground_map, depth_map, normals_map):
        with tf.GradientTape(persistent=False) as tape:
            pred_depth_map, pred_normals_map = self(x)
            if np.isnan(pred_depth_map).any():
                print("Problem detected")
            loss, depth_loss, normal_loss = prediction_loss_separate_losses(pred_depth_map, depth_map, pred_normals_map, normals_map, foreground_map)

        parameters = self.encoder.trainable_variables + \
            self.depth_decoder.trainable_variables + \
            self.normals_decoder.trainable_variables
        grads = tape.gradient(loss, parameters)

        self.opt.apply_gradients(zip(grads, parameters))
        return loss, depth_loss, normal_loss


    def validation_step(self, x, foreground_map, depth_map, normals_map):
        pred_depth_map, pred_normals_map = self(x)
        #loss = mean_squared_error(depth_map, pred_depth_map) + mean_squared_error(normals_map, pred_normals_map)
        return prediction_loss(pred_depth_map, depth_map, pred_normals_map, normals_map, foreground_map)

    def validation_step_separate_loss(self, x, foreground_map, depth_map, normals_map):
        pred_depth_map, pred_normals_map = self(x)
        #loss = mean_squared_error(depth_map, pred_depth_map) + mean_squared_error(normals_map, pred_normals_map)
        return prediction_loss_separate_losses(pred_depth_map, depth_map, pred_normals_map, normals_map, foreground_map)
        
    def forward_image(self, img, foreground_map, print_maps = True):
        patches, height_intervals, width_intervals = tensor_patching(img, self.patch_size)
        # forward pass
        depth_maps, normals_maps = self(patches)
        # stitch the maps together
        pred_depth_map = depth_map_stitching(img.shape, depth_maps, height_intervals, width_intervals)
        pred_normals_map = normals_map_stitching(img.shape, normals_maps, height_intervals, width_intervals)
        if print_maps:
            plt.imshow(tf.cast(pred_depth_map, dtype="float32") * foreground_map)
            #plt.imshow(normals_maps)
        return pred_depth_map, pred_normals_map
        
    # method for feeding a whole picture and 
    def validate_on_image(self, img, foreground_map, depth_map, normals_map, print_maps = True):
        pred_depth_map, pred_normals_map = self.forward_image(img, foreground_map, print_maps)
        # cast to correct float format
        pred_depth_map = tf.cast(pred_depth_map, dtype = "float32")
        pred_normals_map = tf.cast(pred_normals_map, dtype = "float32")
        # only consider foreground pixels
        depth_map_fg = depth_map * foreground_map
        pred_depth_map_fg = pred_depth_map * foreground_map
        # normalize the depth maps
        depth_map_fg = depth_map_fg - tf.reduce_mean(depth_map_fg)
        pred_depth_map_fg = pred_depth_map_fg - tf.reduce_mean(pred_depth_map_fg)
        # compute the loss 
        return prediction_loss(pred_depth_map_fg, depth_map_fg, pred_normals_map, normals_map, foreground_map)
               