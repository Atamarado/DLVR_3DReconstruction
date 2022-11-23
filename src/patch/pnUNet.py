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
from patch.Losses import prediction_loss
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

class ConvTranposeLayer():
    def __init__(self, out_channels, name="ConvLayer"):
        super(Conv2DTranspose, self).__init__(name)
        self.conv = Conv2DTranspose(out_channels, 3, padding="same")
        self.batchnorm = BatchNormalization()
        self.relu = ReLU()

    def __call__(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        return self.relu(x)
    
class Decoder():
    def __init__(self, min_channels, out_channels, input_layer, conv_connections, name = "decoder"):
        (c1, c2, c3, c4, c5) = conv_connections
        up5 = Concatenate(axis=3)([c5, input_layer])
        ct5 = ConvTranposeLayer(min_channels * 8)(up5)
        ct5 = ConvTranposeLayer(min_channels * 8)(ct5)
        ct5 = ConvTranposeLayer(min_channels * 8)(ct5)
        up4 = UpSampling2D()(ct5)
        u4 = Concatenate([c4, up4], axis=3)
        ct4 = ConvTranposeLayer(min_channels * 8)(u4)
        ct4 = ConvTranposeLayer(min_channels * 8)(ct4)
        ct4 = ConvTranposeLayer(min_channels * 8)(ct4)
        up3 = UpSampling2D()(ct4)
        u3 = Concatenate([c3, up3], axis=3)
        ct3 = ConvTranposeLayer(min_channels * 4)(u3)
        ct3 = ConvTranposeLayer(min_channels * 4)(ct3)
        ct3 = ConvTranposeLayer(min_channels * 4)(ct3)
        up2 = UpSampling2D()(ct3)
        u2 = Concatenate([c2, up2], axis=3)
        ct2 = ConvTranposeLayer(min_channels * 2)(u2)
        ct2 = ConvTranposeLayer(min_channels * 2)(ct2)
        up1 = UpSampling2D()(ct2)
        u1 = Concatenate([c1, up1], axis=3)
        ct1 = ConvTranposeLayer(min_channels)(u1)
        ct1 = ConvTranposeLayer(min_channels)(ct1)
        out = Conv2D(out_channels, 1)(ct1)
        return out
    
    def __call__(self, x):
        return self.layers(x)
        

class PatchNet(tf.Module):
    def __init__(self, patch_size, min_channels, fixed_overlaps, name = "patchnet"):
        # seed = 758
        # random.seed(seed)
        # np.random.seed(seed)
        # tf.random.set_seed(seed)
        # tf.experimental.numpy.random.seed(seed)
        # print("Tensorflow seed", seed)
        super(PatchNet, self).__init__(name)
        input_size = (patch_size, patch_size, 3) # TODO: Check out input_size dimensionality
        encoded_size = (3, int(patch_size / 32), int(patch_size / 32), min_channels * 8)

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

        input_decoder = tf.reshape(d1, c5.get_shape()[1:])

        conv_connections = [c1, c2, c3, c4, c5]

        depth_layers = Decoder(min_channels, 1, input_decoder, conv_connections, "depth_decoder")
        self.depth_decoder = tf.keras.Model(i, depth_layers)
        normal_layers = Decoder(min_channels, 3, input_decoder, conv_connections, "normals_decoder")
        self.normals_decoder = tf.keras.Models(i, normal_layers)
        # initialize optimizer
        self.opt = Adam(learning_rate = 0.001)
        # save patch size for later usage
        self.patch_size = patch_size
        self.fixed_overlaps = fixed_overlaps
    
    def __call__(self, x):
        depth_map = self.depth_decoder(x)
        normals_map = self.normals_decoder(x)
        
        return depth_map, normals_map
    
    def training_step(self, x, foreground_map, depth_map, normals_map):
        with tf.GradientTape(persistent = False) as tape:
            pred_depth_map, pred_normals_map = self(x)
            if np.isnan(pred_depth_map).any():
                print("Problem detected")
            loss = prediction_loss(pred_depth_map, depth_map, pred_normals_map, normals_map, foreground_map)
    
        parameters = self.encoder.trainable_variables + self.depth_decoder.trainable_variables + self.normals_decoder.trainable_variables
        grads = tape.gradient(loss, parameters)
        
        self.opt.apply_gradients(zip(grads, parameters))
        return loss
    
    def validation_step(self, x, foreground_map, depth_map, normals_map):
        pred_depth_map, pred_normals_map = self(x)
        return prediction_loss(pred_depth_map, depth_map, pred_normals_map, normals_map, foreground_map)
        
    # TO-DO: delete overlap after investigation
    def forward_image(self, img, foreground_map, print_maps = True, true_depth_map = None, true_normals_map = None):
        patches, height_intervals, width_intervals = tensor_patching(img, self.patch_size, self.fixed_overlaps)
        # forward pass
        depth_maps, normals_maps = self(patches)
        # stitch the maps together
        pred_depth_map = depth_map_stitching(img.shape, depth_maps, height_intervals, width_intervals, sigma = 10)
        pred_normals_map = normals_map_stitching(img.shape, normals_maps, height_intervals, width_intervals)
        if print_maps:
            plt.imshow(tf.math.abs(tf.cast(pred_depth_map, dtype="float32") - true_depth_map) * foreground_map)
            plt.imshow(tf.math.abs(tf.cast(pred_normals_map, dtype="float32") - true_normals_map) * foreground_map)
            #plt.imshow(normals_maps)
        return pred_depth_map, pred_normals_map
        
    # method for feeding a whole picture and 
    def validate_on_image(self, img, foreground_map, depth_map, normals_map, print_maps = False):
        pred_depth_map, pred_normals_map = self.forward_image(img, foreground_map, print_maps, depth_map, normals_map)
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
               