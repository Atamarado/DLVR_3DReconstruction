# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 19:19:22 2022

@author: Marc Johler
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
from patch.Stitching import depth_map_stitching, normals_map_stitching
from patch.Losses import prediction_loss, prediction_loss_separate_losses
from patch.Patching import tensor_patching

class PatchNet(tf.Module):
    def __init__(self, patch_size, min_channels, fixed_overlaps, network, name = "patchnet"):
        # seed = 758
        # random.seed(seed)
        # np.random.seed(seed)
        # tf.random.set_seed(seed)
        # tf.experimental.numpy.random.seed(seed)
        # print("Tensorflow seed", seed)
        super(PatchNet, self).__init__(name)
        input_size = (3, patch_size, patch_size, 3)
        encoded_size = (3, int(patch_size / 32), int(patch_size / 32), min_channels * 8)
        self.network = network
        # initialize optimizer
        self.opt = Adam(learning_rate = 0.001)
        # save patch size for later usage
        self.patch_size = patch_size
        self.fixed_overlaps = fixed_overlaps
    
    def __call__(self, x):
        return self.network(x)
    
    def training_step(self, x, foreground_map, depth_map, normals_map):
        with tf.GradientTape(persistent = False) as tape:
            pred_depth_map, pred_normals_map = self(x)
            if np.isnan(pred_depth_map).any():
                print("Problem detected")
            loss = prediction_loss(pred_depth_map, depth_map, pred_normals_map, normals_map, foreground_map)
    
        parameters = self.network.trainableVariables
        grads = tape.gradient(loss, parameters)
        
        self.opt.apply_gradients(zip(grads, parameters))
        return loss
    
    def training_step_separate_loss(self, x, foreground_map, depth_map, normals_map):
        with tf.GradientTape(persistent=False) as tape:
            pred_depth_map, pred_normals_map = self(x)
            if np.isnan(pred_depth_map).any():
                print("Problem detected")
            loss, depth_loss, normal_loss = prediction_loss_separate_losses(pred_depth_map, depth_map, pred_normals_map, normals_map, foreground_map)

        parameters = self.network.trainableVariables
        grads = tape.gradient(loss, parameters)

        self.opt.apply_gradients(zip(grads, parameters))
        return loss, depth_loss, normal_loss


    def validation_step(self, x, foreground_map, depth_map, normals_map):
        pred_depth_map, pred_normals_map = self(x)
        return prediction_loss(pred_depth_map, depth_map, pred_normals_map, normals_map, foreground_map)

    def validation_step_separate_loss(self, x, foreground_map, depth_map, normals_map):
        pred_depth_map, pred_normals_map = self(x)
        #loss = mean_squared_error(depth_map, pred_depth_map) + mean_squared_error(normals_map, pred_normals_map)
        return prediction_loss_separate_losses(pred_depth_map, depth_map, pred_normals_map, normals_map, foreground_map)
        
    # TO-DO: delete overlap after investigation
    def forward_image(self, img, foreground_map, print_maps = True, true_depth_map = None, true_normals_map = None):
        patches, height_intervals, width_intervals = tensor_patching(img, self.patch_size, self.fixed_overlaps)
        # forward pass
        depth_maps, normals_maps = self(patches)
        # stitch the maps together
        pred_depth_map = depth_map_stitching(img.shape, depth_maps, height_intervals, width_intervals, sigma = 10)
        pred_normals_map = normals_map_stitching(img.shape, normals_maps, height_intervals, width_intervals)
        # DELETE ALL OF THIS AFTER DEBUGGING
        """
        # without filtering
        pred_depth_map_uf = depth_map_stitching(img.shape, depth_maps, height_intervals, width_intervals, apply_smoothing = False)
        pred_normals_map_uf = normals_map_stitching(img.shape, normals_maps, height_intervals, width_intervals, apply_smoothing = False)
        # cast to correct float format
        pred_depth_map_uf = tf.cast(pred_depth_map_uf, dtype = "float32")
        pred_normals_map_uf = tf.cast(pred_normals_map_uf, dtype = "float32")
        # only consider foreground pixels
        depth_map_fg = true_depth_map * foreground_map
        pred_depth_map_fg = pred_depth_map * foreground_map
        pred_depth_map_uf_fg = pred_depth_map_uf * foreground_map
        # normalize the depth maps
        depth_map_fg = depth_map_fg - tf.reduce_mean(depth_map_fg)
        pred_depth_map_fg = pred_depth_map_fg - tf.reduce_mean(pred_depth_map_fg)
        pred_depth_map_uf_fg = pred_depth_map_uf_fg - tf.reduce_mean(pred_depth_map_uf_fg)
        
        filtered_loss = prediction_loss(pred_depth_map_fg, depth_map_fg, pred_normals_map, true_normals_map, foreground_map)
        unfiltered_loss = prediction_loss(pred_depth_map_uf_fg, depth_map_fg, pred_normals_map_uf, true_normals_map, foreground_map)
        
        print("Filter loss improvement:", unfiltered_loss - filtered_loss)
        """
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
        pred_depth_map_fg = pred_depth_map_fg - tf.reduce_mean(pred_depth_map_fg) + tf.reduce_mean(pred_depth_map_fg)
        # compute the loss 
        return prediction_loss(pred_depth_map_fg, depth_map_fg, pred_normals_map, normals_map, foreground_map) 
               