# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 11:02:17 2022

@author: Marc Johler
"""
from PatchNet_tf import PatchNet
from Patching import tensor_patching
from Stitching import depth_map_stitching, normals_map_stitching
import cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

#
patch_size = 128
min_channels = 8

# load images
car = cv2.imread("Pixel2Mesh\\Data\\examples\\car.png")
plane = cv2.imread("Pixel2Mesh\\Data\\examples\\plane.png")

car = tf.convert_to_tensor(car)
plane = tf.convert_to_tensor(plane)
# create a batch

# train process happens as follows
patchnet = PatchNet(patch_size, min_channels)

# generate patches and height and width intervals
patches, height_intervals, width_intervals = tensor_patching(car, patch_size)

# compute foreground map
zero_bool = np.array(car[:,:]) == [0, 0, 0]
n_zeros = zero_bool.sum(axis=2)
foreground = tf.convert_to_tensor(n_zeros != 3, dtype = "float32")
foreground = tf.reshape(foreground, car.shape[:-1] + tuple([1]))

# training process will be done seperately on each patch by now
# placeholder for a true depth map
true_depth_map  = tf.cast(patches[:,:,:,0], dtype = "float32")
# add channel dimension to depth map
true_depth_map = tf.reshape(true_depth_map, true_depth_map.shape + tuple([1]))
true_normals_map = tf.cast(patches, dtype = "float32")
foreground_patches, _, _ = tensor_patching(foreground, patch_size)
# cast to "float32"
foreground_patches = tf.cast(foreground_patches, dtype = "float32")

# do the training
for iteration in range(1):
    patchnet.training_step(patches, foreground_patches, true_depth_map, true_normals_map)
    if iteration % 10 == 0:
        print(iteration, " done")

# after training compute the depth_maps again, stitch them together and evaluate based on the whole picture
true_depth_map = tf.cast(car[:,:,0], dtype = "float32")
# add channel dimension to depth map
true_depth_map = tf.reshape(true_depth_map, true_depth_map.shape + tuple([1]))
true_normals_map = tf.cast(car, dtype = "float32")
valuation_loss = patchnet.evaluate_on_image(car, foreground, true_depth_map, true_normals_map)
