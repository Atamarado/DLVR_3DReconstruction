# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 11:02:17 2022

@author: Marc Johler
"""
from PatchNet_tf import PatchNet
from Patching import tensor_patching
from Stitching import depth_map_stitching, normals_map_stitching
from Losses import mean_squared_error
import cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

#
patch_size = 128
min_channels = 8

# load images
car = cv2.imread("images/car.png")
plane = cv2.imread("images/plane.png")

car = tf.convert_to_tensor(car)
plane = tf.convert_to_tensor(plane)
# create a batch

# train process happens as follows
patchnet = PatchNet(patch_size, min_channels)

# generate patches and height and width intervals
patches, height_intervals, width_intervals = tensor_patching(car, patch_size)

# Our training process will looks like follows 

## LOOP over the input images 

# training process will be done seperately on each patch by now

# placeholder for a true depth map
true_depth_map  = patches[:,:,:,0]
# do the training
for iteration in range(100):
    for i, patch in enumerate(patches):
        patch = tf.reshape(patch, (1, patch_size, patch_size, 3))
        true_depth_map_i = tf.reshape(tf.cast(true_depth_map[i], dtype = tf.float32), (1, patch_size, patch_size))
        patchnet.step(patch, true_depth_map_i, "XXXXX_normals_map_XXXXX")
    if iteration % 10 == 0:
        print(iteration, " done")

# after training compute the depth_maps again, stitch them together and evaluate based on the whole picture
true_depth_map = tf.cast(car[:,:,0], dtype = tf.double)
true_normals_map = car
valuation_loss = patchnet.evaluate_on_image(car, true_depth_map, car)
