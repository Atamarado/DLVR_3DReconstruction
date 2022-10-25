# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:52:07 2022

@author: Marc Johler
"""

import numpy as np
import tensorflow as tf
from PatchNet_tf import Encoder_common, Decoder, PatchNet
from Patching import patching
from Stitching import compute_overlap_matrix, compute_pixel_differences, compute_translation_loss

batch_size = 32
patch_size = 128
min_channels = 32

input_size = (batch_size, patch_size, patch_size, 3)
encoder_common = Encoder_common(input_size, min_channels)

encoded_size = (batch_size, int(patch_size / 32), int(patch_size / 32), min_channels * 8)
depth_decoder = Decoder(encoded_size, min_channels, 1, "depth_decoder")
normals_decoder = Decoder(encoded_size, min_channels, 3, "normals_decoder")

patch_net = PatchNet(batch_size, patch_size, min_channels)


import cv2
from matplotlib import pyplot as plt
from Patching import patching

car = cv2.imread("images/car.png")
plane = cv2.imread("images/plane.png")
  
plt.imshow(car)
#plt.imshow(plane)
         
# test patches with simple numbers
numfield = np.array([[(10 * j) + i for i in range(0, 10)] for j in range(0, 10)])
numfield = numfield.reshape((10, 10, 1))
patches, hi, wi = patching(numfield, 4)

# add random noise to patches, so that they can be used as dummy for the depth map
for j in range(len(patches[0])):
    patches[0][j] = patches[0][j] +  ((np.random.rand(4, 4) - 0.5) * 40).astype(int)

overlaps = compute_overlap_matrix(hi, wi)
differences = compute_pixel_differences(patches[0], hi, wi, overlaps) 

# minimize translation offset
from scipy.optimize import least_squares
x0 = np.repeat(0, 8)

LQ_results = least_squares(compute_translation_loss, x0, kwargs = {"differences": differences})

# test patches with car picture
car_patches, _, _ = patching(car, 80)

# add random noise to the car patches
for j in range(len(car_patches[1])):
    car_patches[1][j] = car_patches[1][j] +  ((np.random.rand(80, 80) - 0.5) * 40).astype(int)


fig, axes = plt.subplots(nrows=3, ncols=3)
plt.tight_layout()
for i in range(3):
    for j in range(3):
        axes[i][j].imshow(car_patches[1][i * 3 + j], cmap = "binary_r")
               
plt.show()