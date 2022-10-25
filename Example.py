# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:52:07 2022

@author: Marc Johler
"""

import numpy as np
import tensorflow as tf
from PatchNet_tf import Encoder_common, Decoder, PatchNet
from Patching import patching
from Stitching import compute_translation_offsets
import cv2
from matplotlib import pyplot as plt
from Patching import patching

##### Example 1: Usage of PatchNet #####
batch_size = 32
patch_size = 128
min_channels = 32

#input_size = (batch_size, patch_size, patch_size, 3)
#encoder_common = Encoder_common(input_size, min_channels)

#encoded_size = (batch_size, int(patch_size / 32), int(patch_size / 32), min_channels * 8)
#depth_decoder = Decoder(encoded_size, min_channels, 1, "depth_decoder")
#normals_decoder = Decoder(encoded_size, min_channels, 3, "normals_decoder")

patch_net = PatchNet(batch_size, patch_size, min_channels)
  
       
##### Example 2: Visualisation of Patches #####
# test patches with car picture
car = cv2.imread("images/car.png")
plane = cv2.imread("images/plane.png")
plt.imshow(car)

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

##### Example 3: Computation of Translation offsets for stitching #####
# generate patches from a simple number field
numfield = np.array([[(10 * j) + i for i in range(0, 10)] for j in range(0, 10)])
numfield = numfield.reshape((10, 10, 1))
patches, hi, wi = patching(numfield, 4)

# add random noise to patches, so that they can be used as dummy for the depth map
for j in range(len(patches[0])):
    patches[0][j] = patches[0][j] +  ((np.random.rand(4, 4) - 0.5) * 40).astype(int)

# compute translation offsets for the 2nd to the last patch 
# (first patch has fixed offset of 0) 
translation_offsets = compute_translation_offsets(patches[0], hi, wi)