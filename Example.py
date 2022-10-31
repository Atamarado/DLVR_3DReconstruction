# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:52:07 2022

@author: Marc Johler
"""
import numpy as np
import tensorflow as tf
from PatchNet_tf import Encoder_common, Decoder, PatchNet, VANet_adapted
from Patching import patching
from Stitching import compute_translation_offsets, depth_map_stitching
import cv2
from matplotlib import pyplot as plt
from Patching import patching
import math

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

car = tf.convert_to_tensor(car)
plt.imshow(car)
car_patches, _, _ = patching(car, 80)

# add random noise to the car patches
#for j in range(len(car_patches[1])):
    #car_patches[1][j] = car_patches[1][j] +  ((np.random.rand(80, 80) - 0.5) * 40).astype(int)


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
numfield = tf.convert_to_tensor(numfield)
patches, hi, wi = patching(numfield, 4)

# add random noise to patches, so that they can be used as dummy for the depth map
patches_random_noise = patches[0].copy()
for j in range(len(patches_random_noise)):
    patches_random_noise[j] = patches_random_noise[j] +  tf.convert_to_tensor(((np.random.rand(4, 4) - 0.5) * 20).astype(int))

# compute translation offsets for the 2nd to the last patch 
# (first patch has fixed offset of 0) 
translation_offsets_rn = compute_translation_offsets(patches_random_noise, hi, wi)
# --> way too complex to create test set from random noise

# add fixed offset to each patch, to check if function can reconstruct it
patches_biased = patches[0].copy()
for j in range(len(patches_biased)):
    patches_biased[j] = patches_biased[j] + j
    
# compute translation offsets for the 2nd to the last patch 
# (first patch has fixed offset of 0) 
translation_offsets_b = compute_translation_offsets(patches_biased, hi, wi)

# check closeness to expected result for each offset 
closeness_bool = np.repeat(False, len(patches_biased) - 1)
for j in range(len(patches_biased) -1):
    closeness_bool[j] = math.isclose(translation_offsets_b[j], -j - 1, abs_tol = 10**-6)

assert closeness_bool.all() 

##### Example 4: Stitch depth maps of patches back together #####
car = cv2.imread("images/car.png")
plane = cv2.imread("images/plane.png")
# choose one channel, so that there is only one output channel
car_channel0 = car[:,:,0]
car_channel0_shape = car_channel0.shape

car_channel0 = car_channel0.reshape((car_channel0_shape[0], 
                                     car_channel0_shape[1], 1))

car_patches, hi, wi = patching(car_channel0, 40)

# add random noise to the car patches
for j in range(len(car_patches[0])):
    car_patches[0][j] = (car_patches[0][j] / (j + 1)).astype(int)

fig, axes = plt.subplots(nrows=6, ncols=6)
plt.tight_layout()
for i in range(6):
    for j in range(6):
        axes[i][j].imshow(car_patches[0][i * 6 + j], cmap = "binary_r")
               
plt.show()

# stitch the pieces back together
# without translation offsets
final_depth_map = depth_map_stitching(car_channel0_shape, car_patches, hi, wi, False)
# with translation offsets
final_depth_map_translated = depth_map_stitching(car_channel0_shape, car_patches, hi, wi)
# compare the depth maps
plt.imshow(final_depth_map, cmap = "binary_r")
plt.show()
plt.imshow(final_depth_map_translated, cmap = "binary_r")


##### Example 5: Stitch encoder feature maps back together #####
car = cv2.imread("images/car.png")

car = tf.convert_to_tensor(car)

# patch_size and decoder_shape must be compatible 
# (otherwise there will be an assertion error to tell the user what is wrong)
patch_size = 128
min_channels = 16
decoder_shape = (12, 12)

vanet = VANet_adapted(1, patch_size, min_channels, decoder_shape)

stitched_map = vanet(car)

assert stitched_map.shape == (decoder_shape[0], decoder_shape[1], min_channels * 8)

##### Example 6: Test backpropagation
encoder_gradients = vanet.step(car)