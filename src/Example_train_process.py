# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 17:45:12 2022

@author: Marc Johler
"""

from patch.PatchNet_tf import PatchNet
from patch.Patching import tensor_patching
from patch.Stitching import depth_map_stitching, normals_map_stitching, normalize_predictions
from DataGenerator import DataGenerator
import cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

#
patch_size = 128
min_channels = 8
batch_size = 32
train_path = "C:\\Users\\xZoCk\\DLVR_3DReconstruction\\preprocess\\data\\pnData\\train"

# train process happens as follows
patchnet = PatchNet(patch_size, min_channels)

# do the training
for epoch in range(1):
    datagen = DataGenerator(train_path, batch_size, patching = True, patch_size = patch_size)
    n_batches = datagen.__len__()
    for i in range(n_batches):
        inputs, maps = datagen.__getitem__(i)
        patches = inputs[:,:,:,0:3]
        foreground = tf.reshape(inputs[:,:,:,3], inputs.shape[:-1] + tuple([1]))
        depth_map = tf.reshape(maps[:,:,:,0], maps.shape[:-1] + tuple([1]))
        normals_map = maps[:,:,:,1:]
        loss = patchnet.training_step(patches, foreground, depth_map, normals_map)
        print("batch ", i, " done. Loss: ", loss)
        

test_path = "C:\\Users\\xZoCk\\DLVR_3DReconstruction\\preprocess\\data\\pnData\\test"

datagen_test = DataGenerator(test_path, 1, patching = False, patch_size = patch_size)

for i in range(1):
    inputs, maps = datagen_test.__getitem__(i)
    img = inputs[0][:,:,0:3]
    foreground_map = tf.reshape(inputs[0][:,:,3], img.shape[:-1] + tuple([1]))
    # reshape does weird typecast so force it back
    foreground_map = tf.cast(foreground_map, dtype = "float32")
    depth_map = tf.reshape(maps[0][:,:,0], img.shape[:-1] + tuple([1]))
    normals_map = maps[0][:,:,1:]
    
    loss = patchnet.evaluate_on_image(img, foreground_map, depth_map, normals_map)