# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 17:45:12 2022

@author: Marc Johler
"""

from patch.PatchNet_tf import PatchNet
from DataGenerator import DataGenerator
import tensorflow as tf
import numpy as np

#
epochs = 2
patch_size = 128
min_channels = 8
batch_size = 32
train_path = "preprocess\\data\\pnData\\train"

# train process happens as follows
patchnet = PatchNet(patch_size, min_channels)
datagen = DataGenerator(train_path, batch_size, patching = True, patch_size = patch_size)

# TO-DOS compute function for the outer and inner loops to avoid repetition
# do the training
for epoch in range(epochs):
    # set datagenerator to training mode
    datagen.set_validation(False)
    datagen.set_patching(True)
    n_batches = datagen.__len__()
    training_loss = 0
    for i in range(10): # TO-DO: replace 10 with n_batches for final training loop
        inputs, maps = datagen.__getitem__(i)
        patches = inputs[:,:,:,0:3]
        foreground = tf.reshape(inputs[:,:,:,3], inputs.shape[:-1] + tuple([1]))
        depth_map = tf.reshape(maps[:,:,:,0], maps.shape[:-1] + tuple([1]))
        normals_map = maps[:,:,:,1:]
        training_loss += patchnet.training_step(patches, foreground, depth_map, normals_map)
    # compute loss per patch
    training_loss = training_loss / np.min([datagen.n_train, 10 * batch_size]) # TO-DO: replace 10 with n_batches for final training loop
    # set datagenerator to validation mode
    datagen.set_validation(True)
    n_batches = datagen.__len__()
    validation_loss_patches = 0
    for i in range(10): # TO-DO: replace 10 with n_batches for final training loop
        inputs, maps = datagen.__getitem__(i)
        patches = inputs[:,:,:,0:3]
        foreground = tf.reshape(inputs[:,:,:,3], inputs.shape[:-1] + tuple([1]))
        depth_map = tf.reshape(maps[:,:,:,0], maps.shape[:-1] + tuple([1]))
        normals_map = maps[:,:,:,1:]
        validation_loss_patches += patchnet.validation_step(patches, foreground, depth_map, normals_map)
    # compute loss per patch
    validation_loss_patches = validation_loss_patches / np.min([datagen.n_val, 10 * batch_size]) # TO-DO: replace 10 with n_batches for final training    
    # compute loss for the whole images
    datagen.set_patching(False)
    """
    validation_loss_images = 0
    for i in range(10): # TO-DO: replace 10 with n_batches for final training loop
        inputs, maps = datagen.__getitem__(i)
        img = inputs[0][:,:,0:3]
        foreground_map = tf.reshape(inputs[0][:,:,3], img.shape[:-1] + tuple([1]))
        # reshape does weird typecast so force it back
        foreground_map = tf.cast(foreground_map, dtype = "float32")
        depth_map = tf.reshape(maps[0][:,:,0], img.shape[:-1] + tuple([1]))
        normals_map = maps[0][:,:,1:]
        validation_loss_images += patchnet.validate_on_image(img, foreground, depth_map, normals_map, print_maps = False)
    # compute loss per image
    validation_loss_images = validation_loss_images / 10 # TO-DO: replace 10 with n_batches for final training
    
    """
    print("Epoch ", epoch, " done with losses: ")
    print("Training: ", training_loss)
    print("Validation on patches ", validation_loss_patches)
    ###print("Validation on images ", validation_loss_images)
    
    
test_path = "preprocess\\data\\pnData\\test"

datagen_test = DataGenerator(test_path, 1, patching = False, patch_size = patch_size)

for i in range(1):
    inputs, maps = datagen_test.__getitem__(i)
    img = inputs[0][:,:,0:3]
    foreground_map = tf.reshape(inputs[0][:,:,3], img.shape[:-1] + tuple([1]))
    # reshape does weird typecast so force it back
    foreground_map = tf.cast(foreground_map, dtype = "float32")
    depth_map = tf.reshape(maps[0][:,:,0], img.shape[:-1] + tuple([1]))
    normals_map = maps[0][:,:,1:]
    
    loss = patchnet.validate_on_image(img, foreground_map, depth_map, normals_map)