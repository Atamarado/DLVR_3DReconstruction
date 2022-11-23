# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 13:46:53 2022

@author: xZoCk
"""

import math
import numpy as np
import tensorflow as tf
from tqdm import tqdm

def patch_loop(model, data_generator, validation = False, n_batches = math.inf):
    # set the options for the data generator
    data_generator.set_validation(validation)
    data_generator.set_patching(True)
    n_batches = np.min([data_generator.__len__(), n_batches])
    total_patches = 0
    loss = 0
    # description for progress bar
    if validation:
        desc = "Validation progress (patches)"
    else:
        desc = "Training progress"
    # loop over the data
    for i in tqdm(range(n_batches), desc = desc): # TO-DO: replace 10 with n_batches for final training loop
        inputs, maps = data_generator.__getitem__(i)
        patches = inputs[:,:,:,0:3]
        foreground_map = tf.reshape(inputs[:,:,:,3], inputs.shape[:-1] + tuple([1]))
        depth_map = tf.reshape(maps[:,:,:,0], maps.shape[:-1] + tuple([1]))
        normals_map = maps[:,:,:,1:]
        # do the respective step
        if validation:
            loss += model.validation_step(patches, foreground_map, depth_map, normals_map)
        else:
            loss += model.training_step(patches, foreground_map, depth_map, normals_map)
        # remember number of patches
        total_patches += len(patches)
        
    return loss / total_patches


def patch_loop_separate_loss(model, data_generator, validation=False, n_batches=math.inf):
    # set the options for the data generator
    data_generator.set_validation(validation)
    data_generator.set_patching(True)
    n_batches = np.min([data_generator.__len__(), n_batches])
    total_patches = 0
    total_loss = 0
    depth_loss = 0
    normal_loss = 0
    # description for progress bar
    if validation:
        desc = "Validation progress (patches)"
    else:
        desc = "Training progress"
    # loop over the data
    # TO-DO: replace 10 with n_batches for final training loop
    for i in tqdm(range(n_batches), desc=desc):
        inputs, maps = data_generator.__getitem__(i)
        patches = inputs[:, :, :, 0:3]
        foreground_map = tf.reshape(
            inputs[:, :, :, 3], inputs.shape[:-1] + tuple([1]))
        depth_map = tf.reshape(maps[:, :, :, 0], maps.shape[:-1] + tuple([1]))
        normals_map = maps[:, :, :, 1:]
        # do the respective step
        if validation:
            t_loss, d_loss, n_loss = model.training_step_separate_loss(patches,
                                        foreground_map, depth_map, normals_map)
            total_loss += t_loss
            depth_loss += d_loss
            normal_loss += n_loss
        else:
            t_loss, d_loss, n_loss = model.validation_step_separate_loss(patches,
                                          foreground_map, depth_map, normals_map)
            total_loss += t_loss
            depth_loss += d_loss
            normal_loss += n_loss
        # remember number of patches
        total_patches += len(patches)

    return total_loss / total_patches, depth_loss / total_patches, normal_loss / total_patches

def image_loop(model, data_generator, n_batches):
    # set the options for the data generator
    data_generator.set_validation(True)
    data_generator.set_patching(False)
    n_batches = np.min([data_generator.__len__(), n_batches])
    batch_size = data_generator.batch_size
    # To-Do change this after investigation
    loss = 0
    # loop over all images
    for i in tqdm(range(n_batches), desc = "Validation progress (images)"):
        inputs, maps = data_generator.__getitem__(i)
        n_images = len(inputs)
        for j in range(n_images):
            img = inputs[j][:,:,0:3]
            foreground_map = tf.reshape(inputs[j][:,:,3], img.shape[:-1] + tuple([1]))
            # reshape does weird typecast so force it back
            foreground_map = tf.cast(foreground_map, dtype = "float32")
            depth_map = tf.reshape(maps[j][:,:,0], img.shape[:-1] + tuple([1]))
            normals_map = maps[j][:,:,1:]
            loss += model.validate_on_image(img, foreground_map, depth_map, normals_map, print_maps = False)
    
    return loss / np.min([data_generator.n_val, n_batches * batch_size])
        

def train(model, data_generator, epochs, n_batches = None, n_train_batches = None, n_val_batches = None):
    if n_val_batches is None and n_train_batches is None:
        n_val_batches = n_batches
        n_train_batches = n_batches
        
    assert n_val_batches is not None
    assert n_train_batches is not None
    
    for epoch in range(epochs):
        train_loss = patch_loop(model, data_generator, validation = False, n_batches = n_train_batches)
        val_loss_patch = patch_loop(model, data_generator, validation = True, n_batches = n_val_batches)
        val_loss_img = image_loop(model, data_generator, n_batches = n_val_batches)
        
        print("Epoch", epoch, "done with losses:")
        print("Training:", train_loss)
        print("Validation on patches", val_loss_patch)
        print("Validation on images:", val_loss_img)
        
    print("Training finished after", epochs, "Epochs")
    

def test(model, data_generator, n_batches):
    test_loss = image_loop(model, data_generator, n_batches = n_batches)
    batch_size = data_generator.batch_size 
    print("Tested on", n_batches * batch_size, "images")
    print("Loss:", test_loss)
    
    return test_loss

    
    
    