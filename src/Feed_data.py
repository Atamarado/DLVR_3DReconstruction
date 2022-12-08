# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 13:46:53 2022

@author: Krisztián Bokor, Ginés Carreto Picón, Marc Johler
"""

import math
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Function to loop over a number of batches from the data_generator
def patch_loop(model, data_generator, validation = False, n_batches = math.inf):
    """
    Parameters
    ----------
    model : PatchNet object
        model which shall be trained/validated
    data_generator : DataGenerator object
        data generator which is used for loading and preparing the data
    validation : boolean, optional
        If TRUE uses the validation data. Otherwise uses the training data. The default is False.
    n_batches : integerish, optional
        If it is set lower than the maximum number of batches, only uses the first
        n_batches of the DataGenerator. The default is math.inf.

    Returns
    -------
    loss per patch
        the combined depth and normals loss averaged over the total number of patches
        in all the batches (metric for performance comparison)

    """
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

# function to loop over a set number of batches; will return depth and normals
# loss seperately
def patch_loop_separate_loss(model, data_generator, validation=False, n_batches=math.inf):
    """
    Parameters
    ----------
    model : PatchNet object
        model which shall be trained/validated
    data_generator : DataGenerator object
        data generator which is used for loading and preparing the data
    validation : boolean, optional
        If TRUE uses the validation data. Otherwise uses the training data. The default is False.
    n_batches : integerish, optional
        If it is set lower than the maximum number of batches, only uses the first
        n_batches of the DataGenerator. The default is math.inf.
    
    Returns
    -------
    loss per patch
        the summed depth and normals loss averaged over the total number of patches
        in all the batches (metric for performance comparison)
    depth loss per patch
        the depth loss averaged over the total number of patches
        in all the batches (metric for performance comparison)
    normals loss per patch
        the normals loss averaged over the total number of patches
        in all the batches (metric for performance comparison)
    
    """
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
    for i in tqdm(range(n_batches), desc=desc): # tqdm prints the dynamic 
        inputs, maps = data_generator.__getitem__(i)
        patches = inputs[:, :, :, 0:3]
        foreground_map = tf.reshape(inputs[:, :, :, 3], inputs.shape[:-1] + tuple([1]))
        depth_map = tf.reshape(maps[:, :, :, 0], maps.shape[:-1] + tuple([1]))
        normals_map = maps[:, :, :, 1:]
        # do the respective step
        if validation:
            t_loss, d_loss, n_loss = model.validation_step_separate_loss(patches, foreground_map, depth_map, normals_map)
            total_loss += t_loss
            depth_loss += d_loss
            normal_loss += n_loss
        else:
            t_loss, d_loss, n_loss = model.training_step_separate_loss(patches, foreground_map, depth_map, normals_map)
            total_loss += t_loss
            depth_loss += d_loss
            normal_loss += n_loss
        # remember number of patches
        total_patches += len(patches)

    return total_loss / total_patches, depth_loss / total_patches, normal_loss / total_patches

# loops over a given number of batches of images
def image_loop(model, data_generator, n_batches):
    """
    Parameters
    ----------
    model : PatchNet object
        model which shall be trained/validated
    data_generator : DataGenerator object
        data generator which is used for loading and preparing the data
    n_batches : integerish, optional
        If it is set lower than the maximum number of batches, only uses the first
        n_batches of the DataGenerator. The default is math.inf.

    Returns
    -------
    loss per image
        the summed depth and normals loss averaged over the total number of images 
        in the batches (metric for performance validation/tests)

    """
    # set the options for the data generator
    data_generator.set_validation(True)
    data_generator.set_patching(False)
    n_batches = np.min([data_generator.__len__(), n_batches])
    batch_size = data_generator.batch_size
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
        

# Top-level function for training a PatchNet model
def train(model, data_generator, epochs, n_batches = None, n_train_batches = None, n_val_batches = None):
    """
    Parameters
    ----------
    model : PatchNet object
        model which shall be trained/validated
    data_generator : DataGenerator object
        data generator which is used for loading and preparing the data
    epochs : integerish
        The number of epochs for which the model shall be trained
    n_batches : integerish, optional
        If it is set lower than the maximum number of batches, only uses the first
        n_batches of the DataGenerator. The default is math.inf.
    n_train_batches : integerish, optional
        If it is set lower than the maximum number of TRAINING batches, only uses the first
        n_batches of the DataGenerator for TRAINING. The default is math.inf.
    n_val_batches : integerish, optional
        If it is set lower than the maximum number of VALIDATION batches, only uses the first
        n_batches of the DataGenerator for VALIDATION. The default is math.inf.

    Returns
    -------
    None.

    """
    # if the number of validation or training batches is not set, use the the common batch number 
    if n_val_batches is None and n_train_batches is None:
        n_val_batches = n_batches
        n_train_batches = n_batches
        
    assert n_val_batches is not None
    assert n_train_batches is not None
    
    # repeat the training and validation for 'epoch' times
    for epoch in range(epochs):
        train_loss = patch_loop(model, data_generator, validation = False, n_batches = n_train_batches)
        val_loss_patch = patch_loop(model, data_generator, validation = True, n_batches = n_val_batches)
        val_loss_img = image_loop(model, data_generator, n_batches = n_val_batches)
        
        print("Epoch", epoch, "done with losses:")
        print("Training:", train_loss)
        print("Validation on patches", val_loss_patch)
        print("Validation on images:", val_loss_img)
        
    print("Training finished after", epochs, "Epochs")
    

# Top-level function for validation for testing of the model
def test(model, data_generator, n_batches):
    """
    Parameters
    ----------
     model : PatchNet object
         model which shall be trained/validated
     data_generator : DataGenerator object
         data generator which is used for loading and preparing the data
     n_batches : integerish, optional
         If it is set lower than the maximum number of batches, only uses the first
         n_batches of the DataGenerator. The default is math.inf.

    Returns
    -------
    test_loss : numeric
        The average combined depth and normals loss per image

    """
    test_loss = image_loop(model, data_generator, n_batches = n_batches)
    batch_size = data_generator.batch_size 
    print("Tested on", n_batches * batch_size, "images")
    print("Loss:", test_loss)
    
    return test_loss

    
    
    