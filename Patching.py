# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:58:19 2022

@author: Marc Johler
"""
import numpy as np
import tensorflow as tf

def patch_matrix(img_matrix, patch_size, n_height, n_width):
    height, width = img_matrix.shape
    # type conversion which is necessary for Colab
    width = int(width)
    height = int(height)
    # compute the number of pixels after which a new patch is necessary
    height_ratio = height / n_height
    width_ratio = width / n_width
    # compute the cutting points
    # if there are more than one patch necessary
    height_cuts = np.array([height_ratio * i for i in range(0, n_height - 1)])
    width_cuts = np.array([width_ratio * i for i in range(0, n_width - 1)])
    
    # create the patches 
    patches = np.repeat(None, n_height * n_width)
    height_intervals = np.repeat(None, n_height * n_width) 
    width_intervals = np.repeat(None, n_height * n_width)
    for h in range(n_height - 1):
        height_cut_start = int(np.round(height_cuts[h]))
        height_cut_end = height_cut_start + patch_size
        # patches in the interior of the picture
        for w in range(n_width - 1):
            width_cut_start = int(np.round(width_cuts[w]))
            width_cut_end = width_cut_start + patch_size
            index = h * n_width + w
            patches[index] = img_matrix[height_cut_start:height_cut_end, width_cut_start:width_cut_end]
            # save the intervals of the pixels (necessary for stitching later)
            height_intervals[index] = np.array([height_cut_start, height_cut_end])
            width_intervals[index] = np.array([width_cut_start, width_cut_end])
        # last patch in width
        index = (h + 1) * n_width - 1
        patches[index] = img_matrix[height_cut_start:height_cut_end, -patch_size:]
        # save the intervals of the pixels (necessary for stitching later)
        height_intervals[index] = np.array([height_cut_start, height_cut_end])
        width_intervals[index] = np.array([width - patch_size, width])
    # last patches in height
    for w in range(n_width - 1):
        width_cut_start = int(np.round(width_cuts[w]))
        width_cut_end = width_cut_start + patch_size
        index = (n_height - 1) * n_width + w
        patches[index] = img_matrix[-patch_size:, width_cut_start:width_cut_end]
        # save the intervals of the pixels (necessary for stitching later)
        height_intervals[index] = np.array([height - patch_size, height])
        width_intervals[index] = np.array([width_cut_start, width_cut_end])
    # last patch in the lower right corner
    patches[-1] = img_matrix[-patch_size:,-patch_size:]
    height_intervals[-1] = np.array([height - patch_size, height])
    width_intervals[-1] = np.array([width - patch_size, width])
    
    return patches, height_intervals, width_intervals
    return tf.convert_to_tensor(patches), height_intervals, width_intervals
    

def patching(img, patch_size, return_intervals = True):
    height, width, channels = img.shape
    
    # weird type cast to avoid error in Colab
    height = int(height)
    width = int(width)
    channels = int(channels)
    
    # compute number of necessary patches in each dimension
    n_height = int(np.ceil(float(height) / float(patch_size)))
    n_width = int(np.ceil(float(width) / float(patch_size)))
    
    height_intervals = None
    width_intervals = None
    patches = np.repeat(None, channels)
    
    for i in range(channels):
        patches[i], height_intervals, width_intervals = patch_matrix(img[:,:,i], patch_size, n_height, n_width)
    
    if return_intervals:
        return patches, height_intervals, width_intervals
    return patches, n_height, n_width
        

def patches_to_tensor(patches, patch_size):
    channels = len(patches)
    n_patches = len(patches[0])
    # allocate output space
    patches_array = np.zeros((n_patches, patch_size, patch_size, channels))
    # assign patches to the correct indices
    for i in range(channels):
        channel_i = patches[i]
        for j in range(n_patches):
            patches_array[j, :, :, i] = channel_i[j]
    # convert to tensor once done
    return tf.convert_to_tensor(patches_array)

def tensor_patching(img, patch_size, return_intervals = True):
    patches, n_height, n_width = patching(img, patch_size, return_intervals)
    return patches_to_tensor(patches, patch_size), n_height, n_width
    
    