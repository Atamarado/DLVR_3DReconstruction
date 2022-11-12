# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:55:26 2022

@author: Marc Johler
"""
import numpy as np
import tensorflow as tf
from scipy.optimize import least_squares

def compute_interval_overlap(interval1, interval2):
    min1 = np.min(interval1)
    min2 = np.min(interval2)
    max1 = np.max(interval1)
    max2 = np.max(interval2)
    
    if (min1 >= max2) or (min2 >= max1):
        return np.array([])
    if min2 >= min1 and max2 >= max1:
        return np.array([min2, max1])
    if min2 >= min1 and max2 <= max1:
        return np.array([min2, max2])
    if min2 <= min1 and max2 >= max1:
        return np.array([min1, max1])
    if min2 <= min1 and max2 <= max1:
        return np.array([min1, max2])
        
    
def compute_patch_overlap(patch1_h, patch1_w, patch2_h, patch2_w):
    height_overlap = compute_interval_overlap(patch1_h, patch2_h)
    width_overlap = compute_interval_overlap(patch1_w, patch2_w)
    
    return height_overlap, width_overlap


def compute_overlap_matrix(height_intervals, width_intervals):
    n_patches = len(height_intervals)
    overlap_matrix = np.repeat([np.repeat(None, n_patches)], n_patches, axis = 0)
    for i in range(n_patches - 1):
        for j in range(i + 1, n_patches):
            overlap_matrix[i, j] = np.array(compute_patch_overlap(height_intervals[i], width_intervals[i],
                                                                  height_intervals[j], width_intervals[j]),
                                            dtype = "object")
            
    return overlap_matrix


def compute_pixel_differences(patches, height_intervals, width_intervals, overlap_matrix):
    n_patches = len(patches)
    difference_matrix = np.repeat([np.repeat(None, n_patches)], n_patches, axis = 0)
    for i in range(n_patches - 1):
        patch_i = np.array(patches[i])
        h_ids_i = np.arange(height_intervals[i][0], height_intervals[i][1])
        w_ids_i = np.arange(width_intervals[i][0], width_intervals[i][1])
        for j in range(i + 1, n_patches):
            patch_j = np.array(patches[j])
            h_ids_j = np.arange(height_intervals[j][0], height_intervals[j][1])
            w_ids_j = np.arange(width_intervals[j][0], width_intervals[j][1])
            h_overlap, w_overlap = overlap_matrix[i, j]
            # check if there is overlap in both dimensions
            if len(h_overlap) == 0 or len(w_overlap) == 0:
                continue 
            # otherwise compute differences
            differences = np.array([])
            for k in range(h_overlap[0], h_overlap[1]):
                h_i = h_ids_i == k
                h_j = h_ids_j == k
                for l in range(w_overlap[0], w_overlap[1]):
                    w_i = w_ids_i == l
                    w_j = w_ids_j == l
                    differences = np.append(differences, patch_i[h_i, w_i] - patch_j[h_j, w_j])
            difference_matrix[i, j] = differences
    return difference_matrix
             

def compute_translation_loss(offsets, differences):
    n_patches = differences.shape[0]
    # length of the residuals is triangle number of n_patches - 1
    residuals = np.array([])
    for i in range(n_patches - 1):
        if i == 0:
            offset_i = 0
        else:
            offset_i = offsets[i - 1]
        for j in range(i + 1, n_patches):
            difference = differences[i, j]
            if difference is None:
                residuals = np.append(residuals, 0)
            else:
                residuals = np.concatenate((residuals, difference + offset_i - offsets[j - 1]))
            
    return residuals

def compute_translation_offsets(patches, height_intervals, width_intervals):
    # compute overlaps 
    overlaps = compute_overlap_matrix(height_intervals, width_intervals)
    # compute pixel differences
    differences = compute_pixel_differences(patches, height_intervals, width_intervals, overlaps)
    # optimize translation offset
    x0 = np.zeros(len(patches) - 1)
    LQ_result = least_squares(compute_translation_loss, x0, kwargs = {"differences": differences})
    # return the optimal offsets
    return LQ_result.x

def depth_map_stitching(image_shape, patches, height_intervals, width_intervals, include_offsets = True):
    # exclude the channel dimension
    patches = patches[:,:,:,0]
    # initialize the output variables
    image_depth_map = np.zeros(image_shape[:-1])
    denominators = np.zeros(image_shape[:-1])
    # compute the offsets necessary for stitching
    translation_offsets = compute_translation_offsets(patches, height_intervals, width_intervals)
    # first patch depth map with no offset
    height_interval = height_intervals[0]
    width_interval = width_intervals[0]
    image_depth_map[height_interval[0]:height_interval[1], 
                    width_interval[0]:width_interval[1]] += patches[0]
    denominators[height_interval[0]:height_interval[1], 
                 width_interval[0]:width_interval[1]] += 1
    # compute the final depth_map
    for i in range(len(patches) - 1):
        height_interval = height_intervals[i + 1]
        width_interval = width_intervals[i + 1]
        # add patch depth map to the depth map for the whole image
        image_depth_map[height_interval[0]:height_interval[1], 
                        width_interval[0]:width_interval[1]] += patches[i + 1] + translation_offsets[i] * include_offsets
        denominators[height_interval[0]:height_interval[1], 
                     width_interval[0]:width_interval[1]] += 1
    # return depth_map for the whole image
    return tf.convert_to_tensor(image_depth_map / denominators)


def normalize_predictions(patches):
    normalized_patches = np.array(patches)
    for i in range(len(normalized_patches)):
        normalize_with = np.reshape(np.sum(normalized_patches[i]**2, axis = -1)**(0.5),
                                    (patches.shape[1], patches.shape[2], 1))
        normalized_patches[i] = normalized_patches[i] / normalize_with
    return tf.convert_to_tensor(normalized_patches)

# THIS FUNCTION IS INCOMPLETE
def normals_map_stitching(image_shape, patches, height_intervals, width_intervals):
    # initialize an average map
    normals_map = np.zeros(image_shape)
    denominators = np.zeros(image_shape)
    # normalize patches
    patches = normalize_predictions(patches)
    # compute the final normals map
    for i in range(len(patches)):
        height_interval = height_intervals[i]
        width_interval = width_intervals[i]
        # add patch depth map to the depth map for the whole image
        normals_map[height_interval[0]:height_interval[1], 
                    width_interval[0]:width_interval[1]] += patches[i]
        denominators[height_interval[0]:height_interval[1], 
                     width_interval[0]:width_interval[1]] += 1
    # normalize the average normals map again
    average_normals = np.reshape(normals_map / denominators, tuple([1]) + image_shape)
    return normalize_predictions(average_normals)[0]


def pad_patch(patch, goal_dim, h_i_min, h_i_max, w_i_min, w_i_max, ones):
    goal_height, goal_width = goal_dim
    channels = patch.shape[-1]
    # measure how much padding is required
    add_above = h_i_min 
    add_below = goal_height - h_i_max
    add_left = w_i_min
    add_right = goal_width - w_i_max

    paddings = tf.constant(([add_above, add_below], [add_left, add_right]))
    
    all_channels = tf.zeros((goal_height, goal_width, 0), dtype = tf.dtypes.float32)
    all_denominators = tf.zeros((goal_height, goal_width, 0), dtype = tf.dtypes.float32)
    
    padded_ones = tf.reshape(tf.pad(ones, paddings), (goal_height, goal_width, 1))
    
    for i in range(channels):
        padded_patch = tf.reshape(tf.pad(patch[:,:,i], paddings), (goal_height, goal_width, 1))
        all_channels = tf.concat([all_channels, padded_patch], axis = -1)
        all_denominators = tf.concat([all_denominators, padded_ones], axis = -1)
    return all_channels, all_denominators
    
def feature_map_stitching(patches, n_height, n_width, decoder_dim):
    # batch_size > 1 not possible currently
    if len(patches.shape) == 5:
        patches = patches[0]
    
    height, width, channels = patches[0].shape
    
    # assert that height of the patches don't exceed decoder dimension
    if height > decoder_dim[0]:
        raise Exception("decoder dimension (height) is too small for patch size")
    if width > decoder_dim[1]:
        raise Exception("decoder dimension (width) is too small for patch size")
    
    # apply the stitching for each channel
    stitched_map = tf.zeros((decoder_dim[0] , decoder_dim[1], channels))
    # compute the number of active patches for each feature in the common map
    denominators = tf.zeros((decoder_dim[0] , decoder_dim[1], channels))
    # ones for the denominators
    ones = tf.ones((height, width), dtype = tf.dtypes.float32)
    
    # compute overlap
    height_overlap = n_height * height - decoder_dim[0]
    width_overlap = n_width * width - decoder_dim[1]
    
    # assert that overlaps are positive
    if height_overlap < 0:
        raise Exception("decoder dimension (height) is too large for patch size and number of patches")
    if width_overlap < 0:
        raise Exception("decoder dimension (width) is too large for patch size and number of patches")
    
    # per patch overlap
    if n_height > 1:
        height_overlap_average = height_overlap / (n_height - 1)
    else:
        height_overlap_average = 0
    if n_width > 1:
        width_overlap_average = width_overlap / (n_width - 1)
    else:
        width_overlap_average = 0
    
    for i in range(len(patches)):
        height_level = np.floor(i / n_width)
        
        h_i_min = int(height_level * height - np.round(height_overlap_average * height_level))
        h_i_max = h_i_min + height
        
        weight_level = i % n_width
        
        w_i_min = int(weight_level * width - np.round(width_overlap_average * weight_level))
        w_i_max = w_i_min + width
        
        padded_patch, padded_ones = pad_patch(patches[i], decoder_dim, 
                                              h_i_min, h_i_max, w_i_min, w_i_max,
                                              ones)
        
        stitched_map = stitched_map + padded_patch
        denominators = denominators + padded_ones
    
    return stitched_map / denominators