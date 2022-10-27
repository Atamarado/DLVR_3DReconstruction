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
    # check if number of sets of patches is accurate for a depth map
    assert len(patches) == 1
    # overwrite patches with their only entry
    patches = patches[0]
    image_depth_map = np.zeros(image_shape)
    denominators = np.zeros(image_shape)
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

# THIS FUNCTION IS INCOMPLETE
def normals_map_stitching(image_shape, patches, height_intervals, width_intervals):
    # check if number of sets of patches is accurate for a normals map
    assert len(patches) == 3
    n_patches = len(patches[0])
    patch_shape = patches[0][0].shape
    image_normals_map = tf.repeat(tf.zeros(image_shape).reshape((image_shape[0], image_shape[1], 1)), 3, axis = 2)
    denominators = image_normals_map.copy()
    # compute the final normals map
    for i in range(n_patches):
        sq_dir = tf.zeros(patch_shape).reshape((patch_shape[0], patch_shape[1]), 1)
        for axis in range(3):
            sq_dir = patches[axis][i]**2
    pass
            

def feature_map_stitching(patches, n_height, n_width):
    if len(patches.shape) == 4:
        patches = patches[0]
    
    height, width, channels = patches[0].shape

    # apply the stitching for each channel
    stitched_map = np.zeros((height * n_height , width * n_width, channels))
    for i in range(len(patches)):
        h_i_min = int(np.floor(i / n_width) * height)
        h_i_max = h_i_min + height
        
        w_i_min = int((i % n_width) * width)
        w_i_max = w_i_min + width
        
        stitched_map[h_i_min:h_i_max,w_i_min:w_i_max] = patches[i]

    return stitched_map