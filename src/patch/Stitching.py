# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:55:26 2022

@author: Marc Johler
"""
import numpy as np
import tensorflow as tf
from scipy.optimize import least_squares
from cv2 import bilateralFilter

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


# functions to compute boundary regions from a list of index intervals for patches
def compute_boundary_regions_1D(intervals):
    unique_intervals = np.unique(np.stack(intervals), axis = 0)
    if len(unique_intervals) == 1:
        raise("there is only one patch")
    # now generate a list of indices to adapt
    begin_patch = unique_intervals[1:, 0]
    end_patch = unique_intervals[:-1, 1]
    return np.concatenate([begin_patch - 1, begin_patch, end_patch - 1, end_patch])

def compute_boundary_regions_2D(height_intervals, width_intervals):
    height_indices = compute_boundary_regions_1D(height_intervals)
    width_indices = compute_boundary_regions_1D(width_intervals)
    
    return height_indices, width_indices

def smoothen_boundaries(stitched_map, height_intervals, width_intervals, sigma):
    # first compute the indices which need to be changed
    height_indices, width_indices = compute_boundary_regions_2D(height_intervals, width_intervals)
    # convert to float32, since this is necessary for the bilateral filtering
    stitched_map = stitched_map.astype("float32")
    # then compute the bilateral filtering for the whole map
    # NOTE that sigmaSpace is not used anyway, we just have to define it, since cv2 didn't implemented it appropriately
    filtered_map = bilateralFilter(stitched_map, d = 3, sigmaColor = sigma, sigmaSpace = 1)
    # TO-DO: DELETE ME AFTER TRYING
    return filtered_map
    # change values from the initial map to the values of the filtered map
    stitched_map[height_indices] = filtered_map[height_indices]
    stitched_map[:, width_indices] = filtered_map[:, width_indices]
    return stitched_map

def depth_map_stitching(image_shape, patches, height_intervals, width_intervals, 
                        apply_smoothing = True, include_offsets = True, sigma = 10):
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
    average_depths = image_depth_map / denominators
    # apply the bilateral filtering
    if apply_smoothing:
        average_depths = smoothen_boundaries(average_depths, height_intervals, width_intervals, sigma)
    # return depth_map for the whole image
    average_depths = np.reshape(average_depths, average_depths.shape + tuple([1]))
    return tf.convert_to_tensor(average_depths)


def normalize_predictions(patches):
    normalized_patches = np.array(patches)
    for i in range(len(normalized_patches)):
        normalize_with = np.reshape(np.sum(normalized_patches[i]**2, axis = -1)**(0.5),
                                    (patches.shape[1], patches.shape[2], 1))
        # exception handling to avoid zero division errors
        normalize_with[np.where(normalize_with == 0)] = 1
        normalized_patches[i] = normalized_patches[i] / normalize_with
    return tf.convert_to_tensor(normalized_patches)

# THIS FUNCTION IS INCOMPLETE
def normals_map_stitching(image_shape, patches, height_intervals, width_intervals, apply_smoothing = True):
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
    average_normals = normals_map / denominators
    # apply the bilateral filtering
    if apply_smoothing:
        average_normals = smoothen_boundaries(average_normals, height_intervals, width_intervals, 0.3)
    # normalize the average normals map again
    average_normals = np.reshape(average_normals, tuple([1]) + average_normals.shape)
    return normalize_predictions(average_normals)[0]