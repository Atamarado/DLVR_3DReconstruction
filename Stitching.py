# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:55:26 2022

@author: Marc Johler
"""
import numpy as np

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
        patch_i = patches[i]
        h_ids_i = np.arange(height_intervals[i][0], height_intervals[i][1])
        w_ids_i = np.arange(width_intervals[i][0], width_intervals[i][1])
        for j in range(i + 1, n_patches):
            patch_j = patches[j]
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
        
    