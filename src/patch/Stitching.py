# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:55:26 2022

@author: Krisztián Bokor, Ginés Carreto Picón, Marc Johler
"""
import numpy as np
import tensorflow as tf
from scipy.optimize import least_squares
from cv2 import bilateralFilter

# compute the overlap of two 1-dimensional intervals
def compute_interval_overlap(interval1, interval2):
    """
    Parameters
    ----------
    interval1 : 1-dimensional array-like of length 2
        first interval to be compared, defined by start- and endpoint
    interval2 : 1-dimensional array-like of length 2
        second interval to be comapred, defined by start- and endpoint

    Returns
    -------
    1-dimensional array-like of length 2 or None
        start and endpoint of overlap interval or None if overlap is empty 
    """
    min1 = np.min(interval1)
    min2 = np.min(interval2)
    max1 = np.max(interval1)
    max2 = np.max(interval2)
    
    # this code block should be self-explanatory
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
        
# function to compute the overlap region of two patches
def compute_patch_overlap(patch1_h, patch1_w, patch2_h, patch2_w):
    """
    Parameters
    ----------
    patch1_h : 1-dimensional array-like of length 2
        start and enpoint of first patch in the height dimension
    patch1_w : 1-dimensional array-like of length 2
        start and enpoint of first patch in the width dimension
    patch2_h : 1-dimensional array-like of length 2
        start and enpoint of second patch in the height dimension
    patch2_w : 1-dimensional array-like of length 2
        start and enpoint of second patch in the width dimension

    Returns
    -------
    height_overlap : 1-dimensional array-like of length 2 or None
        start and endpoint of overlap interval in the height dimension
        or None if overlap is empty 
    width_overlap : 1-dimensional array-like of length 2 or None
        start and endpoint of overlap interval in the width dimension
        or None if overlap is empty

    """
    height_overlap = compute_interval_overlap(patch1_h, patch2_h)
    width_overlap = compute_interval_overlap(patch1_w, patch2_w)
    
    return height_overlap, width_overlap


# compute a matrix, containing the overlaps of all pairs of patches
def compute_overlap_matrix(height_intervals, width_intervals):
    """
    Parameters
    ----------
    height_intervals : list
        a list of start and end points in the height dimension to reconstruct 
        the full image later
    width_intervals : list
        a list of start and end points in the width dimension to reconstruct 
        the full image later

    Returns
    -------
    overlap_matrix : numpy-array
        matrix containing the overlap of each pair of patches represented by a
        height interval and a width interval

    """
    n_patches = len(height_intervals)
    overlap_matrix = np.repeat([np.repeat(None, n_patches)], n_patches, axis = 0)
    # iterate over all the patches in the input lists
    for i in range(n_patches - 1):
        for j in range(i + 1, n_patches):
            overlap_matrix[i, j] = np.array(compute_patch_overlap(height_intervals[i], width_intervals[i],
                                                                  height_intervals[j], width_intervals[j]),
                                            dtype = "object")
            
    return overlap_matrix


# compute pixel-wise difference of two patches in their overlap region
def compute_pixel_differences(patches, height_intervals, width_intervals, overlap_matrix):
    """
    Parameters
    ----------
    patches : numpy.array
        an array of patches resulting from the input
    height_intervals : list
        a list of start and end points in the height dimension to reconstruct 
        the full image later
    width_intervals : list
        a list of start and end points in the width dimension to reconstruct 
        the full image later
    overlap_matrix : numpy-array
        matrix containing the overlap of each pair of patches represented by a
        height interval and a width interval

    Returns
    -------
    difference_matrix : numpy-array
        matrix containing lists of pixel-wise differences for each pair of 
        patches. Same dimensions as the overlap_matrix

    """
    n_patches = len(patches)
    difference_matrix = np.repeat([np.repeat(None, n_patches)], n_patches, axis = 0)
    # loop over all pairs of patches
    for i in range(n_patches - 1):
        patch_i = np.array(patches[i])
        h_ids_i = np.arange(height_intervals[i][0], height_intervals[i][1])
        w_ids_i = np.arange(width_intervals[i][0], width_intervals[i][1])
        for j in range(i + 1, n_patches):
            patch_j = np.array(patches[j])
            h_ids_j = np.arange(height_intervals[j][0], height_intervals[j][1])
            w_ids_j = np.arange(width_intervals[j][0], width_intervals[j][1])
            # find overlap region for the two patches
            h_overlap, w_overlap = overlap_matrix[i, j]
            # check if there is overlap in both dimensions
            if len(h_overlap) == 0 or len(w_overlap) == 0:
                continue 
            # otherwise compute differences
            differences = np.array([])
            for k in range(h_overlap[0], h_overlap[1]):
                # access the respective entry via boolean indexing
                h_i = h_ids_i == k
                h_j = h_ids_j == k
                for l in range(w_overlap[0], w_overlap[1]):
                    w_i = w_ids_i == l
                    w_j = w_ids_j == l
                    # add the pixel-difference value to the list of differences
                    # of the pair of patches
                    differences = np.append(differences, patch_i[h_i, w_i] - patch_j[h_j, w_j])
            difference_matrix[i, j] = differences
    return difference_matrix
             
# compute the loss of the patch translation 
def compute_translation_loss(offsets, differences):
    """
    Parameters
    ----------
    offsets : array-like containing numerical values
        the offsets which are applied to all the patches 
        (starting from the 2nd patch, since the first offset is set to 0)
    difference_matrix : numpy-array
        matrix containing lists of pixel-wise differences for each pair of 
        patches. 

    Returns
    -------
    residuals : array-like 
        a list of residuals based on the offsets

    """
    n_patches = differences.shape[0]
    # length of the residuals is triangle number of n_patches - 1
    residuals = np.array([])
    for i in range(n_patches - 1):
        # access the correct offset value
        if i == 0:
            offset_i = 0
        else:
            offset_i = offsets[i - 1]
        for j in range(i + 1, n_patches):
            # compute the difference
            difference = differences[i, j]
            if difference is None:
                # if there is no overlap the difference is 0
                residuals = np.append(residuals, 0)
            else:
                residuals = np.concatenate((residuals, difference + offset_i - offsets[j - 1]))
            
    return residuals

# compute the optimal translation offsets for the patches, which need to be 
# stitched together
def compute_translation_offsets(patches, height_intervals, width_intervals):
    """
    Parameters
    ----------
    patches : numpy.array
        an array of patches resulting from the input
    height_intervals : list
        a list of start and end points in the height dimension to reconstruct 
        the full image later
    width_intervals : list
        a list of start and end points in the width dimension to reconstruct 
        the full image later
        
    Returns
    -------
    array-like
        solution of the LQ problem computed by scipy.optimize.least_squares()

    """
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
    """
    Parameters
    ----------
    intervals : array-like
        a list of intervals 

    Returns
    -------
    array-like
        a list of indices which correspond to the boundary region in the respective dimension

    """
    unique_intervals = np.unique(np.stack(intervals), axis = 0)
    if len(unique_intervals) == 1:
        raise("there is only one patch")
    # now generate a list of indices to adapt
    begin_patch = unique_intervals[1:, 0]
    end_patch = unique_intervals[:-1, 1]
    return np.concatenate([begin_patch - 1, begin_patch, end_patch - 1, end_patch])

# compute the boundary region in height and width dimension
def compute_boundary_regions_2D(height_intervals, width_intervals):
    """
    Parameters
    ----------
    height_intervals : list
        a list of start and end points in the height dimension to reconstruct 
        the full image later
    width_intervals : list
        a list of start and end points in the width dimension to reconstruct 
        the full image later

    Returns
    -------
    height_indices : array-like
        boudary indices in height dimension
    width_indices : array-like
        boundary indices in width dimension

    """
    height_indices = compute_boundary_regions_1D(height_intervals)
    width_indices = compute_boundary_regions_1D(width_intervals)
    
    return height_indices, width_indices

# function to apply bilateral filtering in the regions aroung patch boundaries
def smoothen_boundaries(stitched_map, height_intervals, width_intervals, sigma):
    """
    Parameters
    ----------
    stitched_map : array-like
        an array constructed by stitching together patches
    height_intervals : list
        a list of start and end points in the height dimension to reconstruct 
        the full image later
    width_intervals : list
        a list of start and end points in the width dimension to reconstruct 
        the full image later
    sigma : numeric
        the variance used as sigmaColor in cv2.bilateralFilter

    Returns
    -------
    stitched_map: array-like
        the stitched array after applying the bilateral filtering

    """
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

# function to stitch the depth maps together
def depth_map_stitching(image_shape, patches, height_intervals, width_intervals, 
                        apply_smoothing = True, sigma = 10, include_offsets = True):
    """

    Parameters
    ----------
    image_shape : tuple of length 3
        shape of the initial image
    patches : tensorflow.Tensor
        list of depth maps which shall be stitched together
    height_intervals : list
        a list of start and end points in the height dimension to reconstruct 
        the full image later
    width_intervals : list
        a list of start and end points in the width dimension to reconstruct 
        the full image later
    apply_smoothing : boolean, optional
        Shall bilateral filtering be applied or not. The default is True.
    sigma : numeric, optional
        The variance used as sigmaColor if bilateral filtering is used. The default is 10.
    include_offsets : TYPE, optional
        Shall the translation offsets be computed and applied?. The default is True.

    Returns
    -------
    average_depths: tensorflow.Tensor
        The stitched depth map

    """
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


# function to normalize the pixel values over the channel dimension
def normalize_predictions(patches):
    """

    Parameters
    ----------
    patches : tensorflow.Tensor
        list of normals maps which shall be normalized

    Returns
    -------
    normalized_patches: tensorflow.Tensor
        list of normal maps which are guaranteed to contain normalized values

    """
    normalized_patches = np.array(patches)
    for i in range(len(normalized_patches)):
        normalize_with = np.reshape(np.sum(normalized_patches[i]**2, axis = -1)**(0.5),
                                    (patches.shape[1], patches.shape[2], 1))
        # exception handling to avoid zero division errors
        normalize_with[np.where(normalize_with == 0)] = 1
        normalized_patches[i] = normalized_patches[i] / normalize_with
    return tf.convert_to_tensor(normalized_patches)

# function to stitch normal maps together
def normals_map_stitching(image_shape, patches, height_intervals, width_intervals, apply_smoothing = True):
    """
    Parameters
    ----------
    image_shape : tuple of length 3
        shape of the initial image
    patches : tensorflow.Tensor
        list of normals maps which shall be stitched together
    height_intervals : list
        a list of start and end points in the height dimension to reconstruct 
        the full image later
    width_intervals : list
        a list of start and end points in the width dimension to reconstruct 
        the full image later
    apply_smoothing : boolean, optional
        Shall bilateral filtering be applied or not. The default is True.

    Returns
    -------
    average_normals: tensorflow.Tensor
        the stitched normals map

    """
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