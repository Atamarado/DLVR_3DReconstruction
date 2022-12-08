# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:58:19 2022

@author: Krisztián Bokor, Ginés Carreto Picón, Marc Johler
"""
import numpy as np
import tensorflow as tf

# use this function to split a matrix-representation of an image into patches
def patch_matrix(img_matrix, patch_size, n_height, n_width, height_ratio, width_ratio):
    """
    Parameters
    ----------
    img_matrix : array-like
        2D matrix representation of an image
    patch_size : integerish
        the size of the square window which defines the patches
    n_height : integerish
        the number of patches in the height dimension
    n_width : integerish
        the number of patches in the width dimension
    height_ratio : integerish
        the number of pixels in the height dimension between the 
        first pixel of two overlapping patches
    width_ratio : integerish
        the number of pixels in the height dimension between the 
        first pixel of two overlapping patches

    Returns
    -------
    patches : numpy.array
        an array of patches resulting from the input
    height_intervals : list
        a list of start and end points in the height dimension to reconstruct 
        the full image later
    width_intervals : list
        a list of start and end points in the width dimension to reconstruct 
        the full image later

    """
    height, width = img_matrix.shape
    
    # compute the cutting points
    # if there are more than one patch necessary
    height_cuts = np.array([height_ratio * i for i in range(0, n_height - 1)])
    width_cuts = np.array([width_ratio * i for i in range(0, n_width - 1)])
    
    # create the patches 
    patches = np.repeat(None, n_height * n_width)
    height_intervals = np.repeat(None, n_height * n_width) 
    width_intervals = np.repeat(None, n_height * n_width)
    # create the patches which are not from completely right or at the bottom
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
    # create the patches at the bottom by considering the image dimensions
    for w in range(n_width - 1):
        width_cut_start = int(np.round(width_cuts[w]))
        width_cut_end = width_cut_start + patch_size
        index = (n_height - 1) * n_width + w
        patches[index] = img_matrix[-patch_size:, width_cut_start:width_cut_end]
        # save the intervals of the pixels (necessary for stitching later)
        height_intervals[index] = np.array([height - patch_size, height])
        width_intervals[index] = np.array([width_cut_start, width_cut_end])
    # last patch in the lower right corner by considering the image dimensions
    patches[-1] = img_matrix[-patch_size:,-patch_size:]
    height_intervals[-1] = np.array([height - patch_size, height])
    width_intervals[-1] = np.array([width - patch_size, width])
    
    return patches, height_intervals, width_intervals  

# This function is used to create patches from an image with variable overlap size
def patching(img, patch_size, return_intervals = True):
    """
    Parameters
    ----------
    img : array-like
        input image of shape (height, width, channels)
    patch_size : integerish
        the size of the square window which defines the patches
    return_intervals : boolean, optional
        If set to TRUE, it returns the intervals defining the patch boundaries
        in addition to the patches. If FALSE returns the number of patches in 
        each dimension. The default is True.

    Returns
    -------
    patches : numpy.array
        an array of patches resulting from the input
    
    height_intervals : list
        a list of start and end points in the height dimension to reconstruct 
        the full image later
    width_intervals : list
        a list of start and end points in the width dimension to reconstruct 
        the full image later 
    OR
    n_height : integerish
        the number of patches in the height dimension
    n_width : integerish
        the number of patches in the width dimension 

    """
    height, width, channels = img.shape
    
    # weird type cast to avoid error in Colab
    height = int(height)
    width = int(width)
    channels = int(channels)
    
    # compute number of necessary patches in each dimension
    n_height = int(np.ceil(float(height) / float(patch_size)))
    n_width = int(np.ceil(float(width) / float(patch_size)))
    
    # compute the number of pixels after which a new patch is necessary
    height_ratio = height / n_height
    width_ratio = width / n_width
    
    height_intervals = None
    width_intervals = None
    patches = np.repeat(None, channels)
    
    # compute the patches for each channel of the input image
    for i in range(channels):
        patches[i], height_intervals, width_intervals = patch_matrix(img[:,:,i], patch_size, 
                                                                     n_height, n_width, height_ratio, width_ratio)
    # return the specified variables
    if return_intervals:
        return patches, height_intervals, width_intervals
    return patches, n_height, n_width

# this function creates patches from an image with fixed overlap size 
def patching_fo(img, patch_size, return_intervals = True):
    """
    Parameters
    ----------
    img : array-like
        input image of shape (height, width, channels)
    patch_size : integerish
        the size of the square window which defines the patches
    return_intervals : boolean, optional
        If set to TRUE, it returns the intervals defining the patch boundaries
        in addition to the patches. If FALSE returns the number of patches in 
        each dimension. The default is True.

    Returns
    -------
    
    patches : numpy.array
        an array of patches resulting from the input
    
    height_intervals : list
        a list of start and end points in the height dimension to reconstruct 
        the full image later
    width_intervals : list
        a list of start and end points in the width dimension to reconstruct 
        the full image later 
    OR
    n_height : integerish
        the number of patches in the height dimension
    n_width : integerish
        the number of patches in the width dimension
    """
    
    height, width, channels = img.shape
    
    # weird type cast to avoid error in Colab
    height = int(height)
    width = int(width)
    channels = int(channels)
    
    # compute number of necessary patches in each dimension
    n_height = int(np.round((height - patch_size) * 2 / patch_size) + 1)
    n_width = int(np.round((width - patch_size) * 2 / patch_size) + 1) 
    
    # compute the number of pixels after which a new patch is necessary
    overlap = int(np.floor(patch_size / 2))
    
    height_intervals = None
    width_intervals = None
    patches = np.repeat(None, channels)
    
    # compute the patches for each channel of the input image
    for i in range(channels):
        patches[i], height_intervals, width_intervals = patch_matrix(img[:,:,i], patch_size, 
                                                                     n_height, n_width, overlap, overlap)
    # return the specified variables
    if return_intervals:
        return patches, height_intervals, width_intervals
    return patches, n_height, n_width

# converts the output of the patching functions above to a keras tensor
def patches_to_tensor(patches, patch_size):
    """
    Parameters
    ----------
    patches : numpy-array
        output of patching or patching_fo
    patch_size : integerish
        the size of the square window which defines the patches

    Returns
    -------
    tensorflow.Tensor
        tensor of patches

    """
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

# top-level function for the user which can be used for different purposes 
# and overlap configurations
def tensor_patching(img, patch_size, fixed_overlaps = False, return_intervals = True):
    """
    Parameters
    ----------
    img : array-like
        input image of shape (height, width, channels)
    patch_size : integerish
        the size of the square window which defines the patches
    fixed_overlaps : boolean, optional
        Creates batches with fixed overlap size if set to TRUE. Otherwise computes
        the overlap variable in order to have a minimal possible number of patches
        in each dimension. The default is False.
    return_intervals : boolean, optional
        If set to TRUE, it returns the intervals defining the patch boundaries
        in addition to the patches. If FALSE returns the number of patches in 
        each dimension. The default is True.

    Returns
    -------
    tensorflow.Tensor
        tensor of patches
    height_info : 
        Either the cutting points of patches or the number of patches in the 
        height dimension dependent on the variable 'return_intervals'
    width_info : 
        Either the cutting points of patches or the number of patches in the 
        width dimension dependent on the variable 'return_intervals'

    """
    if fixed_overlaps:
        patches, height_info, width_info = patching_fo(img, patch_size, return_intervals)
    else:
        patches, height_info, width_info = patching(img, patch_size, return_intervals)
    return patches_to_tensor(patches, patch_size), height_info, width_info
    

