# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:58:19 2022

@author: Marc Johler
"""

import numpy as np

def patch_matrix(img_matrix, patch_size, n_height, n_width):
    height, width = img_matrix.shape
    # compute the number of pixels after which a new patch is necessary
    height_ratio = height / n_height
    width_ratio = width / n_width
    # compute the cutting points
    # if there are more than one patch necessary
    height_cuts = np.array([height_ratio * i for i in range(0, n_height - 1)])
    width_cuts = np.array([width_ratio * i for i in range(0, n_width - 1)])
    
    # create the patches 
    patches = np.repeat(None, n_height * n_width)
    for h in range(n_height - 1):
        height_cut_start = int(np.round(height_cuts[h]))
        height_cut_end = height_cut_start + patch_size
        # patches in the interior of the picture
        for w in range(n_width - 1):
            width_cut_start = int(np.round(width_cuts[w]))
            patches[h * n_width + w] = img_matrix[height_cut_start:height_cut_end, width_cut_start:(width_cut_start + patch_size)]
        # last patch in width
        patches[(h + 1) * n_width - 1] = img_matrix[height_cut_start:height_cut_end, -patch_size:]
    # last patches in height
    for w in range(n_width - 1):
        width_cut_start = int(np.round(width_cuts[w]))
        patches[(n_height - 1) * n_width + w] = img_matrix[-patch_size:, width_cut_start:(width_cut_start + patch_size)]
    # last patch in the lower right corner
    patches[-1] = img_matrix[-patch_size:,-patch_size:]
    
    return patches, height_cuts, width_cuts
    

def patching(img, patch_size):
    height, width, channels = img.shape
    
    output = np.repeat(None, channels)
    
    # compute number of necessary patches in each dimension
    n_height = int(np.ceil(height / patch_size))
    n_width = int(np.ceil(width / patch_size))
    
    height_cuts = None
    width_cuts = None
    patches = np.repeat(None, channels)
    
    for i in range(channels):
        patches[i], height_cuts, width_cuts = patch_matrix(img[:,:,i], patch_size, n_height, n_width)
    
    return patches, height_cuts, width_cuts
        
    