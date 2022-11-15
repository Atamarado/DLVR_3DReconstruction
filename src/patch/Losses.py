# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 13:32:55 2022

@author: Marc Johler
"""

import tensorflow as tf
from math import pi

def mean_squared_error(true, pred, batched = True):
    if batched:
        pred = pred[0]
        true = true[0]
    diff = true - pred
    return tf.math.reduce_mean(diff**2)

def depth_loss(pred_patch: tf.Tensor, truth_patch: tf.Tensor, foreground_mask_patch: tf.Tensor) -> tf.float32:
    """Calculates the loss of a patch's relative depth map prediction

    Args:
        pred_patch (tf.Tensor): the prediction of the depth map of a patch
        truth_patch (tf.Tensor): the ground truth, RELATIVE depth map of the same patch
        foreground_mask_patch (tf.Tensor): the foreground mask of the patch (0 - back, 1 - fore)

    Returns:
        tf.float32: the calculated total loss of foreground pixels
    """
    # Calcualte the total absolute error of foreground pixels
    abs_diff: tf.Tensor
    abs_diff = tf.math.abs(tf.math.subtract(truth_patch, pred_patch))
    abs_diff = tf.math.multiply(abs_diff, foreground_mask_patch)
    return tf.math.reduce_sum(abs_diff) / tf.math.reduce_sum(foreground_mask_patch)

def normal_cosine_loss(pred_patch: tf.Tensor, truth_patch: tf.Tensor, foreground_mask_patch: tf.Tensor) -> tf.float32:
    """Calculate the linearized version of the cosine similarity of the predicted normals

    pred_patch (tf.Tensor): the prediction of the normal map of the patch
        truth_patch (tf.Tensor): the ground truth normal map of the patch
        foreground_mask_patch (tf.Tensor): the foreground mask of the patch (0 - back, 1 - fore)

    Returns:
        tf.float32: the calculated total loss of foreground pixels
    """
    # Calculate dot product of each pred pixel's and truth pixel's normal vector
    dot_product = tf.math.reduce_sum(tf.math.multiply(
        pred_patch, truth_patch), axis=-1, keepdims=True)

    # Calculate product of norms of each pixel's pred and truth normal vector
    pred_patch_norms = tf.norm(pred_patch, axis=-1, keepdims=True)
    truth_patch_norms = tf.norm(truth_patch, axis=-1, keepdims=True)

    norm_product = tf.math.multiply(pred_patch_norms, truth_patch_norms)

    # add an epsilon to the norms to avoid zero division
    EPS = 10 ** -6
    norm_product = norm_product + EPS

    # divide the dot products with the norm products
    divided_products = tf.math.divide(dot_product, norm_product)

    # Arccos the ratio
    arccos_ratio = tf.math.acos(divided_products)

    # Multiply with 1 / pi
    arccos_ratio = tf.math.divide(arccos_ratio, pi)

    # Probably not needed cause of keepdims
    # add channel dimension
    # arccos_ratio = tf.reshape(arccos_ratio, arccos_ratio.shape + tuple([1]))

    # Rule out the background loss
    arccos_ratio = tf.math.multiply(arccos_ratio, foreground_mask_patch)

    assert tf.reduce_max(arccos_ratio) <= 1
    assert tf.reduce_min(arccos_ratio) >= 0

    return tf.math.reduce_sum(arccos_ratio)

def length_loss(pred_patch: tf.Tensor, foreground_mask_patch: tf.Tensor) -> tf.float32:
    """Calculate the squared length loss of each pixel's normal vector

    Args:
        pred_patch (tf.Tensor): the prediction of the normal map of the patch
        foreground_mask_patch (tf.Tensor): the foreground mask of the patch (0 - back, 1 - fore)

    Returns:
        tf.float32: the calculated total loss of foreground pixels
    """

    # Calculate the norm of the predicted normal vectors
    norm_pred = tf.norm(pred_patch, axis=-1, keepdims=True)

    # Subtract one (the incentive is to create unit length normals)
    norm_pred = norm_pred - 1

    # Square it
    norm_pred = tf.math.pow(norm_pred, 2)
    
    # add channel dimension
    # norm_pred = tf.reshape(norm_pred, norm_pred.shape + tuple([1]))

    # Remove the background losses
    norm_pred = tf.multiply(norm_pred, foreground_mask_patch)

    return tf.math.reduce_sum(norm_pred)

def normal_loss(pred_patch: tf.Tensor, truth_patch: tf.Tensor, foreground_mask_patch: tf.Tensor) -> tf.float32:
    """Calculates the loss of a patch's normal map prediction

    Args:
        pred_patch (tf.Tensor): the prediction of the normal map of the patch
        truth_patch (tf.Tensor): the ground truth normal map of the patch
        foreground_mask_patch (tf.Tensor): the foreground mask of the patch (0 - back, 1 - fore)

    Returns:
        tf.float32: the calculated average pixel loss
    """

    # Calculate the separate losses
    cosine_loss = normal_cosine_loss(pred_patch=pred_patch, truth_patch=truth_patch, foreground_mask_patch=foreground_mask_patch)
    norm_loss = length_loss(pred_patch=pred_patch, foreground_mask_patch=foreground_mask_patch)

    # Weigh them
    K = 10
    total_loss = K * cosine_loss + norm_loss

    # Return the average per pixel loss
    return total_loss / tf.reduce_sum(foreground_mask_patch)

def prediction_loss(pred_depth_patch: tf.Tensor, depth_patch: tf.Tensor, pred_normal_patch: tf.Tensor, normal_patch: tf.Tensor, foreground_mask_patch: tf.Tensor) -> tf.float32:
    """Calculates the total combined loss of the patch predictions

    Args:
        pred_depth_patch (tf.Tensor): the predicted depth values
        depth_patch (tf.Tensor): the ground-truth depth values
        pred_normal_patch (tf.Tensor): the predicted normal vector values
        normal_patch (tf.Tensor): the ground-truth normal vector values
        foreground_mask_patch (tf.Tensor): the foreground mask of the patch (0 - back, 1 - fore)

    Returns:
        tf.float32: the total average pixel loss
    """

    # Calculate the two losses
    loss_depth = depth_loss(pred_depth_patch, depth_patch, foreground_mask_patch)
    loss_normal = normal_loss(pred_normal_patch, normal_patch, foreground_mask_patch)
    
    return loss_depth + loss_normal
