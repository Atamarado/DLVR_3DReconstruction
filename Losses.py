# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 13:32:55 2022

@author: Marc Johler
"""

import tensorflow as tf

def mean_squared_error(true, pred, batched = True):
    if batched:
        pred = pred[0]
        true = true[0]
    diff = true - pred
    return tf.math.reduce_mean(diff**2)

