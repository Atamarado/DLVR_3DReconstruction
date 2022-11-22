# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 17:45:12 2022

@author: Marc Johler
"""

from patch.PatchNet_tf import PatchNet
from DataGenerator import DataGenerator
from Feed_data import train, test

#
epochs = 1
patch_size = 128
min_channels = 8
batch_size = 32
train_path = "data\\pnData\\train"


# train process happens as follows
patchnet = PatchNet(patch_size, min_channels)
datagen = DataGenerator(train_path, batch_size, patching = True, patch_size = patch_size)

# use train to train patchnet
train(patchnet, datagen, epochs, n_train_batches = 8, n_val_batches = 2)

# test with test data
test_path = "data\\pnData\\test"
datagen_test = DataGenerator(test_path, batch_size, patching = False, train_val_split = 1.0)

test(patchnet, datagen_test, n_batches = 2)

print("Finished")
