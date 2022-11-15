import tensorflow as tf
import os
import numpy as np
from PIL import Image
from patch.Patching import tensor_patching
import random

# Inspired by: https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 path,
                 batch_size,
                 shuffle=True,
                 seed = random.random(),
                 validation=False,
                 train_val_split=0.2,
                 patching=True,
                 patch_size=128):

        self.batch_size = batch_size
        self.validation = False
        self.train_val_split = train_val_split
        self.patching = patching
        self.patch_size = patch_size

        self.imagePath = os.path.join(path, 'images/')
        self.depthPath = os.path.join(path, 'depth_maps/')
        self.normalPath = os.path.join(path, 'normals/')
        
        self.objs = [os.path.splitext(filename)[0] for filename in os.listdir(self.imagePath)]
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(self.objs)

        n_objs = len(self.objs)
        if validation:
            self.n = n_objs * train_val_split
        else:
            self.n = n_objs * (1.0-train_val_split)
    
    def on_epoch_end(self):
        pass
    
    def __get_input__(self, name):
        img = Image.open(os.path.join(self.imagePath, name+".tiff"))
        return np.array(img)
    
    def __calculate_foreground__(self, img_batch: np.ndarray) -> np.ndarray:
        """
            Assuming imgs has shape (n, h, w, 3)
        """
        zero_bool = img_batch[:, :, :] == [0., 0., 0.]
        n_zeros = zero_bool.sum(axis=3, keepdims=True)
        foreground = (n_zeros != 3).astype(np.float32)

        assert foreground.shape == (
            img_batch.shape[0], img_batch.shape[1], img_batch.shape[2], 1)

        return foreground

    def __normalize_depth__(self, depth_batch: np.ndarray) -> np.ndarray:
        # Calculate average depth of each depth map
        mean_depth_per_batch = depth_batch.mean(axis=(1, 2, 3), keepdims=True)

        # Subtract it from each depth map (broadcasting)
        zero_mean_depth_batch = depth_batch - mean_depth_per_batch

        assert zero_mean_depth_batch.shape == depth_batch.shape

        return zero_mean_depth_batch

    def __get_data__(self, batches):
        # Generates data containing batch_size samples
        X_batch = np.asarray([self.__get_input__(name) for name in batches])
        y_batch = np.asarray([self.__get_output__(name) for name in batches])

        # X_batch: Dimensions: (224, 224, 3): The base image in RGB. Range (0, 255)
        # y_batch: Dimensions: (224, 224, 4): Depth (y_batch[:, :, 0]) and normal (y_batch[:, :, 1:4]) maps

        if self.patching:
            foreground_batch = self.__calculate_foreground__(X_batch)
            depth_batch = np.reshape(y_batch[:, :, :, 0], y_batch.shape[:-1] + tuple([1]))
            normalized_depth_batch = self.__normalize_depth__(depth_batch)
            
            # Shall we normalize normal maps

            y_batch = np.concatenate((normalized_depth_batch, y_batch[:, :, :, 1:]), axis=3)
            conc = np.concatenate((X_batch, y_batch, foreground_batch), axis=3)

            # conc = np.concatenate((X_batch, y_batch), axis=2)
            # TODO: Implement patching functions. Check if that's alright
            X_batch, _, _ = tensor_patching(X_batch, self.patch_size)
            y_batch, _, _ = tensor_patching(y_batch, self.patch_size)

        return X_batch, y_batch
    
    def __get_output__(self, name):
        depth = np.load(os.path.join(self.depthPath, name+".npz"))
        normal = np.load(os.path.join(self.normalPath, name+".npz"))

        depth = depth.f.depth
        normal = normal.f.normals

        # Concatenate both arrays. We need to reshape depth from (w x h) to (w x h x 1)
        depth = np.reshape(depth, (np.shape(depth)[0], np.shape(depth)[1], 1))
        conc = np.concatenate((depth, normal), axis=2)
        return conc

    def __getitem__(self, index):
        batches = self.objs[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data__(batches)
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size