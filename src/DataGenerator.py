import tensorflow as tf
import os
import numpy as np
from PIL import Image

# Inspired by: https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 path,
                 batch_size,
                 shuffle=True,
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
            np.random.shuffle(self.objs)

        n_objs = len(self.objs)
        if validation:
            self.n = n_objs * train_val_split
        else:
            self.n = n_objs * (1.0-train_val_split)
    
    def on_epoch_end(self):
        pass
    
    def __get_input(self, name):
        img = Image.open(os.path.join(self.imagePath, name+".tiff"))
        return np.array(img)
    
    def __get_data(self, batches):
        # Generates data containing batch_size samples
        X_batch = np.asarray([self.__get_input(name) for name in batches])
        y_batch = np.asarray([self.__get_output(name) for name in batches])

        # X_batch: Dimensions: (224, 224, 3): The base image in RGB. Range (0, 255)
        # y_batch: Dimensions: (224, 224, 4): Depth (y_batch[:, :, 0]) and normal (y_batch[:, :, 1:4]) maps

        if self.patching:
            pass
            # TODO: Implement patching functions and normalize each patch. Get foreground mask. See comment above
        else:
            return X_batch, y_batch
    
    def __get_output(self, name):
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
        X, y = self.__get_data(batches)
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size