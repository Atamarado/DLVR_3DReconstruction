import tensorflow as tf
import os
import numpy as np

# Inspired by: https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 path,
                 batch_size,
                 shuffle=True,
                 validation=False,
                 train_val_split=0.2):

        self.batch_size = batch_size
        self.validation = False
        self.train_val_split = train_val_split

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
        try:
            return np.load(os.path.join(self.imagePath, name+".npy"))
        except:
            print("Hola")
    
    def __get_data(self, batches):
        # Generates data containing batch_size samples
        X_batch = np.asarray([self.__get_input(name) for name in batches])
        y_batch = np.asarray([self.__get_output(name) for name in batches])
        return X_batch, y_batch
    
    def __get_output(self, name):
        depth = np.load(os.path.join(self.depthPath, name+".npy"))
        normal = np.load(os.path.join(self.normalPath, name+".npy"))
        return depth, normal

    def __getitem__(self, index):
        batches = self.objs[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size