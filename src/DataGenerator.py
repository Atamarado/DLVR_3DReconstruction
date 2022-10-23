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
        self.train_val_split = 0.2
        self.x_folder = os.path.join(path, 'render_data/')
        self.y_folder = os.path.join(path, 'model_data/')
        
        self.objs = os.listdir(os.path.join(path, self.y_folder))
        if shuffle:
            np.random.shuffle(self.objs)

        n_objs = len(self.objs)
        if validation:
            self.n = n_objs * train_val_split
        else:
            self.n = n_objs * (1-train_val_split)
    
    def on_epoch_end(self):
        pass
    
    def __get_input(self, name):
        objFolder = os.path.join(self.x_folder, name)

        # Load main image
        mainImg = tf.keras.preprocessing.image.load_img(os.path.join(objFolder, 'main.png'))
        mainImg_arr = tf.keras.preprocessing.image.img_to_array(mainImg)

        # Load additional views
        # THE MAIN IMAGE IS LOADED HERE TOO! IT'S ON PURPOSE!!!!
        imgList = os.listdir(objFolder)
        otherImages = []
        for img in imgList:
            otherImages.append(tf.keras.prorpocessing.image.img_to_array(tf.keras.preprocessing.image_load_img(os.path.join(objFolder, img))))

        return tuple([mainImg_arr, otherImages])
    
    def __get_data(self, batches):
        # Generates data containing batch_size samples
        X_batch = np.asarray([self.__get_input(name) for name in batches])
        y_batch = np.asarray([self.__get_output(name) for name in batches]])
        return X_batch, y_batch
    
    def __get_output(self, name):
        # TODO: Decide output format for the model as the y_data
        pass

    def __getitem__(self, index):
        batches = self.objs[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size