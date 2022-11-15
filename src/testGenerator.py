from DataGenerator import DataGenerator
import os
from matplotlib import pyplot as plt

data_path = "C:\\Users\\xZoCk\\DLVR_3DReconstruction\\preprocess\\data\\pnData\\train"
os.chdir(data_path)

if __name__ == "__main__":
    datagen = DataGenerator(data_path, 32, patching=False)
    X_batch, y_batch = datagen.__getitem__(1)
    plt.imshow(X_batch[0][:,:,-1])
    