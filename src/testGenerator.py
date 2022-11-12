from DataGenerator import DataGenerator

if __name__ == "__main__":
    datagen = DataGenerator('preprocess/data/pnData', 32)
    X_batch, y_batch = datagen.__getitem__(1)

    print(X_batch)