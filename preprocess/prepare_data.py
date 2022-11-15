import os
import shutil
import re
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2
SEED = 6

def create_empty_folder(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.mkdir(path)

def main():
    ORIGIN_FOLDER = 'data/textureless_deformable_surfaces'
    TRAIN_DEST_FOLDER = 'data/pnData/train'
    TEST_DEST_FOLDER = 'data/pnData/test'
    INFO_TYPE = ['images', 'depth_maps', 'normals']

    PATCH_SIZE = 128

    # Create folder structure
    create_empty_folder(TRAIN_DEST_FOLDER)
    create_empty_folder(TEST_DEST_FOLDER)
    for i in INFO_TYPE:
        create_empty_folder(os.path.join(TRAIN_DEST_FOLDER, i))
        create_empty_folder(os.path.join(TEST_DEST_FOLDER, i))

    categoryList = os.listdir(ORIGIN_FOLDER)
    # For each object category
    for cat in categoryList:
        print("Category", cat)
        catPath = os.path.join(ORIGIN_FOLDER, cat)
        if os.path.isfile(catPath):
            continue

        # For each type of file we want
        for iType in INFO_TYPE:
            print("\tFileType", iType)
            iTypePath = os.path.join(catPath, iType)
            powList = os.listdir(iTypePath)
            # For each point of view
            for pow in powList:
                print("\t\tPow", pow)
                powPath = os.path.join(iTypePath, pow)
                rootName = "_".join([cat, pow])

                baseTrainDestPath = os.path.join(os.path.join(TRAIN_DEST_FOLDER, iType), rootName)
                baseTestDestPath = os.path.join(os.path.join(TEST_DEST_FOLDER, iType), rootName)

                fileList = os.listdir(powPath)
                trainList, testList = train_test_split(fileList, test_size=TEST_SIZE, random_state=SEED)
                for file in trainList:
                    shutil.copy(os.path.join(powPath, file), baseTrainDestPath+re.sub(".*_", "_", file))
                for file in testList:
                    shutil.copy(os.path.join(powPath, file), baseTestDestPath+re.sub(".*_", "_", file))  


if __name__=="__main__":
    main()