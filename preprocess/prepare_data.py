import os
import shutil
import numpy as np

def create_empty_folder(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.mkdir(path)

def main():
    ORIGIN_FOLDER = 'data/textureless_deformable_surfaces'
    DEST_FOLDER = 'data/pnData'
    INFO_TYPE = ['depth_maps', 'images', 'normals']

    PATCH_SIZE = 128

    # Create folder structure
    create_empty_folder(DEST_FOLDER)
    for i in INFO_TYPE:
        create_empty_folder(os.path.join(DEST_FOLDER, i))

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
                rootName = "_".join([cat, iType, pow])
                baseDestPath = os.path.join(os.path.join(DEST_FOLDER, iType), rootName)

                fileList = os.listdir(powPath)
                # For each file in the filelist
                if iType == "images":
                    for file in fileList:
                        shutil.copy(os.path.join(powPath, file), baseDestPath+file) # powPath It should be patches of 128 x 128
                elif iType == "depth_maps":
                    for file in fileList:
                        map = np.load(os.path.join(powPath, file))
                        map = map.f.depth
                        patchList = [
                            map[:PATCH_SIZE, :PATCH_SIZE],
                            map[:PATCH_SIZE, -PATCH_SIZE:],
                            map[-PATCH_SIZE:, :PATCH_SIZE],
                            map[-PATCH_SIZE:, -PATCH_SIZE:]
                        ]
                        patchMaps = np.array(patchList)
                        np.save(baseDestPath + file, patchMaps)
                else:
                    for file in fileList:
                        map = np.load(os.path.join(powPath, file))
                        map = map.f.normals
                        patchList = [
                            map[:PATCH_SIZE, :PATCH_SIZE],
                            map[:PATCH_SIZE, -PATCH_SIZE:],
                            map[-PATCH_SIZE:, :PATCH_SIZE],
                            map[-PATCH_SIZE:, -PATCH_SIZE:]
                        ]
                        patchMaps = np.array(patchList)
                        np.save(baseDestPath + file, patchMaps)

    # depth = np.load("data/textureless_deformable_surfaces/cloth/depth_maps/Lc_left_edge/depth_0051.npz")
    # print("Hola")


if __name__=="__main__":
    main()