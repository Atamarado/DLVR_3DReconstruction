import os
import shutil
import re

def create_empty_folder(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.mkdir(path)

def main():
    ORIGIN_FOLDER = 'data/textureless_deformable_surfaces'
    DEST_FOLDER = 'data/pnData'
    INFO_TYPE = ['images', 'depth_maps', 'normals']

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
                rootName = "_".join([cat, pow])
                baseDestPath = os.path.join(os.path.join(DEST_FOLDER, iType), rootName)

                fileList = os.listdir(powPath)
                for file in fileList:
                    shutil.copy(os.path.join(powPath, file), baseDestPath+re.sub(".*_", "_", file))

                # # For each file in the filelist
                # if iType == "images":
                #     for file in fileList:
                #         img = Image.open(os.path.join(powPath, file))
                #         imarray = np.array(img)/255
                #         patchList = np.array([
                #             imarray[:PATCH_SIZE, :PATCH_SIZE],
                #             imarray[:PATCH_SIZE, -PATCH_SIZE:],
                #             imarray[-PATCH_SIZE:, :PATCH_SIZE],
                #             imarray[-PATCH_SIZE:, -PATCH_SIZE:]
                #         ])
                #         np.save(baseDestPath+file.replace(".tiff", "").replace("rgb"), patchList)
                #
                # elif iType == "depth_maps":
                #     for file in fileList:
                #         map = np.load(os.path.join(powPath, file))
                #         map = map.f.depth
                #         np.save(baseDestPath + file.replace(".npz", ""), map)
                #
                #         # Patch images here and on normal maps too
                # else:
                #     for file in fileList:
                #         map = np.load(os.path.join(powPath, file))
                #         map = map.f.normals
                #         np.save(baseDestPath + file.replace(".npz", ""), map)

    # depth = np.load("data/textureless_deformable_surfaces/cloth/depth_maps/Lc_left_edge/depth_0051.npz")
    # print("Hola")


if __name__=="__main__":
    main()