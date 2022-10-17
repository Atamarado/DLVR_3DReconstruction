import os
import random
import shutil

def copy_renderings():
    DEST_PATH = 'data/render_data/'
    if os.path.isdir(DEST_PATH):
        shutil.rmtree(DEST_PATH)
    os.mkdir(DEST_PATH)

    allObjects = []
    MIN_IMAGES = 1
    MAX_IMAGES = 10
    random.seed(0)
    ORIGIN_PATH = 'data/ShapeNetRendering/'
    classes = os.listdir(ORIGIN_PATH)

    print("Start copying renders")
    for c in classes:
        print("Copying renders class "+c)
        # For every object class
        random.seed(6)
        classPath = ORIGIN_PATH+c+'/'
        objects = os.listdir(classPath)
        allObjects.append(objects)
        for o in objects:
            # For each object class
            path = classPath+o+'/rendering/'
            files = os.listdir(path)
            files = [f for f in files if f.endswith('.png')]
            # Select how many images are going to be selected
            n_images = random.randint(MIN_IMAGES, MAX_IMAGES)
            random.shuffle(files)
            # Copy selected images
            selectedFiles = files[:n_images]
            d_path = DEST_PATH+o+'/'
            try:
                os.mkdir(d_path)
                for img in selectedFiles:
                    copyOrigin = path+img
                    copyDestination = d_path+img
                    shutil.copy(copyOrigin, copyDestination)
            except:
                print("Repeated object:", o)
        return classes[:1], allObjects
    
    return classes, allObjects

def getObjIds():
    objects = []

    RENDER_PATH = '/data/ShapeNetRendering/'
    classes = os.listdir(RENDER_PATH)

    for c in classes:
        objects.append(os.listdir(RENDER_PATH+c))

    return classes, objects

def copy_models(classes, objects):
    DEST_PATH = 'data/model_data/'
    if os.path.isdir(DEST_PATH):
        shutil.rmtree(DEST_PATH)
    os.mkdir(DEST_PATH)

    ORIGIN_PATH = 'data/ShapeNetCore.v2/'

    print("Copyings models")
    for i, c in enumerate(classes):
        print("Copying model class", c)
        classPath = os.path.join(ORIGIN_PATH, c)+"/"
        for o in objects[i]:
            copyOrigin = classPath+o+"/models/model_normalized.obj"
            copyDestination = DEST_PATH+o+".obj"
            try:
                shutil.copy(copyOrigin, copyDestination)
            except:
                pass

def main():
    #getObjIds()
    classes, objects = copy_renderings()
    copy_models(classes, objects)




if __name__=="__main__":
    main()