import os
import random
import shutil

def preprocess_renderings():
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

    for c in classes:
        # For every object class
        random.seed(6)
        classPath = ORIGIN_PATH+c+'/'
        objects = os.listdir(classPath)
        allObjects.append(objects)
        for i, o in enumerate(objects):
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
            os.mkdir(d_path)
            for img in selectedFiles:
                copyOrigin = path+img
                copyDestination = d_path+img
                shutil.copy(copyOrigin, copyDestination)
            
            if i==9: break

        break # Remove when using all of them
    
    return classes, allObjects

def getObjIds():
    objects = []

    RENDER_PATH = '/data/ShapeNetRendering/'
    classes = os.listdir(RENDER_PATH)

    for c in classes:
        objects.append(os.listdir(RENDER_PATH+c))
        
        break # Remove when using all of them

    return classes, objects


def main():
    preprocess_renderings()


if __name__=="__main__":
    main()