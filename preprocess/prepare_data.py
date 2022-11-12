import os
import random
import shutil
import numpy as np
import pickle

# def copy_renderings():
#     DEST_PATH = 'data/render_data/'
#     if os.path.isdir(DEST_PATH):
#         shutil.rmtree(DEST_PATH)
#     os.mkdir(DEST_PATH)
#
#     allObjects = []
#     MIN_IMAGES = 1
#     MAX_IMAGES = 10
#     random.seed(0)
#     ORIGIN_PATH = 'data/ShapeNetRendering/'
#     classes = os.listdir(ORIGIN_PATH)
#
#     print("Start copying renders")
#     for c in classes:
#         print("Copying renders class "+c)
#         # For every object class
#         random.seed(6)
#         classPath = ORIGIN_PATH+c+'/'
#         objects = os.listdir(classPath)
#         allObjects.append(objects)
#         for o in objects:
#             # For each object class
#             path = classPath+o+'/rendering/'
#             files = os.listdir(path)
#             files = [f for f in files if f.endswith('.png')]
#             # Select how many images are going to be selected
#             n_images = random.randint(MIN_IMAGES, MAX_IMAGES)
#             random.shuffle(files)
#             # Copy selected images
#             mainImg = files[0]
#             selectedFiles = files[1:n_images]
#             d_path = DEST_PATH+o+'/'
#             try:
#                 os.mkdir(d_path)
#                 # Copy main image
#                 copyOrigin = path+mainImg
#                 copyDestination = d_path+'main.png'
#                 shutil.copy(copyOrigin, copyDestination)
#
#                 # Copy the other images (if any)
#                 for img in selectedFiles:
#                     copyOrigin = path+img
#                     copyDestination = d_path+img
#                     shutil.copy(copyOrigin, copyDestination)
#             except:
#                 print("Repeated object:", o)
#         return classes[:1], allObjects
#
#     return classes, allObjects
#
# def getObjIds():
#     objects = []
#
#     RENDER_PATH = '/data/ShapeNetRendering/'
#     classes = os.listdir(RENDER_PATH)
#
#     for c in classes:
#         objects.append(os.listdir(RENDER_PATH+c))
#
#     return classes, objects
#
# def process3DMesh(originPath: str, destPath: str):
#     info = {}
#     info['coords'] = None
#     info['support'] = {'stage1':None,'stage2':None,'stage3':None, 'stage4':None}
#     info['unpool_idx'] = {'stage1_2':None,'stage2_3':None, 'stage3_4':None}
#     info['lap_idx'] = {'stage1':None,'stage2':None,'stage3':None,'stage4':None}
#
#     raw_mesh = load_obj(originPath,no_normal=True)
#     mesh = trimesh.Trimesh(vertices=raw_mesh['vertices'], faces=(raw_mesh['faces']-1), process=False)
#     trimesh.grouping.merge_vertices(mesh)
#     # assert np.all(raw_mesh['faces'] == mesh.faces+1)
#     coords_1 = np.array(mesh.vertices, dtype=np.float32)
#     info['coords'] = coords_1
#
#     # Stage 1 Auxiliary matrix
#     adj_1 = nx.adjacency_matrix(mesh.vertex_adjacency_graph, nodelist=range(len(coords_1)))
#     cheb_1 = chebyshev_polynomials(adj_1,1)
#     info['support']['stage1'] = cheb_1
#
#     edges_1 = mesh.edges_unique
#     edges_1 = edges_1[edges_1[:,1].argsort(kind='mergesort')]
#     edges_1 = edges_1[edges_1[:,0].argsort(kind='mergesort')]
#     info['unpool_idx']['stage1_2'] = edges_1
#
#     lap_1 = cal_lap_index(mesh.vertex_neighbors)
#     info['lap_idx']['stage1'] = lap_1
#
#     # Stage 2 Auxiliary matrix
#     faces_1 = np.array(mesh.faces)
#
#     edges_2, faces_2 = unpool_face(faces_1, edges_1, coords_1)
#
#     tmp_1_2 = 0.5*(coords_1[info['unpool_idx']['stage1_2'][:,0]] + coords_1[info['unpool_idx']['stage1_2'][:,1]])
#     coords_2 = np.vstack([coords_1, tmp_1_2])
#
#     mesh2 = trimesh.Trimesh(vertices=coords_2, faces=faces_2, process=False)
#
#     adj_2 = nx.adjacency_matrix(mesh2.vertex_adjacency_graph, nodelist=range(len(coords_2)))
#     cheb_2 = chebyshev_polynomials(adj_2,1)
#     info['support']['stage2'] = cheb_2
#
#     edges_2 = edges_2[edges_2[:,1].argsort(kind='mergesort')]
#     edges_2 = edges_2[edges_2[:,0].argsort(kind='mergesort')]
#     info['unpool_idx']['stage2_3'] = edges_2
#
#     lap_2 = cal_lap_index(mesh2.vertex_neighbors)
#     info['lap_idx']['stage2'] = lap_2
#
#     # Stage 3 Auxiliary matrix
#     edges_3, faces_3 = unpool_face(faces_2, edges_2, coords_2)
#
#     tmp_2_3 = 0.5*(coords_2[info['unpool_idx']['stage2_3'][:,0]] + coords_2[info['unpool_idx']['stage2_3'][:,1]])
#     coords_3 = np.vstack([coords_2, tmp_2_3])
#
#     mesh3 = trimesh.Trimesh(vertices=coords_3, faces=faces_3, process=False)
#
#     adj_3 = nx.adjacency_matrix(mesh3.vertex_adjacency_graph, nodelist=range(len(coords_3)))
#     cheb_3 = chebyshev_polynomials(adj_3,1)
#     info['support']['stage3'] = cheb_3
#
#     edges_3 = edges_3[edges_3[:,1].argsort(kind='mergesort')]
#     edges_3 = edges_3[edges_3[:,0].argsort(kind='mergesort')]
#     info['unpool_idx']['stage3_4'] = edges_3
#
#     lap_3 = cal_lap_index(mesh3.vertex_neighbors)
#     info['lap_idx']['stage3'] = lap_3
#
#     # Stage 4 Auxiliary matrix
#     edges_4, faces_4 = unpool_face(faces_3, edges_3, coords_3)
#
#     tmp_3_4 = 0.5*(coords_3[info['unpool_idx']['stage3_4'][:,0]] + coords_3[info['unpool_idx']['stage3_4'][:,1]])
#     coords_4 = np.vstack([coords_3, tmp_3_4])
#
#     mesh4 = trimesh.Trimesh(vertices=coords_4, faces=faces_4, process=False)
#
#     adj_4 = nx.adjacency_matrix(mesh4.vertex_adjacency_graph, nodelist=range(len(coords_4)))
#     cheb_4 = chebyshev_polynomials(adj_4,1)
#     info['support']['stage4'] = cheb_4
#
#     edges_4 = edges_4[edges_4[:,1].argsort(kind='mergesort')]
#     edges_4 = edges_4[edges_4[:,0].argsort(kind='mergesort')]
#     info['unpool_idx']['stage4_5'] = edges_4
#
#     lap_4 = cal_lap_index(mesh4.vertex_neighbors)
#     info['lap_idx']['stage4'] = lap_4
#
#     # Dump .dat file
#     dat = [info['coords'],
#        info['support']['stage1'],
#        info['support']['stage2'],
#        info['support']['stage3'],
#        info['support']['stage4'],
#        [info['unpool_idx']['stage1_2'],
#         info['unpool_idx']['stage2_3'],
#         info['unpool_idx']['stage3_4']
#        ],
#        [np.zeros((1,4), dtype=np.int32)]*4,
#        [np.zeros((1,4))]*4,
#        [info['lap_idx']['stage1'],
#         info['lap_idx']['stage2'],
#         info['lap_idx']['stage3'],
#         info['lap_idx']['stage4']
#        ],
#       ]
#     pickle.dump(dat, open(destPath,"wb"), protocol=2)
#
# def copy_models(classes, objects):
#     DEST_PATH = 'data/model_data/'
#     if os.path.isdir(DEST_PATH):
#         shutil.rmtree(DEST_PATH)
#     os.mkdir(DEST_PATH)
#
#     ORIGIN_PATH = 'data/ShapeNetCore.v2/'
#
#     print("Copyings models")
#     for i, c in enumerate(classes):
#         print("Copying model class", c)
#         classPath = os.path.join(ORIGIN_PATH, c)+"/"
#         for o in objects[i]:
#             # Mesh processing
#             meshPath = classPath+o+"/models/model_normalized.obj"
#             copyDestination = DEST_PATH+o+".dat"
#             print(meshPath, copyDestination)
#             if not(os.path.isfile(copyDestination)):
#                 process3DMesh(meshPath, copyDestination)

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