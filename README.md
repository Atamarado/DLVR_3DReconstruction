# DLVR_3DReconstruction

### Authors:
* *Bokor, Krisztián*
* *Carreto Picón, Ginés*
* *Johler, Marc*

Deep learning-based project in 3D Reconstruction, created as part of the [***Deep Learning For Visual Recognition***](https://kursuskatalog.au.dk/en/course/114392/Deep-Learning-for-Visual-Recognition) course at Aarhus University.

This project focuses in revisiting the publication [*Patch-based reconstruction of a textureless deformable 3d surface from a single rgb image*](https://ieeexplore.ieee.org/document/9022546) and doing some experiments trying different architectures and evaluating its performance.

In order to replicate the architecture and the _patching_ technique, a own created library has been developed (*PatchNet library*), based on `Tensorflow 2.9.2` API.

Please, check out the project report (TBD) and the Demo notebook (`Demo_notebook.ipynb`) to fully understand the project idea and the library.

## *Base environment*

If you want to set up the environment, locally or on a server, you can use Python base 3.9.15, with all the packages determined in `env.yml`.

## *PatchNet library*

*PatchNet library* can be found under `src/` folder, and contains multiple functionalities to allow training and evaluation over the problem proposed.

## _Dataset_ used

The *dataset* used for this project was created for the article [Learning to Reconstruct Texture-less Deformable Surfaces from a Single View](https://arxiv.org/abs/1803.08908), and is freely available to download [**here**](https://www.epfl.ch/labs/cvlab/data/texless-defsurf-data/). 

A preprocessed version created for this project can be downloaded [**using this link**](https://drive.google.com/file/d/1Wg2dB8y98aektVxC70ZPl62QjtSBxiYZ/view?usp=sharing). You can find the script that led to this preprocessed version in the file `preprocess/prepare_data.py`.

Please, remember to cite the original creators of the _dataset_ if you use it.
