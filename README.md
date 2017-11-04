# [Work In Progress] Portrait segmentation

# Warning: The commands below don't work yet.

## Input: portrait.jpg
(example)

## Output: mask.png
(example)

## More examples
(more examples, 2x4 grid)

# Quick start

## To turn a portrait into a mask

### Via CPU

`./portrait.sh portrait.jpg`

Uses Docker.

### Via GPU (Linux only)

1. Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker#quick-start) (requires Linux)
2. Run `./portrait.sh --gpu portrait.jpg`

# Docker commands

## To turn a portrait into a mask

### CPU

`docker run --rm mwksmith/portraitseg portrait.jpg`

### GPU

`nvidia-docker run --rm mwksmith/portraitseg:gpu portrait.jpg`

## To explore the code in a Jupyter notebook


### CPU

1. Run `docker run --shm-size=2g -p 8888:8888 mwksmith/portraitseg run_notebook.sh`
2. Navigate to http://localhost:8888/notebooks/portraitseg.ipynb

### GPU

1. Run `nvidia-docker run --shm-size=2g -p 8888:8888 mwksmith/portraitseg:gpu run_notebook.sh`
2. Navigate to http://localhost:8888/notebooks/portraitseg_gpu.ipynb

### To access the Docker container's bash

1. Run `name=$(docker ps --format "{{.Names}}")`
2. Run `docker exec -it $name bash`

### Options

- Jupyter notebook port: 8888
- Shared memory size: 2 GB
    - If memory size becomes an issue, maximize it by replacing `--shm-size=2g` with `--ipc=host`. See [NVIDIA's documentation](http://docs.nvidia.com/deeplearning/dgx/pytorch-release-notes/running.html) for details.


# Manual setup (no Docker)

`git clone`

(and other steps)

# References

## Papers

- Facetracking: [Face Alignment through Subspace Constrained Mean-Shifts](https://www.ri.cmu.edu/pub_files/2009/9/CameraReady-6.pdf) (Saragih *et al.*) (2009)
- FCNs: [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1605.06211.pdf) (Shelhamer *et al.*) (2014, updated 2016)
- Portrait segmentation: [Automatic Portrait Segmentation for Image Stylization](http://xiaoyongshen.me/webpage_portrait/papers/portrait_eg16.pdf) (Shen *et al.*) (2016)


## Code

- [FaceTracker](https://github.com/kylemcdonald/FaceTracker) (Kyle McDonald)
- [PyTorch train-valid-test split DataLoaders](https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb) (kevinzakka)
- [PyTorch FCN implementations](https://github.com/wkentaro/pytorch-fcn) (Kentaro Wada)

## Data

- [Flickr portrait-mask dataset from Shen et al. 2016](https://github.com/PetroWu/AutoPortraitMatting/issues/22) (mirror provided by timbe16)

## Pretrained models

- [Weights for PyTorch FCN8s implementation](https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/models/fcn8s.py#L17) (provided by Kentaro Wada) (includes VGG16 ImageNet weights implicitly)

## Tutorials

- [PyTorch - Using Datasets and DataLoaders](http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [PyTorch - Defining Datasets](http://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
- [Fast.ai - Deep learning course for coders](http://course.fast.ai)


# Licenses

In general: Anything I've created here is under MIT. Anything others have created is under various licenses.

## Code

## Models

## Data

## Implications
