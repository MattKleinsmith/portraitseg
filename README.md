# Portrait segmentation

If you have questions, please create an issue. Polishing this repo is on my TODO list, but it's not a priority.

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
