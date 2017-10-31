import mkl
nproc = mkl.get_max_threads()  # e.g. 12
mkl.set_num_threads(nproc)
print("nproc: %d" % nproc)

import bcolz
import numpy as np
from PIL import Image
from skimage import transform

from portraitseg.utils import get_fnames, plots, rm_dir_and_ext


MEAN_BGR = np.array([104.00698793, 116.66876762, 122.67891434])


def transform_portrait(img):
    img = np.array(img, dtype=np.uint8)
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float64)
    img -= MEAN_BGR
    #img /= 255  # TODO: Compare training w/ and w/o division by 255
    img = img.transpose(2, 0, 1)  # HxWxC --> CxHxW
    return img


DATA_DIR = "../data/portraits/flickr/"
CROP_DIR = DATA_DIR + "cropped/"
PORTRAIT_DIR = CROP_DIR + "portraits/"
POINTS_DIR = CROP_DIR + "tracker_points/"
REF_POINTS_PATH = CROP_DIR + "tracker_points_of_canonical_pose.npy"
MEAN_MASK_PATH = CROP_DIR + "mean_mask.png"
SUPERPORTRAITS_BCOLZ_PATH = CROP_DIR + "training_superportraits.bcolz"

REF_POINTS = np.load(REF_POINTS_PATH)
SHAPE = Image.open(MEAN_MASK_PATH)
NB_CHANNELS = 6

IM_WIDTH, IM_HEIGHT = 600, 800
PADDING = 600
GRID_WIDTH, GRID_HEIGHT = 1800, 2000
H1, H2 = PADDING, PADDING + IM_HEIGHT
W1, W2 = PADDING, PADDING + IM_WIDTH
XX_GRID, YY_GRID = np.meshgrid(
    np.arange(1, GRID_WIDTH+1, dtype=np.int64),
    np.arange(1, GRID_HEIGHT+1, dtype=np.int64))
GRID_SHAPE = XX_GRID.shape
REF_POS = np.floor(np.mean(REF_POINTS, 0))
XX_GRID = (XX_GRID - PADDING - REF_POS[0]) * 1.0 / PADDING
YY_GRID = (YY_GRID - PADDING - REF_POS[1]) * 1.0 / PADDING
SHAPE_NP = np.array(SHAPE, dtype=np.float64)
SHAPE_GRID = np.pad(SHAPE_NP, (PADDING, PADDING), "minimum")

superportraits = []
fnames = get_fnames(PORTRAIT_DIR)
for i, fname in enumerate(fnames):
    name = rm_dir_and_ext(fname)
    # Create position and shape channels
    dest_points_path = POINTS_DIR + name + ".npy"
    dest_points = np.load(dest_points_path)
    tform = transform.estimate_transform('affine',
                                         dest_points + PADDING,
                                         REF_POINTS + PADDING)
    warped_xx_grid = transform.warp(XX_GRID, tform,
                                    output_shape=GRID_SHAPE)
    warped_yy_grid = transform.warp(YY_GRID, tform,
                                    output_shape=GRID_SHAPE)
    warped_mask_grid = transform.warp(SHAPE_GRID, tform,
                                      output_shape=GRID_SHAPE)
    warped_xx = warped_xx_grid[H1:H2, W1:W2]
    warped_yy = warped_yy_grid[H1:H2, W1:W2]
    warped_mask = warped_mask_grid[H1:H2, W1:W2]
    # Transform portrait, and add position and shape channels
    portrait_path = PORTRAIT_DIR + name + ".jpg"
    portrait = Image.open(portrait_path)
    portrait = transform_portrait(portrait)
    superportrait = np.zeros((NB_CHANNELS, IM_HEIGHT, IM_WIDTH),
                             dtype=np.float64)
    superportrait[:3] = portrait
    superportrait[3] = warped_xx
    superportrait[4] = warped_yy
    superportrait[5] = warped_mask
    superportraits.append(superportrait)

superportraits = np.array(superportraits)
print("superportraits: %.2f GB in RAM"%(superportraits.nbytes/ 024**3))
c = bcolz.carray(superportraits, rootdir=SUPERPORTRAITS_BCOLZ_PATH,
                 mode='w')
c.flush()
print("Saved to " + SUPERPORTRAITS_BCOLZ_PATH)
