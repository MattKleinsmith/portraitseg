
# coding: utf-8

# In[ ]:

get_ipython().magic('matplotlib inline')

import os
os.chdir("/nbs/Portraitseg/")

import numpy as np
from PIL import Image
from scipy.io import loadmat

from portraitseg.utils import get_fnames


def get_portrait_id(portrait_fname):
    """
    Input (string): '../data/portraits/flickr/cropped/portraits/00074.jpg'
    Output (int): 74
    """
    return int(portrait_fname.split('/')[-1].split('.')[0])


DATA_DIR = "../data/portraits/flickr/"
CROP_DIR = DATA_DIR + "cropped/"
PORTRAIT_DIR = CROP_DIR + "portraits/"
MASK_DIR = CROP_DIR + "masks/targets/"
EXT = ".jpg"
TRAINLIST = loadmat(DATA_DIR + "trainlist.mat")['trainlist'][0]
TESTLIST = loadmat(DATA_DIR + "testlist.mat")['testlist'][0]
PORTRAIT_FNAMES = get_fnames(PORTRAIT_DIR)
IDs = [get_portrait_id(fname) for fname in PORTRAIT_FNAMES]
TESTLIST_CLEAN = [x for x in TESTLIST if x in IDs]
TRAINLIST_CLEAN = [x for x in TRAINLIST if x in IDs]
FULLLIST_CLEAN = TESTLIST_CLEAN + TRAINLIST_CLEAN
MISSING = [x for x in IDs if x not in FULLLIST_CLEAN]
TRAINLIST_CLEAN_PLUS = TRAINLIST_CLEAN + MISSING

print(len(TRAINLIST_CLEAN_PLUS) + len(TESTLIST_CLEAN) == len(PORTRAIT_FNAMES))

np.save(DATA_DIR + "trainlist_clean.npy", TRAINLIST_CLEAN_PLUS)
np.save(DATA_DIR + "testlist_clean.npy", TESTLIST_CLEAN)

