
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')

import os
os.chdir("../")

import mkl
nproc = mkl.get_max_threads()  # e.g. 12
mkl.set_num_threads(nproc)

###############################################################################

import bcolz
import numpy as np
from PIL import Image


MEAN_BGR = np.array([104.00698793, 116.66876762, 122.67891434])

def transform_portrait(img):
    img = np.array(img, dtype=np.uint8)
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float64)
    img -= MEAN_BGR
    img = img.transpose(2, 0, 1)  # HxWxC --> CxHxW
    #img = torch.from_numpy(img).float()  # Not sure if bcolz can handle Torch tensors
    return img

def transform_mask(mask):
    mask = np.array(mask, dtype=np.int32)
    mask = mask / 255
    #mask = torch.from_numpy(mask).long()  # Not sure if bcolz can handle Torch tensors
    return mask


DATA_DIR = '../data/portraits/flickr/'   
CROP_DIR = DATA_DIR + "cropped/"
PORTRAIT_DIR = CROP_DIR + 'portraits/' 
MASK_DIR = CROP_DIR + 'masks/targets/'
PORTRAIT_BCOLZ_PATH = CROP_DIR + "training_portraits.bcolz"
MASK_BCOLZ_PATH = CROP_DIR + "training_masks.bcolz"

TRAIN_IDs = np.load(DATA_DIR + "trainlist_clean.npy")
TRAIN_NAMES = ["%05d" % i for i in TRAIN_IDs]


# In[2]:

portraits = []
masks = []
for name in TRAIN_NAMES:
    
    portrait = Image.open(PORTRAIT_DIR + name + ".jpg")
    portrait_np = transform_portrait(portrait)
    portraits.append(portrait_np)
    
    mask = Image.open(MASK_DIR + name + ".png")
    mask_np = transform_mask(mask)
    masks.append(mask_np)
    

portraits = np.array(portraits)
print("portraits: %.2f GB in RAM" % (portraits.nbytes / 1024**3))
c = bcolz.carray(portraits, rootdir=PORTRAIT_BCOLZ_PATH, mode='w')
c.flush()
print("Saved to " + PORTRAIT_BCOLZ_PATH)

masks = np.array(masks)
print("masks: %.2f GB in RAM" % (masks.nbytes / 1024**3))
c = bcolz.carray(masks, rootdir=MASK_BCOLZ_PATH, mode='w')
c.flush()
print("Saved to " + MASK_BCOLZ_PATH)

