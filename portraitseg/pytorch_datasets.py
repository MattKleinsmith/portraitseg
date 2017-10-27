import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class FlickrPortraitMaskDataset(Dataset):
    """
    Dataset for:
        Paper: Automatic Portrait Segmentation for Image Stylization
        URL: http://xiaoyongshen.me/webpage_portrait/index.html
    Args:
        root (string): Root directory of dataset. It should contain the
            cropped/ and raw/ directories.
        train (bool, optional): If True, creates a dataset from
            the training set, otherwise creates it from the test set.
        transform (callable, optional): Optional transform to be
            applied on an input.
        target_transform (callable, optional): Optional transform to
            be applied on a target.
    """

    def __init__(self, root, train=True,
                 transform=None, target_transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        cropped = os.path.join(root, "cropped/")
        self.portrait_dir = cropped + "portraits/"
        self.mask_dir = cropped + "masks/targets/"
        listname = "trainlist" if train else "testlist"
        self.ids = np.load(root + listname + "_clean.npy")
        self.ext = ".jpg"

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        name = "%05d" % self.ids[index]
        portrait = Image.open(self.portrait_dir + name + self.ext)
        mask = Image.open(self.mask_dir + name + self.ext)
        if self.transform:
            portrait = self.transform(portrait)
        if self.target_transform:
            mask = self.target_transform(mask)
        return portrait, mask
