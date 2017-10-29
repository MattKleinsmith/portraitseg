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
        transform (callable, optional): Optional additional transform to
            be applied on an input. Inputs will always be transformed
            as follows:
                - RGB -> BGR
                - Convert to np.float64
                - Subtract BGR mean of dataset
                - HxWxC --> CxHxW
                - Convert to torch.FloatTensor (32-bit floating point)
        target_transform (callable, optional): Optional transform to
            be applied on a target.
    """

    def __init__(self, root, train=True,
                 transform=None, target_transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        # w.r.t. Flickr. Should this be w.r.t. Pascal? w.r.t. ImageNet?
        self.mean_bgr = [104.008, 116.669, 122.675]
        self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

        # To allow the tests to see filenames
        self.portrait_filenames = []
        self.mask_filenames = []

        cropped = os.path.join(root, "cropped/")
        self.portrait_dir = cropped + "portraits/"
        self.mask_dir = cropped + "masks/targets/"
        listname = "trainlist" if train else "testlist"
        self.ids = np.load(root + listname + "_clean.npy")
        self.p_ext = ".jpg"  # Portrait extension
        self.mask_ext = ".png"

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        name = "%05d" % self.ids[index]
        # Load portrait
        portrait = Image.open(self.portrait_dir + name + self.p_ext)
        self.portrait_filenames.append(portrait.filename)
        portrait = self.transform_portrait(portrait)
        # Load mask
        mask = Image.open(self.mask_dir + name + self.mask_ext)
        self.mask_filenames.append(mask.filename)
        mask = self.transform_mask(mask)
        # Apply additional transforms
        if self.transform:
            portrait = self.transform(portrait)
        if self.target_transform:
            mask = self.target_transform(mask)
        return portrait, mask

    def transform_portrait(self, img):
        img = np.array(img, dtype=np.uint8)
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        #img /= 255
        img = img.transpose(2, 0, 1)  # HxWxC --> CxHxW
        img = torch.from_numpy(img).float()
        return img

    def detransform_portrait(self, img):
        """To visualize"""
        img = img.numpy().astype(np.float64)
        img = img.transpose((1, 2, 0)) # CxHxW --> HxWxC
        #img *= 255
        img += self.mean_bgr
        img = img[:, :, ::-1]  # BGR -> RGB
        img = img.astype(np.uint8)
        return img

    def transform_mask(self, mask):
        """Splits the mask into two mask channels."""
        mask = np.array(mask, dtype=np.int32)
        mask_bg = (mask == 0)
        mask_fg = (mask == 255)
        mask_bg = np.array(mask_bg, dtype=np.int32)
        mask_fg = np.array(mask_fg, dtype=np.int32)
        mask = np.zeros((2,)+mask.shape, dtype=np.int32)
        mask[0] = mask_bg
        mask[1] = mask_fg
        mask = torch.from_numpy(mask).float()
        #mask = torch.from_numpy(mask).long()
        return mask

    def detransform_mask(self, mask):
        mask = mask.numpy()[1]
        return mask
