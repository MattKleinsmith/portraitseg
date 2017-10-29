import random

import numpy as np
import PIL
import pytest
import torch

from portraitseg.pytorch_datasets import FlickrPortraitMaskDataset
from portraitseg.utils import get_flickr_id

ROOT = "../data/portraits/flickr/"
TRAINLIST_PATH = ROOT + "trainlist_clean.npy"
TESTLIST_PATH = ROOT + "testlist_clean.npy"
HEIGHT = 800
WIDTH = 600
NB_CHANNELS = 3


@pytest.fixture(scope="module")
def trainlist():
    return np.load(TRAINLIST_PATH)


@pytest.fixture(scope="module")
def dataset_train():
    return FlickrPortraitMaskDataset(root=ROOT)


@pytest.fixture(scope="module")
def dataset_test():
    return FlickrPortraitMaskDataset(root=ROOT, train=False)


@pytest.fixture(scope="module")
def testlist():
    return np.load(TESTLIST_PATH)


class TestFlickrPortraitMaskDataset(object):

    def test_len_train(self, dataset_train, trainlist):
        assert len(dataset_train) == len(trainlist)

    def test_getitem_train(self, dataset_train, trainlist):
        index = random.randint(0, len(dataset_train) - 1)
        portrait, mask = dataset_train[index]
        assert isinstance(portrait, torch.FloatTensor)
        assert isinstance(mask, torch.LongTensor)
        portrait_fname = dataset_train.portrait_filenames[0]
        mask_fname = dataset_train.mask_filenames[0]
        portrait_id = get_flickr_id(portrait_fname)
        mask_id = get_flickr_id(mask_fname)
        assert portrait_id == mask_id
        assert portrait_id in trainlist
        assert portrait.shape == (NB_CHANNELS, HEIGHT, WIDTH)
        assert mask.shape == (HEIGHT, WIDTH)

    def test_len_test(self, dataset_test, testlist):
        assert len(dataset_test) == len(testlist)

    def test_getitem_test(self, dataset_test, testlist):
        index = random.randint(0, len(dataset_test) - 1)
        portrait, mask = dataset_test[index]
        assert isinstance(portrait, torch.FloatTensor)
        assert isinstance(mask, torch.LongTensor)
        portrait_fname = dataset_test.portrait_filenames[0]
        mask_fname = dataset_test.mask_filenames[0]
        portrait_id = get_flickr_id(portrait_fname)
        mask_id = get_flickr_id(mask_fname)
        assert portrait_id == mask_id
        assert portrait_id in testlist
        assert portrait.shape == (NB_CHANNELS, HEIGHT, WIDTH)
        assert mask.shape == (HEIGHT, WIDTH)
