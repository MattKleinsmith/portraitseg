import numpy as np
import pytest

from portraitseg.pytorch_dataloaders import (get_train_valid_loader,
                                             get_test_loader)

SEED = 0
ROOT = "../data/portraits/flickr/"
TRAINLIST_PATH = ROOT + "trainlist_clean.npy"
TESTLIST_PATH = ROOT + "testlist_clean.npy"
BATCH_SIZE = 2


@pytest.fixture(scope="module")
def trainlist():
    return np.load(TRAINLIST_PATH)


@pytest.fixture(scope="module")
def dataloaders_train():
    return get_train_valid_loader(ROOT,
                                  batch_size=BATCH_SIZE,
                                  augment=False,
                                  random_seed=SEED,
                                  valid_size=0.2)


@pytest.fixture(scope="module")
def testlist():
    return np.load(TESTLIST_PATH)


@pytest.fixture(scope="module")
def dataloader_test():
    return get_test_loader(ROOT, batch_size=BATCH_SIZE)


def test_get_train_valid_loader(trainlist, dataloaders_train):
    trn_loader, val_loader = dataloaders_train
    all_training_ids = np.load(ROOT + "trainlist_clean.npy")
    nb_trn_batches = len(trn_loader)
    nb_val_batches = len(val_loader)
    assert nb_trn_batches > nb_val_batches
    nb_batches = nb_trn_batches + nb_val_batches
    assert nb_batches == np.ceil(len(all_training_ids) / BATCH_SIZE)


def test_get_test_loader(testlist, dataloader_test):
    all_test_ids = np.load(ROOT + "testlist_clean.npy")
    nb_test_batches = len(dataloader_test)
    assert nb_test_batches == np.ceil(len(all_test_ids) / BATCH_SIZE)
