#!/usr/bin/env python3

import argparse
import os
import os.path as osp
import mkl

import bcolz
import torch
import seaborn

from portraitseg.portraitfcn import PortraitFCNPlus
from portraitseg.trainer import Trainer
from portraitseg.utils import (set_seed,
                               get_RAM,
                               detransform_portrait,
                               detransform_mask)


###############################################################################

nproc = mkl.get_max_threads()  # e.g. 12
mkl.set_num_threads(nproc)
SEED = 3
set_seed(SEED)

###############################################################################

def main():
    gpu_default = 0
    gpu_help = """
        (int, required) Selects GPU with the given ID. IDs are those
        shown in nvidia-smi.
        Default: %d
        """ % gpu_default
    config_default = 1
    config_help = """
        (int, required) Selects hyperparamter configuration with the given ID.
        IDs are those shown in the configuration table.
        Default: %d
        """ % config_default
    sample_default = 200
    sample_help = """
        (int, optional) Sets the number of samples to train the model
        on. Used to quickly test hyperparameters configurations.
        Default: %d
        """ % sample_default
    sample_epochs_default = 10
    sample_epochs_help = """
        (int, optional) Sets the number of epochs when using a
        subset of the training set. Requires the --sample flag.
        Default: %d
        """ % sample_epochs_default
    resume_help = """
        (str, optional) Checkpoint path
        """

    #  TODO: choices=configurations.keys() for --config
    parser = argparse.ArgumentParser(description="PortraitFCN+ trainer")
    parser.add_argument("-g", "--gpu", type=int, default=gpu_default,
        help=gpu_help)
    parser.add_argument("-c", "--config", type=int, default=config_default,
        help=config_help)
    parser.add_argument("-s", "--sample", type=int, nargs="?",
        const=sample_default, help=sample_help)
    parser.add_argument("-e", "--sample-epochs", type=int, nargs="?",
        const=sample_epochs_default, help=sample_epochs_help)
    parser.add_argument("-r", "--resume", help=resume_help)
    args = parser.parse_args()

    gpu = args.gpu
    config_id = args.config
    sample = args.sample
    sample_epochs = args.sample_epochs
    resume = args.resume

    data_dir = "../data/portraits/flickr/cropped"
    portraits_path = osp.join(data_dir, "training_superportraits.bcolz")
    masks_path = osp.join(data_dir, "training_masks.bcolz")

    portraits_dataset = bcolz.open(portraits_path, 'r')
    masks_dataset = bcolz.open(masks_path, 'r')
    dataset = (portraits_dataset, masks_dataset)
    detransform_input = lambda x: detransform_portrait(x[:3], mean="voc")
    detransform_target = detransform_mask

    with torch.cuda.device(gpu):
        model = PortraitFCNPlus().cuda()
        trainer = Trainer(model, dataset, config_id, resume,
                          sample=sample, sample_epochs=sample_epochs,
                          detransform_input=detransform_input,
                          detransform_target=detransform_target)
        trainer.train()

if __name__ == '__main__':
    print()
    main()
    print()
