#!/usr/bin/env python3

import argparse
import os
import os.path as osp
import mkl

import bcolz
import torch

from portraitseg.portraitfcn import PortraitFCNPlus
from portraitseg.trainer import Trainer
from portraitseg.utils import (detransform_portrait,
                               detransform_mask,
                               print_separator)
from portraitseg.configurations import get_config
from portraitseg.hyperparameter_optimizer import HyperparameterOptimizer


###############################################################################

nproc = mkl.get_max_threads()  # e.g. 12
mkl.set_num_threads(nproc)

###############################################################################


def main():
    gpu_default = 0
    gpu_help = """
        (int, required) Selects GPU with the given ID. IDs are those
        shown in nvidia-smi.
        Default: %d
        """ % gpu_default
    config_id_default = None
    config_id_help = """
        (int, required) Selects hyperparamter configuration with the given ID.
        IDs are those shown in the configuration table.
        If no ID is given, the process will run random search until the user
        sends a KeyboardInterrupt siganl (e.g. CTRL+C).
        Default: %d
        """ % config_id_default
    sample_default = 500
    sample_help = """
        (int, optional) Sets the number of samples to train the model
        on. Used to quickly test hyperparameters configurations.
        Default: %d
        """ % sample_default
    epochs_default = 10
    epochs_help = """
        (int, optional) Sets the number of epochs.
        Default: %d
        """ % epochs_default
    resume_help = """
        (str, optional) Checkpoint path.
        """

    #  TODO: choices=configurations.keys() for --config
    parser = argparse.ArgumentParser(description="PortraitFCN+ trainer")
    parser.add_argument("-g", "--gpu", type=int,
                        default=gpu_default, help=gpu_help)
    parser.add_argument("-c", "--config_id", type=int, nargs="?",
                        default=config_id_help, help=config_id_help)
    parser.add_argument("-s", "--sample", type=int, nargs="?",
                        const=sample_default, help=sample_help)
    parser.add_argument("-e", "--epochs", type=int, nargs="?",
                        default=epochs_default, help=epochs_help)
    parser.add_argument("-r", "--resume", help=resume_help)
    args = parser.parse_args()

    gpu = args.gpu
    config_id = args.config_id
    sample = args.sample
    epochs = args.epochs
    resume = args.resume

    data_dir = "../../data/portraits/flickr/cropped"
    superportraits_path = osp.join(data_dir, "training_superportraits.bcolz")
    masks_path = osp.join(data_dir, "training_masks.bcolz")

    superportraits_dataset = bcolz.open(superportraits_path, 'r')
    masks_dataset = bcolz.open(masks_path, 'r')
    dataset = (superportraits_dataset, masks_dataset)
    detransform_input = lambda x: detransform_portrait(x[:3], mean="voc")  # noqa
    detransform_target = detransform_mask

    # Postgres
    db_env_var_names = ['PGDATABASE', 'PGUSER', 'PGPORT', 'PGHOST']
    db_parameters = [os.environ[var_name] for var_name in db_env_var_names]
    db_connect_str = "dbname={} user={} port={} host={}".format(*db_parameters)

    with torch.cuda.device(gpu):
        if config_id:
            config = get_config(config_id)
            model = PortraitFCNPlus(dropout=config['dropout']).cuda()
            trainer = Trainer(model, dataset, config, db_connect_str,
                              sample_size=sample, epochs=epochs, resume=resume,
                              detransform_input=detransform_input,
                              detransform_target=detransform_target)
            try:
                trainer.train()
            except KeyboardInterrupt:
                trainer.update_trials_table(stopped_early=True)
                print("KeyboardInterrupt")
                print_separator()
        else:
            hp_optim = HyperparameterOptimizer(dataset=dataset,
                                               sample_size=sample,
                                               epochs=epochs,
                                               db_connect_str=db_connect_str)
            hp_optim.optimize()


if __name__ == '__main__':
    print()
    main()
    print()
