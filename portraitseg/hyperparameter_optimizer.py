from collections import OrderedDict

import numpy as np
from torch.optim import SGD, Adam, RMSprop
from torch.nn.functional import cross_entropy

from portraitseg.portraitfcn import PortraitFCNPlus
from portraitseg.trainer import Trainer
from portraitseg.utils import (choose,
                               print_separator,
                               get_max_of_db_column,
                               insert_into_table,
                               update_table)


class HyperparameterOptimizer(object):

    def __init__(self, dataset, sample_size, epochs, db_connect_str):
        """Defaults to random search.
           No other algorithms have been implemented yet."""

        self.dataset = dataset
        self.sample_size = sample_size
        self.epochs = epochs
        self.db_connect_str = db_connect_str
        self.table_name = "configurations"

    def optimize(self):
        while True:
            print_separator()
            config = self.choose_random_config()
            model = PortraitFCNPlus(dropout=config['dropout']).cuda()
            trainer = Trainer(model, self.dataset, config, self.db_connect_str,
                              sample_size=self.sample_size,
                              epochs=self.epochs)
            try:
                trainer.train()
            except ValueError:
                # Assumption: Loss was NaN.
                pass  # End trial and choose another config.
            except KeyboardInterrupt:
                trainer.update_trials_table(stopped_early=True)
                print("KeyboardInterrupt")
                print_separator()
                break

    def choose_random_config(self):
        highest_id = get_max_of_db_column(self.db_connect_str,
                                          self.table_name,
                                          "id")
        config_id = highest_id + 1
        # To reserve ID in table for this config as soon as possible.
        insert_into_table(self.db_connect_str,
                          self.table_name,
                          dict(id=config_id))

        # np.linspace and np.logspace default to 50 samples.
        mirror = choose([True, False])
        random_crop = choose([True, False])
        dropout = choose(np.linspace(0, 0.9))
        optimizer = choose([SGD, Adam, RMSprop])
        optimizer_name = optimizer.__name__
        if optimizer_name in ['Adam', 'RMSprop']:
            lr = choose(np.logspace(-12, 0, base=10))
            lr_bias = choose(2 * np.logspace(-12, 0, base=10))
        elif optimizer_name == 'SGD':
            lr = choose(np.logspace(-12, -7, base=10))
            lr_bias = choose(2 * np.logspace(-12, -7, base=10))
        weight_decay = choose(1e-4 * np.linspace(0, 9))
        weight_decay_bias = choose(1e-4 * np.linspace(0, 9))
        nesterov = choose([True, False])
        if nesterov:
            dampening = 0
            momentum = choose(np.linspace(0.1, .9999))
        else:
            dampening = choose(np.linspace(0, 9))
            momentum = choose(np.linspace(0, .9999))
        divide_by_255 = choose([True, False])
        centered = choose([True, False])

        loss_fn = cross_entropy

        config = OrderedDict(
                     id=config_id,
                     mirror=mirror,
                     random_crop=random_crop,
                     dropout=dropout,
                     lr=lr,
                     momentum=momentum,
                     weight_decay=weight_decay,
                     lr_bias=lr_bias,
                     weight_decay_bias=weight_decay_bias,
                     optimizer=optimizer,
                     loss_fn=loss_fn,
                     divide_by_255=divide_by_255,
                     nesterov=nesterov,
                     dampening=dampening,
                     centered=centered,
                     )
        update_table(self.db_connect_str, self.table_name, config)
        return config
