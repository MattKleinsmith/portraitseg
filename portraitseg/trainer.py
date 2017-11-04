import os
import os.path as osp
from collections import OrderedDict
import datetime
import shutil
from time import sleep
import fcntl
import sqlite3

import bcolz
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from PIL import Image
import pandas as pd
from pandas.io.sql import DatabaseError
import pytz
import psutil
import seaborn
import tqdm

import torch
from torch import nn
from torch.optim import SGD
from torch.autograd import Variable
import torchfcn

from portraitseg.utils import (plots, set_seed,
                               detransform_portrait,
                               detransform_mask,
                               show_portrait_pred_mask,
                               scoretensor2mask,
                               split_trn_val,
                               cross_entropy2d,
                               get_log_dir,
                               create_log,
                               git_hash)

from portraitseg.portraitfcn import PortraitFCNPlus

###############################################################################

# TODO: Test, and ensure it doesn't pick up PortraitFCN or PortraitFCN+
# https://github.com/wkentaro/pytorch-fcn/blob/master/examples/voc/train_fcn32s.py
def get_parameters(model, bias=False):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.Sequential,
        torchfcn.models.FCN32s,
        torchfcn.models.FCN16s,
        torchfcn.models.FCN8s,
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None
        elif isinstance(m, modules_skipped):
            continue
        else:
            raise ValueError('Unexpected module: %s' % str(m))

###############################################################################

TIMEZONE = pytz.timezone('America/Los_Angeles')
NCOLS = 80

###############################################################################

configurations = {
    1: OrderedDict(
        data_aug=1,
        lr=1e-10,
        momentum=0,
        weight_decay=0,
        lr_bias=2e-14,          ### oops
        weight_decay_bias=0,
        optimizer=SGD,
        loss_fn=cross_entropy2d,
        max_iter=100000,
        loss_fn_kwargs={"size_average": False},
        optimizer_kwards={},
        update_interval=1,
        shuffle_every_epoch=True),
    2: OrderedDict(
        data_aug=1,
        lr=1e-10,
        momentum=0.99,
        weight_decay=0.0005,
        lr_bias=2e-10,
        weight_decay_bias=0,
        optimizer=SGD,
        loss_fn=cross_entropy2d,
        max_iter=100000,
        loss_fn_kwargs={"size_average": False},
        update_interval=1,  # 10 in authors' code
        shuffle_every_epoch=True),
    3: OrderedDict(
        data_aug=1,
        lr=1e-14,
        momentum=0.99,
        weight_decay=0.0005,
        lr_bias=2e-14,
        weight_decay_bias=0,
        optimizer=SGD,
        loss_fn=cross_entropy2d,
        max_iter=100000,
        loss_fn_kwargs={"size_average": False},
        update_interval=1,
        shuffle_every_epoch=True),
    4: OrderedDict(
        data_aug=1,
        lr=1e-14,
        momentum=0.99,
        weight_decay=0.0005,
        lr_bias=2e-14,
        weight_decay_bias=0,
        optimizer=SGD,
        loss_fn=cross_entropy2d,  # Replace with CrossEntropyLoss2d
        max_iter=100000,
        loss_fn_kwargs={"size_average": False},
        update_interval=1,
        shuffle_every_epoch=True),
    5: OrderedDict(
        data_aug=1,
        lr=1e-4,  # Divide color channels by 255   # NaN
        momentum=0.99,
        weight_decay=0.0005,
        lr_bias=2e-4,
        weight_decay_bias=0,
        optimizer=SGD,
        loss_fn=cross_entropy2d,
        max_iter=200000,
        loss_fn_kwargs={"size_average": False},
        update_interval=10,
        shuffle_every_epoch=True),
    6: OrderedDict(
        data_aug=2,  # first data aug, see notes
        lr=1e-4,
        momentum=0.99,
        weight_decay=0.0005,
        lr_bias=2e-4,
        weight_decay_bias=0,
        optimizer=SGD,
        loss_fn=cross_entropy2d,
        max_iter=200000,
        loss_fn_kwargs={"size_average": False},
        update_interval=10,
        shuffle_every_epoch=True),
    7: OrderedDict(
        data_aug=1,
        lr=1e-4,  # Divide color channels by 255   # NaN
        momentum=0.99,
        weight_decay=0.0005,
        lr_bias=2e-4,
        weight_decay_bias=0,
        optimizer=SGD,
        loss_fn=cross_entropy2d,
        max_iter=200000,
        loss_fn_kwargs={"size_average": False},
        update_interval=10,
        shuffle_every_epoch=True),
    8: OrderedDict(
        data_aug=1,
        lr=1e-10,
        momentum=0,
        weight_decay=0,
        lr_bias=2e-10,
        weight_decay_bias=0,
        optimizer=SGD,
        loss_fn=cross_entropy2d,
        max_iter=200000,
        loss_fn_kwargs={"size_average": False},
        optimizer_kwards={},
        update_interval=10,
        shuffle_every_epoch=True),
}

###############################################################################

# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/trainer.py
class Trainer(object):

    def __init__(self, model, dataset, config_id, resume=None, sample=None,
                 sample_epochs=20, evaluation_interval=4000, valid_size=0.2,
                 detransform_target=None, detransform_input=None):

        self.model = model

        self.inputs, self.targets = dataset

        self.sample = sample
        self.n_samples = len(self.inputs) if not self.sample else self.sample

        self.config_id = config_id
        self.config = configurations[self.config_id]

        self.data_aug = self.config['data_aug']
        self.lr = self.config['lr']
        self.momentum = self.config['momentum']
        self.weight_decay = self.config['weight_decay']
        self.lr_bias = self.config['lr_bias']
        self.weight_decay_bias = self.config['weight_decay_bias']
        self.optimizer = self.config['optimizer']
        self.loss_fn = self.config['loss_fn']
        self.max_iter = self.config['max_iter']
        self.loss_fn_kwargs = self.config['loss_fn_kwargs']
        self.update_interval = self.config['update_interval']
        self.shuffle_every_epoch = self.config['shuffle_every_epoch']

        parameter_groups = [
            {
                'params': get_parameters(self.model, bias=False)},
            {
                'params': get_parameters(self.model, bias=True),
                'lr': self.lr_bias,
                'weight_decay': self.weight_decay_bias}]

        self.optimizer = self.optimizer(parameter_groups,
                                        lr=self.lr,
                                        momentum=self.momentum,
                                        weight_decay=self.weight_decay)

        self.resume = resume
        if self.resume:
            checkpoint = torch.load(self.resume)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(
                checkpoint['optim_state_dict'])
            self.epoch = checkpoint['epoch']
            self.iteration = checkpoint['iteration']
            self.best_iteration = self.iteration
            self.best_mean_iu = checkpoint['best_mean_iu']
            self.val_loss = checkpoint['val_loss']
            self.trn_loss = checkpoint['trn_loss']
            self.log_dir = osp.dirname(self.resume)
        else:
            self.epoch = 0
            self.iteration = 0
            self.best_mean_iu = 0
            self.log_dir = get_log_dir(config_id, self.config, self.sample)

        if self.sample:
            self.evaluation_interval = self.n_samples
            self.sample_epochs = sample_epochs
        else:
            self.evaluation_interval = evaluation_interval

        self.valid_size = valid_size

        self.indices_trn, self.indices_val = split_trn_val(
            self.n_samples, valid_size=valid_size, shuffle=False)
        self.n_trn = len(self.indices_trn)
        self.n_val = len(self.indices_val)

        if not osp.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.log_csvpath = osp.join(self.log_dir, "log.csv")
        self.table_name = "master_log"
        self.master_log_path = osp.join(osp.dirname(self.log_dir),
                                        "%s.sqlite" % self.table_name)

        self.log_header = [
            'epoch',
            'iteration',
            'train/loss',
            'train/acc',
            'train/acc_cls',
            'train/mean_iu',
            'train/fwavacc',
            'valid/loss',
            'valid/acc',
            'valid/acc_cls',
            'valid/mean_iu',
            'valid/fwavacc',
            'elapsed_time'
        ]

        self.master_log_header = [
            'timestamp',
            'cfg',
            'best_iteration',
            'best_mean_iou',
            'val_loss',
            'trn_loss',
            'val_trn_ratio',
            'sample_size',
            'git_commit',
            'time_elapsed'
        ]

        create_log(self.log_csvpath, self.log_header)

        # Set up visualizations
        self.input_v = self.inputs[self.indices_val[0]]
        self.target_v = detransform_target(self.targets[self.indices_val[0]])
        self.inputs_v = Variable(
            torch.from_numpy(self.input_v).float().cuda())[None, :]
        self.input_v = detransform_input(self.input_v)
        self.preds = []
        self.fig = plt.figure(num=1, figsize=(9.5, 0))
        self.colors = seaborn.xkcd_palette(['red', 'green'])

        self.timestamp_start = datetime.datetime.now(TIMEZONE)

    def visualize(self):
        # Examples of outputs
        # TODO: Make this code more general (less portrait specific)
        outputs = self.model(self.inputs_v)
        scoretensor = outputs[0].data.cpu()
        pred = scoretensor2mask(scoretensor)
        self.preds.append(pred)
        show_portrait_pred_mask(self.input_v, self.preds, self.target_v,
                                self.evaluation_interval, fig=self.fig)
        self.fig.savefig(osp.join(self.log_dir, "portrait_pred_mask.png"))
        # Learning curves
        df = pd.read_csv(self.log_csvpath)
        if len(df) > 1:
            fig = plt.figure(2)
            x = df['iteration']
            plt.xlabel("iteration")
            plt.ylabel("log loss")
            modes = [("train", self.colors[0]), ("valid", self.colors[1])]
            for mode, color in modes:
                y = np.log(df['%s/loss' % mode])
                plot = plt.plot(x, y, color=color, linestyle="-",
                                label="%s loss" % mode, markersize=1)
            plt.legend()
            fig.savefig(osp.join(self.log_dir, "learning_curves.png"))
            fig.clear()
            plt.figure(1)

    def checkpoint(self, is_best):
        checkpoint = {
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optimizer.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_iu': self.best_mean_iu,
            'val_loss': self.val_loss,
            'trn_loss': self.trn_loss}
        torch.save(checkpoint, osp.join(self.log_dir, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(osp.join(self.log_dir, 'checkpoint.pth.tar'),
                        osp.join(self.log_dir, 'model_best.pth.tar'))

    def get_master_log(self):
        try:
            conn = sqlite3.connect(self.master_log_path)
            df = pd.read_sql("SELECT * FROM %s" % self.table_name, conn)
            print("Loading SQLite database.")
        except DatabaseError as e:
            if 'no such table' in e.args[0]:
                print("Creating SQLite database.")
                df = pd.DataFrame(columns=self.master_log_header)
            else:
                print(e)
                raise Exception("Failed to create database. Unknown error.")
        return df, conn

    def update_master_log(self):
        elapsed_time = ((datetime.datetime.now(TIMEZONE) -
                        self.timestamp_start).total_seconds())
        row = dict(
            timestamp=self.timestamp_start.strftime("%Y-%m-%d--%H-%M-%S"),
            cfg=self.config_id,
            best_iteration=self.best_iteration,
            best_mean_iou=self.best_mean_iu,
            val_loss=self.val_loss,
            trn_loss=self.trn_loss,
            val_trn_ratio=self.val_loss / self.trn_loss,
            sample_size=self.sample,
            git_commit=git_hash().decode("utf-8"),
            time_elapsed=elapsed_time)
        df, conn = self.get_master_log()
        df.loc[len(df)] = row
        df.to_sql(self.table_name, conn, index=False, if_exists="replace")

    def update_log(self, trn_loss, trn_metrics, val_loss, val_metrics):
        with open(self.log_csvpath, "a") as f:
            elapsed_time = ((datetime.datetime.now(TIMEZONE) -
                            self.timestamp_start).total_seconds())
            log = ([self.epoch, self.iteration] +
                   [trn_loss] + list(trn_metrics) +
                   [val_loss] + list(val_metrics) +
                   [elapsed_time])
            log = map(str, log)
            f.write(",".join(log) + "\n")

    def _evaluate(self, on_training_set=False):
        if on_training_set:
            mode = "trn"
            indices = self.indices_trn
            n = self.n_trn
        else:
            mode = "val"
            indices = self.indices_val
            n = self.n_val
        running_loss = 0.0
        label_trues = []
        label_preds = []
        tqdm_iter = tqdm.tqdm(enumerate(indices), total=n,
                              desc=("Eval (%s) (iter. %d)" %
                                    (mode, self.iteration)),
                              ncols=NCOLS, leave=False)
        for i, index in tqdm_iter:
            loss, outputs = self.calculate_loss(index)
            running_loss += loss.data[0]
            label_true = self.targets[index].astype(np.int64)
            label_trues.append(label_true)
            label_pred = outputs.data.max(1)[1].cpu().numpy()[0]
            label_preds.append(label_pred)
        metrics = torchfcn.utils.label_accuracy_score(
                label_trues, label_preds, self.model.n_class)
        return running_loss, metrics

    def evaluate(self):
        print(self.iteration)
        self.model.eval()
        trn_loss, trn_metrics = self._evaluate(on_training_set=True)
        val_loss, val_metrics = self._evaluate(on_training_set=False)
        self.update_log(trn_loss, trn_metrics, val_loss, val_metrics)
        mean_iu = val_metrics[2]
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
            self.best_iteration = self.iteration
            self.val_loss = val_loss
            self.trn_loss = trn_loss
        if not self.sample:
            self.checkpoint(is_best)
        self.visualize()
        self.model.train()

    def calculate_loss(self, index):
        inp = self.inputs[index]
        target = self.targets[index]
        if self.config_id == 5:
            print(inp.dtype)
            inp /= 255
        inputs = Variable(
            torch.from_numpy(inp).float().cuda())[None, :]
        targets = Variable(
            torch.from_numpy(target).long().cuda())[None, :]
        outputs = self.model(inputs)
        if self.loss_fn_kwargs:
            loss = self.loss_fn(outputs, targets, **self.loss_fn_kwargs)
        else:
            loss = self.loss_fn(outputs, targets)
        if np.isnan(float(loss.data[0])):
            print("Iteration: %d" % self.iteration)
            raise ValueError("Loss is NaN")
        return loss, outputs

    def train_epoch(self):
        self.model.train()
        if self.shuffle_every_epoch:
            np.random.shuffle(self.indices_trn)
        tqdm_iter = tqdm.tqdm(enumerate(self.indices_trn), total=self.n_trn,
                              desc='Train (epoch %d)' % self.epoch,
                              ncols=NCOLS, leave=False)
        for i, index in tqdm_iter:
            iteration = i + (self.epoch * self.n_trn)
            in_sync = (iteration == self.iteration + 1)
            if not in_sync:
                continue  # Speed through indices to sync tqdm
            self.iteration = iteration
            if iteration % self.evaluation_interval == 0:
                self.evaluate()
            loss, _ = self.calculate_loss(index)
            loss.backward()
            if iteration % self.update_interval == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            if self.iteration >= self.max_iter:
                break

    def train(self):
        if self.sample:
            max_epoch = self.sample_epochs
        else:
            max_epoch = int(np.ceil(1. * self.max_iter / self.n_trn))
        tqdm_iter = tqdm.trange(self.epoch, max_epoch, desc='Train',
                                ncols=NCOLS, leave=True)
        try:
            for epoch in tqdm_iter:
                self.epoch = epoch
                self.train_epoch()
                if self.iteration >= self.max_iter:
                    break
        except KeyboardInterrupt:
            pass
        if self.best_mean_iu:
            self.update_master_log()
