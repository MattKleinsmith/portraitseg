import os
import os.path as osp
from datetime import datetime
import shutil
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import seaborn
import tqdm
import yaml

import torch
from torch.autograd import Variable
import torchfcn

from portraitseg.utils import (show_portrait_pred_mask,
                               scoretensor2mask,
                               split_trn_val,
                               create_log,
                               git_hash,
                               plots,
                               insert_into_table,
                               get_max_of_db_column,
                               update_table)
from portraitseg.portraitfcn import FCN8s
from portraitseg.data_augmentations import (mirror,
                                            random_crop)

plt.switch_backend('agg')

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
        FCN8s
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


# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/trainer.py
class Trainer(object):

    def __init__(self, model, dataset, config, db_connect_str, resume=None,
                 sample_size=None, epochs=9999, evaluation_interval=4000,
                 valid_size=0.2, detransform_target=None,
                 detransform_input=None, seed=None):

        self.model = model
        self.inputs, self.targets = dataset
        self.config = config
        self.db_connect_str = db_connect_str

        self.sample = sample_size
        self.n_samples = len(self.inputs) if not self.sample else self.sample

        self.config_id = self.config['id']
        self.mirror = self.config['mirror']
        self.random_crop = self.config['random_crop']
        self.lr = self.config['lr']
        self.momentum = self.config['momentum']
        self.weight_decay = self.config['weight_decay']
        self.lr_bias = self.config['lr_bias']
        self.weight_decay_bias = self.config['weight_decay_bias']
        self.optimizer = self.config['optimizer']
        self.loss_fn = self.config['loss_fn']
        self.nesterov = self.config['nesterov']
        self.dampening = self.config['dampening']
        self.centered = self.config['centered']

        self.seed = seed

        try:
            self.divide_by_255 = self.config['divide_by_255']
        except KeyError:
            self.divide_by_255 = False

        parameter_groups = [
            {
                'params': get_parameters(self.model, bias=False)},
            {
                'params': get_parameters(self.model, bias=True),
                'lr': self.lr_bias,
                'weight_decay': self.weight_decay_bias}]

        if self.optimizer.__name__ == "SGD":
            self.optimizer = self.optimizer(parameter_groups,
                                            lr=self.lr,
                                            weight_decay=self.weight_decay,
                                            momentum=self.momentum,
                                            nesterov=self.nesterov,
                                            dampening=self.dampening)
        elif self.optimizer.__name__ == "Adam":
            self.optimizer = self.optimizer(parameter_groups,
                                            lr=self.lr,
                                            weight_decay=self.weight_decay)
        elif self.optimizer.__name__ == "RMSprop":
            self.optimizer = self.optimizer(parameter_groups,
                                            lr=self.lr,
                                            weight_decay=self.weight_decay,
                                            momentum=self.momentum,
                                            centered=self.centered)

        self.timezone = pytz.timezone('America/Los_Angeles')
        self.timeformat = "%Y-%m-%d %H:%M:%S"
        self.ncols = 80
        self.here = osp.dirname(osp.abspath(__file__))

        self.table_name = "trials"
        self.resume = resume
        if self.resume:
            print("Resuming training.\n")
            checkpoint = torch.load(self.resume)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(
                checkpoint['optim_state_dict'])
            self.epoch = checkpoint['epoch']
            self.iteration = checkpoint['iteration']
            self.start_iteration = self.iteration
            self.best_iteration = self.iteration
            self.best_mean_iou = checkpoint['best_mean_iou']
            self.trn_mean_iou = checkpoint['trn_mean_iou']
            self.val_loss = checkpoint['val_loss']
            self.trn_loss = checkpoint['trn_loss']
            self.trial_id = checkpoint['trial_id']
            try:
                self.preds = checkpoint['preds']
            except KeyError:
                self.preds = []
            self.log_dir = osp.dirname(self.resume)
        else:
            self.epoch = 0
            self.iteration = 0
            self.start_iteration = 0
            self.best_iteration = 0
            self.best_mean_iou = 0
            self.trn_mean_iou = 0
            self.val_loss = None
            self.trn_loss = None
            self.preds = []
            self.make_log_dir()

        self.indices_trn, self.indices_val = split_trn_val(
            self.n_samples, valid_size=valid_size, shuffle=False)
        self.n_trn = len(self.indices_trn)
        self.n_val = len(self.indices_val)

        if self.sample:
            self.evaluation_interval = sample_size
            self.epochs = epochs
        else:
            self.evaluation_interval = evaluation_interval

        self.log_csvpath = osp.join(self.log_dir, "log.csv")
        self.log_header = [
            'epoch',
            'iteration',
            'train/loss',
            'train/acc',
            'train/mean_iou',
            'valid/loss',
            'valid/acc',
            'valid/mean_iou',
            'val_trn_loss_ratio',
            'elapsed_time'
        ]
        create_log(self.log_csvpath, self.log_header)

        self.trials_header = [
            'trial_id',
            'timestamp',
            'cfg',
            'best_iteration',
            'best_mean_iou',
            'trn_mean_iou',
            'val_loss',
            'trn_loss',
            'val_trn_ratio',
            'sample_size',
            'git_commit',
            'time_elapsed',
            'seed'
        ]

        self.visualizer_enabled = True if detransform_target else None
        if self.visualizer_enabled:
            # Set up visualizations
            self.input_v = self.inputs[self.indices_val[0]]
            self.target_v = self.targets[self.indices_val[0]]
            self.target_v = detransform_target(self.target_v)
            self.inputs_v = Variable(
                torch.from_numpy(self.input_v).float().cuda())[None, :]
            self.input_v = detransform_input(self.input_v)
            self.fig_id = 1
            self.fig = plt.figure(num=self.fig_id, figsize=(9.5, 0))
            self.colors = seaborn.xkcd_palette(['red', 'green'])

        self.timestamp_start = datetime.now(self.timezone)

    def _plot_metric(self, log, metric):
        fig = plt.figure(2)
        x = log['iteration']
        plt.xlabel("iteration")
        plt.ylabel(metric)
        modes = [("train", self.colors[0]), ("valid", self.colors[1])]
        for mode, color in modes:
            y = log["%s/%s" % (mode, metric)]
            if metric == "loss":
                y = np.log(y)
            plt.plot(x, y, color=color, linestyle="-",
                     label="%s %s" % (mode, metric), markersize=1)
        plt.legend()
        fig.savefig(osp.join(self.log_dir, "%s_curves_%03d.png" %
                                           (metric, self.config_id)))
        fig.clear()
        plt.figure(self.fig_id)

    def visualize(self):
        # Examples of outputs
        # TODO: Make this code more general (less portrait specific)
        outputs = self.model(self.inputs_v)
        scoretensor = outputs[0].data.cpu()
        pred = scoretensor2mask(scoretensor)
        self.preds.append(pred)
        show_portrait_pred_mask(self.input_v, self.preds, self.target_v,
                                self.start_iteration, self.evaluation_interval,
                                fig=self.fig)
        self.fig.savefig(osp.join(self.log_dir, "val_outputs_masks_%03d.png"
                                  % self.config_id))
        show_portrait_pred_mask(self.input_v, self.preds, self.target_v,
                                self.start_iteration, self.evaluation_interval,
                                fig=self.fig, opacity=1.0)
        self.fig.savefig(osp.join(self.log_dir, "val_outputs_fg_%03d.png"
                                  % self.config_id))
        # Plot metrics
        log = pd.read_csv(self.log_csvpath)
        if len(log) > 1:
            metrics = ["loss", "mean_iou"]
            for metric in metrics:
                self._plot_metric(log, metric)

    def create_checkpoint(self, is_best):
        checkpoint = {
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optimizer.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_iou': self.best_mean_iou,
            'trn_mean_iou': self.trn_mean_iou,
            'val_loss': self.val_loss,
            'trn_loss': self.trn_loss,
            'trial_id': self.trial_id,
            'preds': self.preds}
        torch.save(checkpoint, osp.join(self.log_dir, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(osp.join(self.log_dir, 'checkpoint.pth.tar'),
                        osp.join(self.log_dir, 'model_best.pth.tar'))

    def make_log_dir(self):
        config_copy = self.config.copy()
        for k, v in self.config.copy().items():
            if callable(v) or isinstance(v, type):
                v = v.__name__
            else:
                v = str(v)  # TODO: This doesn't seem to do anything.
            config_copy[k] = v
        # Get current log IDs
        if self.sample:
            logs_dir = osp.join(self.here, "logs/samples")
        else:
            logs_dir = osp.join(self.here, "logs")
        highest_id = get_max_of_db_column(self.db_connect_str,
                                          self.table_name,
                                          "id")
        self.trial_id = highest_id + 1
        insert_into_table(self.db_connect_str,
                          self.table_name,
                          dict(id=self.trial_id))
        print("\nConfig ID: %d\nTrial ID: %05d\n" %
              (self.config_id, self.trial_id))
        name = "%05d_CFG-%03d" % (self.trial_id, self.config_id)
        name += "_GIT-%s" % git_hash().decode("utf-8")
        now = datetime.now(self.timezone)
        name += "_%s" % now.strftime(self.timeformat)
        self.log_dir = osp.join(logs_dir, name)
        if not osp.exists(self.log_dir):
            os.makedirs(self.log_dir)
        with open(osp.join(self.log_dir, "cfg_%03d.yaml" % self.config_id),
                  "w") as f:
            print(dict(config_copy))
            print()
            yaml.safe_dump(dict(config_copy), f, default_flow_style=False)

    def update_log(self, trn_loss, trn_metrics, val_loss, val_metrics):
        with open(self.log_csvpath, "a") as f:
            elapsed_time = ((datetime.now(self.timezone) -
                            self.timestamp_start).total_seconds())
            log = ([self.epoch, self.iteration] +
                   [trn_loss] + list(trn_metrics) +
                   [val_loss] + list(val_metrics) + [val_loss/trn_loss] +
                   [elapsed_time])
            log = map(str, log)
            f.write(",".join(log) + "\n")

    def update_trials_table(self, stopped_early=False):
        elapsed_time = ((datetime.now(self.timezone) -
                        self.timestamp_start).total_seconds())
        if self.trn_loss:
            val_trn_ratio = self.val_loss / self.trn_loss
        else:
            val_trn_ratio = None
        trial = dict(
            id=self.trial_id,
            timestamp=self.timestamp_start.strftime(self.timeformat),
            cfg=self.config_id,
            best_iteration=self.best_iteration,
            best_mean_iou=self.best_mean_iou,
            trn_mean_iou=self.trn_mean_iou,
            val_loss=self.val_loss,
            trn_loss=self.trn_loss,
            val_trn_ratio=val_trn_ratio,
            sample_size=self.n_samples,
            git_commit=git_hash().decode("utf-8"),
            time_elapsed=elapsed_time,
            seed=self.seed,
            stopped_early=stopped_early)
        update_table(self.db_connect_str, self.table_name, trial)
        print("\nBest mean IoU: %s" % self.best_mean_iou)

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
        data_indices = tqdm.tqdm(enumerate(indices), total=n,
                                 desc=("Eval (%s) (iter. %d)" %
                                       (mode, self.iteration)),
                                 ncols=self.ncols, leave=False)
        for i, index in data_indices:
            inp = self.inputs[index]
            target = self.targets[index]
            loss, outputs = self.calculate_loss(inp, target)
            running_loss += loss.data[0]
            label_true = self.targets[index].astype(np.int64)
            label_trues.append(label_true)
            label_pred = outputs.data.max(1)[1].cpu().numpy()[0]
            label_preds.append(label_pred)
        metrics = torchfcn.utils.label_accuracy_score(
                label_trues, label_preds, self.model.n_class)
        metrics = [metrics[i] for i in (0, 2)]
        return running_loss, metrics

    def evaluate(self):
        self.model.eval()
        trn_loss, trn_metrics = self._evaluate(on_training_set=True)
        val_loss, val_metrics = self._evaluate(on_training_set=False)
        self.update_log(trn_loss, trn_metrics, val_loss, val_metrics)
        mean_iou = val_metrics[1]
        is_best = mean_iou > self.best_mean_iou
        if is_best:
            self.best_mean_iou = mean_iou
            self.trn_mean_iou = trn_metrics[1]
            self.best_iteration = self.iteration
            self.val_loss = val_loss
            self.trn_loss = trn_loss
        if not self.sample:
            self.create_checkpoint(is_best)
        if self.visualizer_enabled:
            self.visualize()
        self.model.train()

    def calculate_loss(self, inp, target):
        if self.divide_by_255:
            inp /= 255
        inputs = Variable(
            torch.from_numpy(inp).float().cuda())[None, :]
        targets = Variable(
            torch.from_numpy(target).long().cuda())[None, :]
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        if np.isnan(float(loss.data[0])):
            self.trn_loss = -1
            self.val_loss = -1
            self.update_trials_table()
            print("Loss is NaN")
            raise ValueError
        return loss, outputs

    def _inspect_data(self, inp, target):
        sleep(0.5)
        fig = plt.figure(3)
        images = np.concatenate([inp, target[None, :]])
        n = len(images)
        plots([images[i] for i in range(n)], cols=n, fig=fig)
        now = datetime.now(self.timezone)
        name = "data_aug_%s.png" % now.strftime(self.timeformat)
        fig.savefig(osp.join(self.log_dir, name))
        fig.clear()
        plt.figure(self.fig_id)

    def augment_data(self, index):
        inp = self.inputs[index]
        target = self.targets[index]
        # self._inspect_data(inp, target)
        # TODO: DRY the pattern below
        if self.mirror:
            if np.random.choice([True, False]):
                inp, target = mirror(inp, target)
        if self.random_crop:
            if np.random.choice([True, False]):
                inp, target = random_crop(inp, target, self.random_crop)
        # self._inspect_data(inp, target)
        return inp, target

    def train_epoch(self):
        self.model.train()
        np.random.shuffle(self.indices_trn)
        data_indices = tqdm.tqdm(enumerate(self.indices_trn), total=self.n_trn,
                                 desc='Train (epoch %d)' % self.epoch,
                                 ncols=self.ncols, leave=False)
        for i, index in data_indices:
            iteration = i + (self.epoch * self.n_trn)
            in_sync = (iteration == self.iteration + 1)
            if not in_sync:
                continue  # Speed through indices to sync tqdm
            self.iteration = iteration
            if iteration % self.evaluation_interval == 0:
                self.evaluate()
            inp, target = self.augment_data(index)
            loss, _ = self.calculate_loss(inp, target)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def train(self):
        epoch_range = tqdm.trange(self.epoch, self.epochs, desc='Train',
                                  ncols=self.ncols, leave=True)
        for epoch in epoch_range:
            self.epoch = epoch
            self.train_epoch()
        self.update_trials_table()
