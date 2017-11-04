import datetime
import os
from os import listdir
import os.path as osp
from random import shuffle
import random
import shlex
import subprocess
import sqlite3
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import psutil
import pytz
import torch
import torch.nn.functional as F
import yaml


HERE = osp.dirname(osp.abspath(__file__))


def create_log(filepath, headers):
    if not osp.exists(filepath):
        with open(filepath, 'w') as f:
            f.write(','.join(headers) + '\n')


def get_RAM():
    return psutil.virtual_memory().used


def git_hash():
    cmd = 'git log -n 1 --pretty="%h"'
    hash = subprocess.check_output(shlex.split(cmd)).strip()
    return hash


# https://github.com/wkentaro/pytorch-fcn/blob/master/examples/voc/train_fcn32s.py
def get_log_dir(config_id, cfg, sample=False):
    for k, v in cfg.copy().items():
        if callable(v) or isinstance(v, type):
            v = v.__name__
            cfg[k] = v
        elif k == "data_aug":
            v = "%02d" % v
        else:
            v = str(v)
        if k == "loss_fn_kwargs":
            break
    # Get current log IDs
    if sample:
        logs_dir = osp.join(HERE, "logs/samples")
    else:
        logs_dir = osp.join(HERE, "logs")
    log_ids = [int(d.split("_")[0]) for d in listdir(logs_dir) if "CFG" in d]
    try:
        log_id = max(log_ids) + 1
    except ValueError:
        log_id = 1
    name = "%05d_CFG-%03d" % (log_id, config_id)
    name += "_GIT-%s" % git_hash().decode("utf-8")
    now = datetime.datetime.now(pytz.timezone("America/Los_Angeles"))
    name += "_%s" % now.strftime("%Y-%m-%d--%H-%M-%S")
    log_dir = osp.join(logs_dir, name)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    with open(osp.join(log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(dict(cfg), f, default_flow_style=False)
    return log_dir


def transform_portrait(img):
    img = np.array(img, dtype=np.uint8)
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float64)
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
    img -= mean_bgr
    img = img.transpose(2, 0, 1)  # HxWxC --> CxHxW
    return img


def split_trn_val(num_train, valid_size=0.2, shuffle=False):
    indices = list(range(num_train))
    if shuffle:
        np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    indices_trn, indices_val = indices[split:], indices[:split]
    return indices_trn, indices_val


def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss


def scoretensor2mask(scoretensor):
    """
    - scoretensor (3D torch tensor) (CxHxW): Each channel contains the scores
        for the corresponding category in the image.
    Returns a numpy array.
    """
    _, labels = scoretensor.max(0)  # Get labels w/ highest scores
    labels_np = labels.numpy().astype(np.uint8)
    mask = labels_np * 255
    return mask


def detransform_portrait(img, mean="voc"):
    """
    - img (torch tensor)
    Returns a numpy array.
    """
    if mean == "voc":
        mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
    else:
        raise ValueError("unknown mean")
    #img = img.numpy().astype(np.float64)
    img = img.transpose((1, 2, 0)) # CxHxW --> HxWxC
    #img *= 255
    img += mean_bgr
    img = img[:, :, ::-1]  # BGR -> RGB
    img = img.astype(np.uint8)
    return img


def detransform_mask(mask):
    #mask = mask.numpy()
    mask = mask.astype(np.uint8)
    mask *= 255
    return mask


def mask_image(img, mask, opacity=1.00, bg=False):
    """
        - img (PIL)
        - mask (PIL)
        - opacity (float) (default: 1.00)
    Returns a PIL image.
    """
    blank = Image.new('RGB', img.size, color=0)
    if bg:
        masked_image = Image.composite(blank, img, mask)
    else:
        masked_image = Image.composite(img, blank, mask)
    if opacity < 1:
        masked_image = Image.blend(img, masked_image, opacity)
    return masked_image


def show_portrait_pred_mask(portrait, preds, mask, evaluation_interval,
                            opacity=None, bg=False, fig=None):
    """
    Args:
        - portrait (torch tensor)
        - preds (list of np.ndarray): list of mask predictions
        - mask (torch tensor)
    A visualization function.
    Returns nothing.
    """
    # Gather images
    images = []
    titles = []
    cmaps = []

    #### Prepare portrait
    portrait_pil = Image.fromarray(portrait)
    images.append(portrait)
    titles.append("input")
    cmaps.append(None)

    #### Prepare predictions
    for i, pred in enumerate(preds):
        pred_pil = Image.fromarray(pred)
        if opacity:
            pred_pil = mask_image(portrait_pil, pred_pil, opacity, bg)
        images.append(pred_pil)
        titles.append("iter. %d" % (i * evaluation_interval))
        cmaps.append("gray")

    #### Prepare target mask
    if opacity:
        mask_pil = Image.fromarray(mask)
        mask = mask_image(portrait_pil, mask_pil, opacity, bg)
    images.append(mask)
    titles.append("target")
    cmaps.append("gray")

    # Show images
    cols = 5
    rows = int(np.ceil(len(images) / cols))
    w = 12
    h = rows * (w / cols + 1)
    figsize = (w, h)  # width x height
    plots(images, titles=titles, cmap=cmaps, rows=rows, cols=cols,
          figsize=figsize, fig=fig)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_fnames(d, random=False):
    fnames = [d + f for f in listdir(d) if osp.isfile(osp.join(d, f))]
    print("Number of files found in %s: %s" % (d, len(fnames)))
    if random: shuffle(fnames)
    return fnames

def rm_dir_and_ext(filepath):
    return filepath.split('/')[-1].split('.')[-2]

def get_flickr_id(portrait_fname):
    """
    Input (string): '../data/portraits/flickr/cropped/portraits/00074.jpg'
    Output (int): 74
    """
    return int(rm_dir_and_ext(portrait_fname))


def get_lines(fname):
    '''Read lines, strip, and split.'''
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip().split() for x in content]
    return content


def hist(data, figsize=(6, 3)):
    fig = plt.figure(figsize=figsize)
    plt.hist(data)
    plt.show()


def plot_portraits_and_masks(portraits, masks):
    assert len(portraits) == len(masks)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.tight_layout()
    for i, ax in enumerate(axes.flat):
        if i < 4:
            ax.imshow(portraits[i], interpolation="spline16")
        else:
            mask = gray2rgb(masks[i-4])
            ax.imshow(mask)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.uint8)
    rgb[:, :, 2] =  rgb[:, :, 1] =  rgb[:, :, 0] = gray
    return rgb


def plots(imgs, figsize=(12, 12), rows=None, cols=None,
          interp=None, titles=None, cmap='gray',
          fig=None):
    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [np.array(img) for img in imgs]
    if not isinstance(cmap, list):
        if imgs[0].ndim == 2:
            cmap = 'gray'
        cmap = [cmap] * len(imgs)
    if not isinstance(interp, list):
        interp = [interp] * len(imgs)
    n = len(imgs)
    if not rows and not cols:
        cols = n
        rows = 1
    elif not rows:
        rows = cols
    elif not cols:
        cols = rows
    if not fig:
        fig = plt.figure(figsize=figsize)
    fontsize = 13 if cols == 5 else 16
    fig.set_figheight(figsize[1], forward=True)
    fig.clear()
    for i in range(len(imgs)):
        sp = fig.add_subplot(rows, cols, i+1)
        if titles:
            sp.set_title(titles[i], fontsize=fontsize)
        plt.imshow(imgs[i], interpolation=interp[i], cmap=cmap[i])
        plt.axis('off')
        plt.subplots_adjust(0, 0, 1, 1, .1, 0)
        #plt.tight_layout()
    if fig: fig.canvas.draw()
