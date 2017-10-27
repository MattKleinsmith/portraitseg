from os import listdir
from os.path import isfile, join
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np


def get_fnames(d, random=False):
    fnames = [d + f for f in listdir(d) if isfile(join(d, f))]
    print("Number of files found in %s: %s" % (d, len(fnames)))
    if random: shuffle(fnames)
    return fnames

def get_flickr_id(portrait_fname):
    """
    Input (string): '../data/portraits/flickr/cropped/portraits/00074.jpg'
    Output (int): 74
    """
    return int(portrait_fname.split('/')[-1].split('.')[0])

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
    assert len(portraits) == len(masks) == 4
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.tight_layout()
    for i, ax in enumerate(axes.flat):
        if i < 4:
            ax.imshow(portraits[i, :, :, :], interpolation='spline16')
        else:
            ax.imshow(gray2rgb(masks[i-4, :, :][0] * 255))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.uint8)
    rgb[:, :, 2] =  rgb[:, :, 1] =  rgb[:, :, 0] = gray
    return rgb


def plots(imgs, figsize=(12, 12), rows=None, cols=None,
          interp=None, titles=None, cmap='gray'):
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
    fig = plt.figure(figsize=figsize)
    for i in range(len(imgs)):
        sp = fig.add_subplot(rows, cols, i+1)
        if titles:
            sp.set_title(titles[i], fontsize=18)
        plt.imshow(imgs[i], interpolation=interp[i], cmap=cmap[i])
        plt.axis('off')
        plt.tight_layout()

def denormalizer(portrait, nb_channels=3, std=0.5, mean=0.5):
    assert isinstance(portrait, np.ndarray)
    assert portrait.shape[-1] == nb_channels
    return portrait * std + mean

def show_input_output_target(portraits, val_outputs, masks, denormalizer):
    assert isinstance(portraits, torch.autograd.variable.Variable)
    assert isinstance(val_outputs, list)
    assert isinstance(masks, torch.autograd.variable.Variable)

    images = []
    titles = []
    cmaps = []

    portrait = portraits.data[0].cpu().numpy().transpose((1, 2, 0))
    portrait = denormalizer(portrait)
    images.append(portrait)
    titles.append("input")
    cmaps.append(None)

    for epoch, outputs in enumerate(val_outputs):
        output = outputs.data[0][0].cpu().numpy()
        images.append(output)
        titles.append("output (epoch %d)" % (epoch+1))
        cmaps.append("gray")

    mask = masks.data[0][0].cpu().numpy()
    images.append(mask)
    titles.append("target")
    cmaps.append("gray")

    cols = 4
    rows = int(np.ceil(len(images) / cols))

    plots(images, titles=titles, cmap=cmaps, rows=rows, cols=cols)
    plt.show()
