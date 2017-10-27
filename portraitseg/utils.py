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
