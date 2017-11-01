#!/usr/bin/env python3

import mkl
nproc = mkl.get_max_threads()  # e.g. 12
mkl.set_num_threads(nproc)
#print("nproc: %d" % nproc)

import subprocess
from time import sleep

import bcolz
import numpy as np
from PIL import Image
from skimage import transform

from portraitseg.utils import (get_fnames,
    plots, rm_dir_and_ext, transform_portrait)


def facetracker(portrait_path,
                facetracker_script="portraitseg/get_tracker_points.py",
                out_dir="portraitseg/outputs/"):
    cmd = "python2 %s -p %s -o %s" % (facetracker_script,
                                      portrait_path,
                                      out_dir)
    subprocess.call(cmd.split())
    portrait_name = rm_dir_and_ext(portrait_path)
    tracker_points = np.load(out_dir + portrait_name + ".npy")
    return tracker_points


def get_tracker_points(portrait_path, points_dir):
    portrait_name = rm_dir_and_ext(portrait_path)
    points_path = points_dir + portrait_name + ".npy"
    try:
        tracker_points = np.load(points_path)
    except FileNotFoundError:
        tracker_points = facetracker(portrait_path, out_dir=points_dir)
    return tracker_points


def get_config(w, h, crop_dir="../data/portraits/flickr/cropped/"):
    ref_points_path = crop_dir + "tracker_points_of_canonical_pose.npy"
    mean_mask_path = crop_dir + "mean_mask.png"
    ref_points = np.load(ref_points_path)
    mean_mask = Image.open(mean_mask_path)
    nb_channels = 6
    im_width, im_height = w, h
    padding = 600
    grid_width, grid_height = 1800, 2000
    h1, h2 = padding, padding + im_height
    w1, w2 = padding, padding + im_width
    xx_grid, yy_grid = np.meshgrid(
        np.arange(1, grid_width+1, dtype=np.int64),
        np.arange(1, grid_height+1, dtype=np.int64))
    grid_shape = xx_grid.shape
    ref_pos = np.floor(np.mean(ref_points, 0))
    xx_grid = (xx_grid - padding - ref_pos[0]) * 1.0 / padding
    yy_grid = (yy_grid - padding - ref_pos[1]) * 1.0 / padding
    mean_mask_np = np.array(mean_mask, dtype=np.float64)
    mean_mask_grid = np.pad(mean_mask_np, (padding, padding), "minimum")

    config = dict(
        nb_channels=nb_channels,
        im_width=im_width,
        im_height=im_height,
        padding=padding,
        ref_points=ref_points,
        xx_grid=xx_grid,
        grid_shape=grid_shape,
        yy_grid=yy_grid,
        mean_mask_grid=mean_mask_grid,
        h1=h1,
        h2=h2,
        w1=w1,
        w2=w2)
    return config


def get_position_and_shape_channels(dest_points, config):
    """
    Create a transform to project objects from the standard space to
    the space of the given portrait.
    Projects these objects:
        - A grid of horizontal distances, with respect to the center of
          the face
            - The motivating intuition / assumption / bias:
                - The farther a pixel is from the center of the face,
                  the less likely the pixel is to be a part of the
                  portrait.
        - A grid of vertical distances, with respect to the center of
          the face
        - A rough, general mask, computed from many observed masks, used
          as a rough estimate of what the final mask should like
            - i.e. The final mask should consist of a head-like shape
              that's connected to a torso-like shape.
            - See mean_mask.png to see what's being projected
    Synonyms in this context: "transform", "project", "align"
    """
    # Get the settings
    padding = config['padding']
    ref_points = config['ref_points']
    grid_shape = config['grid_shape']
    h1, h2 = config['h1'], config['h2']
    w1, w2 = config['w1'], config['w2']

    # Get the standard objects
    xx_grid = config['xx_grid']
    yy_grid = config['yy_grid']
    mean_mask_grid = config['mean_mask_grid']

    # Create the transform and use it to project the standard objects
    #   from the standard space to the space of the given portrait
    tform = transform.estimate_transform('affine',
                                         dest_points + padding,
                                         ref_points + padding)
    projected_xx_grid = transform.warp(xx_grid, tform,
                                      output_shape=grid_shape)
    projected_yy_grid = transform.warp(yy_grid, tform,
                                       output_shape=grid_shape)
    projected_mean_mask_grid = transform.warp(mean_mask_grid, tform,
                                              output_shape=grid_shape)
    # Crop to remove padding
    projected_xx = projected_xx_grid[h1:h2, w1:w2]
    projected_yy = projected_yy_grid[h1:h2, w1:w2]
    projected_mean_mask = projected_mean_mask_grid[h1:h2, w1:w2]
    return projected_xx, projected_yy, projected_mean_mask


def get_superportrait(portrait_path, points_dir="./", config=None):
    portrait = Image.open(portrait_path)
    w, h = portrait.size
    if not config:
        config = get_config(w, h)
    nb_channels = config['nb_channels']
    im_height = config['im_height']
    im_width = config['im_width']
    dest_points = get_tracker_points(portrait_path, points_dir)
    # Create position and shape channels
    xx, yy, mean_mask = get_position_and_shape_channels(dest_points, config)
    # Transform portrait, and add position and shape channels
    portrait = transform_portrait(portrait)
    superportrait = np.zeros((nb_channels, im_height, im_width),
                             dtype=np.float64)
    superportrait[:3] = portrait
    superportrait[3] = xx
    superportrait[4] = yy
    superportrait[5] = mean_mask
    return superportrait


def get_superportraits_of_training_portraits():
    data_dir = "../data/portraits/flickr/"
    crop_dir = data_dir + "cropped/"
    portrait_dir = crop_dir + "portraits/"
    superportraits_bcolz_path = crop_dir + "training_superportraits2.bcolz"
    config = get_ref_config(crop_dir=crop_dir)

    # Get superportraits
    trn_IDs = np.load(data_dir + "trainlist_clean.npy")
    trn_names = ["%05d" % i for i in trn_IDs]
    trn_fpaths = [portrait_dir + name + ".jpg" for name in trn_names]
    superportraits = [get_superportrait(f, config) for f in trn_fpaths]
    superportraits_np = np.array(superportraits)

    # Save superportraits
    n_gb = superportraits_np.nbytes/1024**3
    print("superportraits: %.2f GB in RAM" % n_gb)
    c = bcolz.carray(superportraits_np,
                     rootdir=superportraits_bcolz_path,
                     mode='w')
    c.flush()
    print("Saved to " + superportraits_bcolz_path)

