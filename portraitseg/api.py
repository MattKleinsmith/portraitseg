#!/usr/bin/env python3

import argparse

import numpy as np
from PIL import Image

import torch
from torch.autograd import Variable

from portraitseg.utils import (set_seed, scoretensor2mask,
    rm_dir_and_ext, mask_image)
from portraitseg.portraitfcn import PortraitFCNPlus
from portraitseg.create_superportraits import get_superportrait
from portraitseg.utils import transform_portrait

def get_mask(args):
    portrait_fpath = args.portrait_filepath
    output_dir = args.output_dir
    extract = args.extract
    arch = args.arch
    gpu = args.arch

    portrait = Image.open(portrait_fpath)
    if arch == "fcn":
        model = PortraitFCN()
        path_to_weights = "portraitseg/portraitfcn_untrained.pth"
        inp = transform_portrait(portrait)
    elif arch == "pfcnp":
        model = PortraitFCNPlus()
        path_to_weights = "portraitseg/portraitfcn__plus_46_7.305E+04.pth"
        inp = get_superportrait(portrait_fpath, points_dir=output_dir)

    model.load_state_dict(torch.load(path_to_weights))
    model.eval()
    inputs = Variable(torch.from_numpy(inp).float())[None, :]
    if gpu:
        model = model.cuda()
        inputs = inputs.cuda()
    outputs = model(inputs)
    scoretensor = outputs[0].data.cpu()
    prediction = scoretensor2mask(scoretensor)
    prediction = Image.fromarray(prediction)

    out_fpath_prefix = output_dir + rm_dir_and_ext(portrait_fpath)
    output_fpath = out_fpath_prefix + "_mask.png"
    prediction.save(output_fpath)

    if extract:
        mask_image(portrait, prediction).save(out_fpath_prefix + "_fg.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("portrait_filepath",
                        help="e.g. path/to/file/portrait.jpg")
    parser.add_argument("-o", "--output_dir", default="./portraitseg/outputs/")
    parser.add_argument("-e", "--extract", dest="extract", default=True)
    parser.add_argument("-ne", "--no-extract", dest='extract', action='store_false')
    parser.add_argument("-arch", default="pfcnp")
    parser.add_argument("-g", "--gpu", default=True)
    args = parser.parse_args()

    get_mask(args)
