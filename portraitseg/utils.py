from os import listdir
from os.path import isfile, join
from random import shuffle

import matplotlib.pyplot as plt


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
