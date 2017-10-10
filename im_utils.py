import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plots(imgs, figsize=(12, 12), rows=1, cols=1,
          interp=None, titles=None, cmap='gray'):
    if not isinstance(imgs, list):
        imgs = [imgs]
    if not isinstance(cmap, list):
        if imgs[0].ndim == 2:
            cmap = 'gray'
        cmap = [cmap] * len(imgs)
    if not isinstance(interp, list):
        interp = [interp] * len(imgs)
    fig = plt.figure(figsize=figsize)
    for i in range(len(imgs)):
        sp = fig.add_subplot(rows, cols, i+1)
        if titles:
            sp.set_title(titles[i], fontsize=18)
        plt.imshow(imgs[i], interpolation=interp[i], cmap=cmap[i])
        plt.axis('off')


def plot(im, f=6, r=1, c=1, t=None):
    fs = f if isinstance(f, tuple) else (f, f)
    plots(im, figsize=fs, rows=r, cols=c, titles=t)
    
    
def loadim(path):
    '''Returns np.array. Loaded with PIL.'''
    return np.array(Image.open(path))