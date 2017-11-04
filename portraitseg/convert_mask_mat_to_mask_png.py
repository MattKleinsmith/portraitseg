
# coding: utf-8

# In[1]:

from scipy.io import loadmat                                                                         
from scipy.misc import imsave                                                                        
                                                                                                     
from utils import get_fnames                                                                         
                                                                                                     
                                                                                                     
DATA_DIR = '../../data/portraits/flickr/cropped/'                                                       
TARGET_DIR = DATA_DIR + 'masks/targets/'                                                             
                                                                                                     
mask_fnames = get_fnames(TARGET_DIR + 'mat_files/')                                        
for fname in mask_fnames:                                                                            
    name = fname.split('/')[-1].split('.')[0][:5]                                                    
    out_path = TARGET_DIR + name + ".png"                                                            
    mask = loadmat(fname)['mask'] * 255                                                              
    imsave(out_path, mask)                                                                           

