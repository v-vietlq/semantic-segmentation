import matplotlib.pyplot as plt
import numpy as np
import torch


def get_cityscapes_labels():
    return np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])
    
    
def decode_segmap(label_masks, dataset= 'cityscapes'):
    if dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
        
    
    r = label_masks.copy()
    g = label_masks.copy()
    b = label_masks.copy()
    
    for ll in range(0, n_classes):
        r[label_masks == ll] = label_colours[ll, 0]
        g[label_masks == ll] = label_colours[ll, 1]
        b[label_masks == ll] = label_colours[ll, 2]
        
    rgb = np.zeros((label_masks.shape[0], label_masks.shape[1], 3))
    
    rgb[:,:,0] = r/255.0
    rgb[:,:,1] = g/255.0
    rgb[:,:,2] = b/255.0
    
    
    return rgb
        
    
    