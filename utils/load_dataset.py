import os
from PIL import Image
import numpy as np
import cv2
import torch

from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, dataroot, annpath, trans_func=None, mode='train') -> None:
        super(BaseDataset, self).__init__()
        self.mode = mode 
        assert mode in ('train', 'val', 'test')
        self.trans_func = trans_func
        self.lb_map=None      
        with open(annpath, 'r') as f:
            pairs = f.read().splitlines()
        
        self.img_paths, self.lb_paths = [], []
        for pair in pairs:
            imgpth, lbpth = pair.split(',')
            self.img_paths.append(os.path.join(dataroot, imgpth).replace('\\','/'))
            self.lb_paths.append(os.path.join(dataroot, lbpth).replace('\\','/'))
            
    def __len__(self) -> int:
        return len(self.img_paths)
               
    
    def __getitem__(self, idx: int):
        
        imgpth, lbpth = self.img_paths[idx], self.lb_paths[idx]
        
        img = Image.open(imgpth).convert('RGB')
        gt = np.array(Image.open(lbpth), dtype= np.uint8)
        
        if not self.lb_map is None:
            gt = self.lb_map[gt]
            
        gt = Image.fromarray(gt)
        
        sample = {
            'image' : img,
            'label' : gt
        }
        
        if not self.trans_func is None:
            sample = self.trans_func(sample)
            
        return sample['image'], sample['label']
            
            
            
        
    
         
        