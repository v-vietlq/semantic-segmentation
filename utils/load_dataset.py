import os
import numpy as np
import cv2


import torchvision.transforms as T

from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, dataroot, annpath, trans_func=None, mode='train') -> None:
        super(BaseDataset, self).__init__()
        self.mode = mode 
        assert mode in ('train', 'val', 'test')
        self.trans_func = trans_func
        self.lb_map=None
        self.toTensor = T.ToTensor()        
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
        
        img, gt = cv2.imread(imgpth)[:, :, ::-1], cv2.imread(lbpth, 0)
        # print(img)
        
        if not self.lb_map is None:
            gt = self.lb_map[gt]
            
        if not self.trans_func is None:
            img = self.trans_func(img)
            gt = self.trans_func(gt)
        print(np.unique(gt))
        img = self.toTensor(img)
        gt = self.toTensor(gt)
        return img.detach(), gt.unsqueeze(0).detach()
    
    

    
    
            
        