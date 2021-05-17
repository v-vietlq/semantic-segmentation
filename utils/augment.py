#!/usr/bin/python
# -*- encoding: utf-8 -*-

import math
import random


import numpy as np
import cv2
import torch
from PIL import Image, ImageFilter
from torchvision import transforms
import torchvision.transforms.functional as F


class RandomResize(object):
    def __init__(self, p=0.5, scale_range=None, scale_values=None, interpolation=Image.BICUBIC):
        assert (scale_range is None) ^ (scale_values is None)
        self.p = p
        self.scale_range = scale_range
        self.scale_values = scale_values
        self.interpolation = interpolation

    def __call__(self, img, lbl):
        if random.random() >= self.p:
            return img, lbl
        if self.scale_range is not None:
            scale = random.random() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        else:
            scale = self.scale_values[random.randint(0, len(self.scale_values))]

        size = tuple(np.round(np.array(img.size[::-1]) * scale).astype(int))
        img = F.resize(img, size, self.interpolation)
        lbl = F.resize(lbl, size, Image.NEAREST)

        return img, lbl
    
        

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5) -> None:
        super(RandomHorizontalFlip, self).__init__()
        self.p = p
        
    def __call__(self, img, lb):
        if random.random() < self.p:
            img = F.hflip(img)
            lb = F.hflip(lb)
        return img, lb
    
class RandomVerticalFlip(object):
    def __init__(self, p =0.5) -> None:
        super(RandomVerticalFlip, self).__init__()
        self.p = p
    
    def __call__(self, img, lb):
        if random.random() < self.p:
            img = F.vflip(img)
            lb = F.vflip(lb)
            
        return img, lb
    
    
class RandomGaussianBlur(object):
    def __init__(self, p=0.5, r=0.5):
        super(RandomGaussianBlur, self).__init__()
        self.p = p 
        self.r = r
        self.filter = ImageFilter.GaussianBlur(radius= r)
    
    def __call__(self, img, label):
        if random.random() < self.p:
            img = img.filter(self.filter)
        
        return img, label
        
    
class RandomRotate(object):
    def __init__(self, degree) -> None:
        super(RandomRotate, self).__init__()
        self.degree = degree
    
    def __call__(self, img, lb):
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        lb = lb.rotate(rotate_degree, Image.NEAREST)
        
        return img, lb
    
class Pyramids(object):
    def __init__(self, levels = 1) -> None:
        assert levels >=1
        super().__init__()
        self.levels = levels
    
    def __call__(self,img):
        img_pyd = [img]
        for i in range(self.levels -1):
            img_pyd.append(Image.fromarray(cv2.pyrDown(np.array(img_pyd[-1]))))
        
        return img_pyd
    
class UpDownPyramids(Pyramids):
    def __init__(self, levels =1 , up_levels= 0) -> None:
        super().__init__(levels=levels)
        self.up_levels = up_levels
        
    def __call__(self, img):
        img_pyd =  super(UpDownPyramids, self).__call__(img)
        for i in range(self.up_levels):
            img_pyd.append(Image.fromarray(cv2.pyrUp(np.array(img_pyd[0]))))
            
        return img_pyd
    

class RandomCrop(transforms.RandomCrop):

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, lbl_fill=None, padding_mode='constant'):
        super(RandomCrop, self).__init__(size, padding, pad_if_needed, fill, padding_mode)
        self.lbl_fill = fill if lbl_fill is None else lbl_fill

    def __call__(self, img, lbl):
        assert img.size == lbl.size, 'size of img and lbl should be the same. %s, %s' % (img.size, lbl.size)

        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            lbl = F.pad(lbl, self.padding, self.lbl_fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
            lbl = F.pad(lbl, (self.size[1] - lbl.size[0], 0), self.lbl_fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)
            lbl = F.pad(lbl, (0, self.size[0] - lbl.size[1]), self.lbl_fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), F.crop(lbl, i, j, h, w)

        
class Compose(object):
    def __init__(self, transforms) -> None:
        super().__init__()
        self.transforms = transforms
    
    def __call__(self, img, label):
        for t in self.transforms:
            img, label = t(img, label)
        return img, label
    
class Resize(object):
    def __init__(self, size, interpolation =Image.BILINEAR) -> None:
        super(Resize, self).__init__()
        self.size = size
        self.interpolation = interpolation
        
    def __call__(self,img, label):
        img = F.resize(img, self.size, self.interpolation)
        label = F.resize(label, self.size, self.interpolation)
        return img, label
        
    
if __name__ == '__main__':
    pass