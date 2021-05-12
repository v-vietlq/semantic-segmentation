from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self,in_ch, out_ch, ks, stride=1, padding= 0) -> None:
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, 
            kernel_size=ks, 
            stride=stride, 
            padding=padding,
            bias=True)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)
                    

class UpSample(nn.Module):
    def __init__(self, in_ch, factor = 2) -> None:
        super(UpSample, self).__init__()
        out_ch = in_ch * factor * factor
        self.proj = nn.Conv2d(in_ch, out_ch, 1, 1, 0)
        self.up = nn.PixelShuffle(factor)
        self.init_weight() 
    
    def forward(self, x):
        x = self.proj(x)
        x = self.up(x)
        return x
    
    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.)
        
        
        