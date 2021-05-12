import torch
from torch.nn.modules import activation
from torchvision.models import mobilenet_v2
import torch.nn as nn
import torch.nn.functional as F




class conv_block_nested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, dropprob) -> None:
        super(conv_block_nested, self).__init__()
        
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size= 1, padding= 0, bias= True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        
        self.activation = nn.ReLU6(inplace=True)
        
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size= 3, padding=1, groups= mid_ch)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        
        self.conv3 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, padding=0 , bias=True)
        self.bn3 = nn.BatchNorm2d(out_ch)
        
        self.dropout = nn.Dropout2d(dropprob)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x= self.activation(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        out = self.activation(x)
        
        if self.dropout.p != 0 :
            out = self.dropout(out)
        
        return F.relu(out)
    
class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated) -> None:
        super(non_bottleneck_1d, self).__init__()
        self.conv3x1_1 = nn.Conv2d(chann, chann, (3,1), stride=1, padding=(1,0), bias=True)
        
        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3),stride=1, padding=(0,1), bias= True)
        
        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)
        
        self.conv3x1_2 = nn.Conv2d(chann, chann, (3,1), stride=1, padding=(1*dilated,0), bias=True, dilation=(dilated,1))
        
        self.conv1x3_2 = nn.Conv2d(chann, chann,(1,3), stride= 1, padding=(0, 1*dilated), bias=True, dilation=(1, dilated))
        
        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)
        
        self.dropout = nn.Dropout2d(dropprob)
        
    
    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)
        
        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)
        
        if self.dropout.p !=0:
            output = self.dropout(output)
            
        return F.relu(output +input)
            
class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput) -> None:
        super(UpsamplerBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride= 2, padding=1, output_padding=1, bias=True)
        
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        
    
    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)
    

class VietNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super(VietNet, self).__init__()
              
        mobilenetv2_raw = mobilenet_v2(pretrained=True)
        
        
        
        self.feature_extaction = mobilenetv2_raw.features[:13]
        
        self.bottlenet1 = conv_block_nested(96, 384, 160, 0.2)
        self.bottlenet2 = conv_block_nested(160, 640, 160, 0.2)
        

        
        self.layers = nn.ModuleList()
        
        self.conv = nn.Conv2d(160, 256 , kernel_size= 1, padding= 0, bias=True)
        self.bn = nn.BatchNorm2d(256)
        self.activation = nn.ReLU6(inplace=True)
        
        self.layers.append(self.conv)
        self.layers.append(self.bn)
        self.layers.append(self.activation)
        
        self.layers.append(non_bottleneck_1d(256,0.2,2))
        self.layers.append(non_bottleneck_1d(256, 0.2, 4))
        self.layers.append(non_bottleneck_1d(256,0.2, 8))
        self.layers.append(non_bottleneck_1d(256, 0.2, 16))
        
        
        
        self.layers.append(UpsamplerBlock(256, 128))
        self.layers.append(non_bottleneck_1d(128,0,1))
        self.layers.append(non_bottleneck_1d(128,0,1))
        
        
        
        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64,0,1))
        self.layers.append(non_bottleneck_1d(64,0,1))
        
        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16,0,1))
        self.layers.append(non_bottleneck_1d(16,0,1))
        
        self.output_conv = nn.ConvTranspose2d(16, num_classes,kernel_size=2, stride=2,padding= 0, output_padding=0, bias=True)
        
        
    def forward(self, x):
        
        x = self.feature_extaction(x)
        x = self.bottlenet1(x)
        x_1 = self.bottlenet2(x)

        x = F.relu6(x + x_1, inplace= True)
            
        for layer in self.layers:
            x = layer(x)
        
        out = self.output_conv(x)
        
        return out
    
        
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None) -> None:
        super().__init__()
        self.loss = nn.NLLLoss(weight)
        
    def forward(self, outputs, targets):
        outputs = outputs
        return self.loss(F.log_softmax(outputs, dim=1), targets)



