import torch
import torch.nn as nn
import torchvision
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision.models import resnet

class conv_block_nested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch) -> None:
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3,padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        out = self.activation(x)
        return out
    
class upsample_layer(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super(upsample_layer, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_ch)
        
    def forward(self, x):
        x = self.up(x)
        x = self.conv1(x)
        x = self.bn1(x)
        out = self.activation(x)
        return out
        
        
    
class RoadSeg(nn.Module):
    def __init__(self, num_labels) -> None:
        super(RoadSeg, self).__init__()
        
        resnet_raw_model = torchvision.models.resnet18(pretrained=True)
        filters = [64,64,128,256,512]
        
        #encoder rgb
        
        self.encoder_rgb_conv1 = resnet_raw_model.conv1
        self.encoder_rgb_bn1 = resnet_raw_model.bn1
        self.encoder_rgb_relu = resnet_raw_model.relu
        self.encoder_rgb_maxpool = resnet_raw_model.maxpool
        self.encoder_rgb_layer1 = resnet_raw_model.layer1
        self.encoder_rgb_layer2 = resnet_raw_model.layer2
        self.encoder_rgb_layer3 = resnet_raw_model.layer3
        self.encoder_rgb_layer4 = resnet_raw_model.layer4
        
        #decoder
        
        self.conv1_1 = conv_block_nested(filters[0]*2, filters[0], filters[0])
        self.conv2_1 = conv_block_nested(filters[1]*2, filters[1], filters[1])
        self.conv3_1 = conv_block_nested(filters[2]*2, filters[2], filters[2])
        self.conv4_1 = conv_block_nested(filters[3]*2, filters[3], filters[3])
        
        
        self.conv1_2 = conv_block_nested(filters[0]*3, filters[0], filters[0]) 
        self.conv2_2 = conv_block_nested(filters[1]*3, filters[1], filters[1])
        self.conv3_2 = conv_block_nested(filters[2]*3, filters[2], filters[2])
        
        
        self.conv1_3 = conv_block_nested(filters[0]* 4, filters[0], filters[0])
        self.conv2_3 = conv_block_nested(filters[1]* 4, filters[1], filters[1])
        
        self.conv1_4 = conv_block_nested(filters[0]*5, filters[0], filters[0])
        
        self.up2_0 = upsample_layer(filters[1], filters[0])
        self.up2_1 = upsample_layer(filters[1], filters[0])
        self.up2_2 = upsample_layer(filters[1], filters[0])
        self.up2_3 = upsample_layer(filters[1], filters[0])
        
        self.up3_0 = upsample_layer(filters[2], filters[1])
        self.up3_1 = upsample_layer(filters[2], filters[1])
        self.up3_2 = upsample_layer(filters[2], filters[1])
        
        self.up4_0 = upsample_layer(filters[3], filters[2])
        self.up4_1 = upsample_layer(filters[3], filters[2])
        
        self.up5_0 = upsample_layer(filters[4], filters[3])
        
        self.final = upsample_layer(filters[0], num_labels)
        
        
        self.need_initialization = [self.conv1_1, self.conv2_1, self.conv3_1, self.conv4_1, self.conv1_2,
                                    self.conv2_2, self.conv3_2, self.conv1_3, self.conv2_3, self.conv1_4,
                                    self.up2_0, self.up2_1, self.up2_2, self.up2_3, self.up3_0, self.up3_1,
                                    self.up3_2, self.up4_0, self.up4_1, self.up5_0, self.final]
        
    def forward(self, rgb):
        rgb = self.encoder_rgb_conv1(rgb)
        rgb = self.encoder_rgb_bn1(rgb)
        rgb = self.encoder_rgb_relu(rgb)     
        x1_0 = rgb
        
        rgb = self.encoder_rgb_maxpool(rgb)
        rgb = self.encoder_rgb_layer1(rgb)
        x2_0 = rgb
        
        rgb = self.encoder_rgb_layer2(rgb)
        x3_0 = rgb
        
        rgb = self.encoder_rgb_layer3(rgb)    
        x4_0 = rgb
        
        rgb = self.encoder_rgb_layer4(rgb)
        x5_0 = rgb
        
        #decoder
        
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], dim=1))
        x4_1 = self.conv4_1(torch.cat([x4_0, self.up5_0(x5_0)], dim=1))
        
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_1(x2_1)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_1(x3_1)], dim=1))
        x3_2 = self.conv3_2(torch.cat([x3_0, x3_1, self.up4_1(x4_1)], dim=1))
        
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_2(x2_2)], dim=1))
        x2_3 = self.conv2_3(torch.cat([x2_0, x2_1, x2_2, self.up3_2(x3_2)], dim=1))
        
        x1_4 = self.conv1_4(torch.cat([x1_0, x1_1, x1_2, x1_3, self.up2_3(x2_3)], dim=1))
        
        out = self.final(x1_4)
        
        return out
    
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None) -> None:
        super().__init__()
        self.loss = nn.NLLLoss(weight)
        
    def forward(self, outputs, targets):
        outputs = outputs
        return self.loss(F.log_softmax(outputs, dim=1), targets)
        
        

def init_weights(net, gain=0.02):
    net = net
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            nn.init.xavier_normal_(m.weight.data, gain)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
        if classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)
        
        
        
        
        
        