import torch
import torch.nn as nn
from torch.nn import modules




class ConvBNRELU(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, stride = 1, padding= 1) -> None:
        super(ConvBNRELU, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels= out_ch, kernel_size= ks,stride=stride, padding= padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class StemBlock(nn.Module):
    def __init__(self) -> None:
        super(StemBlock, self).__init__()
        self.conv = ConvBNRELU(3,16,3,stride=2)
        self.left = nn.Sequential(
            ConvBNRELU(16, 8, 1, stride=1, padding= 0),
            ConvBNRELU(8, 16, 3, stride=2)
        )
        
        self.right = nn.MaxPool2d(
            kernel_size=3, stride=2, padding= 1, ceil_mode=False
        )
        
        self.fuse = ConvBNRELU(32, 16, 3, stride=1)
        
    def forward(self,x):
        x = self.conv(x)
        x_left = self.left(x)
        x_right = self.right(x)
        x = torch.cat([x_left, x_right], dim=1)
        x = self.fuse(x)
        return x
    
class CEBBlock(nn.Module):
    def __init__(self) -> None:
        super(CEBBlock, self).__init__()
        self.bn = nn.BatchNorm2d(128)
        # self.gp = nn.AvgPool2d(
        #     kernel_size= 3, stride=2, padding= 1, ceil_mode= False
        # )
        self.conv = nn.Conv2d(128, 128, 1, stride= 1, padding= 0)
        
        self.fuse = ConvBNRELU(128,128,3,stride=1)
        
    def forward(self,x):
        feat = torch.mean(x, dim=(2,3), keepdim= True)
        feat = self.bn(feat)
        feat = self.conv(feat)
        feat = feat + x 
        feat = self.fuse(feat)
        return feat
    
                
        
class GELayerS1(nn.Module):
    def __init__(self, in_ch, out_ch, exp_ratio=6) -> None:
        super(GELayerS1, self).__init__()
        
        mid_ch = in_ch * exp_ratio
        
        self.conv1 = ConvBNRELU(in_ch, in_ch, 3, stride=1)
        
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, stride=1,
                      padding= 1, groups=in_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace= True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, kernel_size= 1, stride= 1,
                      padding= 0, bias= False),
            nn.BatchNorm2d(out_ch),
        )
        
        self.relu = nn.ReLU(inplace= True)
        
    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)
        feat = feat + x
        feat = self.relu(feat)
        return feat
    
class GELayerS2(nn.Module):
    def __init__(self, in_ch, out_ch, exp_ratio= 6) -> None:
        super(GELayerS2, self).__init__()
        
        mid_ch = in_ch * exp_ratio
        
        self.conv1 = ConvBNRELU(in_ch, in_ch, 3, stride= 1)
        
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size= 3, stride= 2, padding= 1,
                      groups= in_ch, bias= False),
            nn.BatchNorm2d(mid_ch),
        )
        
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size= 3, stride=1, padding=1,
                      groups= mid_ch, bias= False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, kernel_size= 1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(out_ch),
            
        )
        
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size= 3,
                      stride= 2, padding= 1, groups= in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, kernel_size= 1, stride= 1, padding= 0, bias= False),
            nn.BatchNorm2d(out_ch)
        )
        
        self.relu = nn.ReLU(inplace= True)
        
    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)
        
        shortcut = self.shortcut(x)
        
        feat = feat + shortcut
        
        feat = self.relu(feat)
        
        return feat


class SegmentBranch(nn.Module):
    def __init__(self) -> None:
        super(SegmentBranch, self).__init__()
        
        self.S1S2 = StemBlock()
        
        self.S3 = nn.Sequential(
            GELayerS2(16, 32),
            GELayerS1(32, 32)
        )
        
        self.S4 = nn.Sequential(
            GELayerS2(32, 64),
            GELayerS1(64,64)
        )
        
        self.S5 = nn.Sequential(
            GELayerS2(64, 128),
            GELayerS1(128,128),
            GELayerS1(128, 128), 
            GELayerS1(128, 128)
        )
        
        self.S5_5 = CEBBlock()
        
    def forward(self, x):
        feat2 = self.S1S2(x)
        feat3 = self.S3(feat2)
        feat4 = self.S4(feat3)
        feat5_4 = self.S5(feat4)
        feat5_5 = self.S5_5(feat5_4)
        return feat2, feat3, feat4, feat5_4, feat5_5
        
        


class DetailBranch(nn.Module):
    def __init__(self) -> None:
        super(DetailBranch, self).__init__()
        
        self.S1 = nn.Sequential(
            ConvBNRELU(3,64,3,stride=2),
            ConvBNRELU(64,64,3,stride=1)
        )
        
        self.S2 = nn.Sequential(
            ConvBNRELU(64,64,3, stride=2),
            ConvBNRELU(64,64,3, stride=1),
            ConvBNRELU(64,64,3, stride=1)
        )
        
        self.S3 = nn.Sequential(
            ConvBNRELU(64, 128,3, stride=2),
            ConvBNRELU(128, 128, 3, stride=1),
            ConvBNRELU(128, 128, 3, stride=1)
        )
        
    def forward(self, x):
        x = self.S1(x)
        x = self.S2(x)
        x = self.S3(x)
        return x
    
   
class BGALayer(nn.Module):
    def __init__(self) -> None:
        super(BGALayer, self).__init__()
        
        #Detail branch
        self.left1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride= 1, padding=1,
                      groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size= 1, stride= 1, padding=0,
                      bias=False)
        )
        
        self.left2 = nn.Sequential(
            nn.Conv2d(128, 128,kernel_size= 3, stride= 2,
                      padding= 1, bias= False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=3,stride=2, padding=1, ceil_mode=False)       
        )
        
        self.right1 = nn.Sequential(
            nn.Conv2d(128,128,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        
        self.right2 = nn.Sequential(
            nn.Conv2d(128,128, kernel_size=3, stride= 1, padding=1,
                      groups= 128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size= 1, stride= 1, padding=0, bias= False)           
        )
        
        self.up1 = nn.Upsample(scale_factor= 4)
        
        self.up2 = nn.Upsample(scale_factor= 4)
        
        self.conv = nn.Sequential(
            nn.Conv2d(128,128, kernel_size=3, stride= 1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x_d, x_s):
        left1 = self.left1(x_d)
        
        left2 = self.left2(x_d)
        
        right1 = self.right1(x_s)
        
        right2 = self.right2(x_s)
        
        
        right1 = self.up1(right1)
        
        # element-wise lef1 vs right1
        left = left1*torch.sigmoid(right1)
        
        # element-wise left2 vs right2
        right = left2* torch.sigmoid(right2)
        
        right = self.up2(right)
        
        out = self.conv(left + right)
        
        return out
        
class SegmentHead(nn.Module):
    def __init__(self, in_ch, mid_ch, n_class, up_factor= 8, aux=True) -> None:
        super(SegmentHead, self).__init__() 
        self.conv = ConvBNRELU(in_ch, mid_ch, 3, stride= 1)
        self.drop = nn.Dropout(0.1)
        self.up_factor = up_factor
        
        out_ch = n_class * up_factor * up_factor
        
        if aux:
            self.conv_out = nn.Sequential(
                ConvBNRELU(mid_ch, up_factor * up_factor, 3, stride= 1),
                nn.Conv2d(up_factor* up_factor, out_ch,1,1,0),
                nn.PixelShuffle(up_factor)
            )
        else:
            self.conv_out = nn.Sequential(
                nn.Conv2d(mid_ch, out_ch, 1,1,0),
                nn.PixelShuffle(up_factor)
            )
    
    def forward(self, x):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        
        return feat
    
    
class BiSeNetV2(nn.Module):
    def __init__(self, n_classes, out_aux=True) -> None:
        super(BiSeNetV2, self).__init__()
        self.detail = DetailBranch()
        self.segment = SegmentBranch()
        self.bga = BGALayer()
        self.out_aux = out_aux
        
        self.head = SegmentHead(128, 1024, n_classes, up_factor= 8, aux=False)
        
        if self.out_aux:
            self.aux2 = SegmentHead(16, 128, n_classes, up_factor= 4)
            self.aux3 = SegmentHead(32, 128, n_classes, up_factor=8)
            self.aux4 = SegmentHead(64,128, n_classes, up_factor= 16)
            self.aux5_4 = SegmentHead(128, 128, n_classes, up_factor=32)
            
        
    def forward(self, x):
        
        feat_d = self.detail(x)
        feat2, feat3, feat4, feat5_4, feat_s = self.segment(x)
        feat_head = self.bga(feat_d, feat_s)
        
        logits = self.head(feat_head)
        
        if self.out_aux:
            logits_aux2 = self.aux2(feat2)
            logits_aux3 = self.aux3(feat3)
            logits_aux4 = self.aux4(feat4)
            logits_aux5_4 = self.aux5_4(feat5_4)
            return logits, logits_aux2, logits_aux3, logits_aux4, logits_aux5_4
        
        pred = logits.argmax(dim=1)
        
        return pred
    
if __name__ == "__main__":
    x = torch.randn(4,3, 1024, 2048)
    model = BiSeNetV2(n_classes=19)
    outs = model(x)
    
    for out in outs:
        print(out.size())
        



        
        
        