from tqdm import tqdm
import numpy as np 
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from utils.cityscapes import get_data_loader
from segmentation_models_pytorch.utils.metrics import IoU, Fscore, Recall, Precision, Accuracy

torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.deterministic = True

class MscEvalV0(object):
    def __init__(self, scales=(0.5, ), flip=False, ignore_lb =255) -> None:
        super().__init__()
        self.scales =scales
        self.flip = flip
        self.ignore_lb = ignore_lb
        
    def __call__(self,net,dl, n_classes):
        hist = torch.zeros(n_classes,n_classes).cuda().detach()
        if dist.is_initialized() and dist.get_rank() !=0:
            diter = enumerate(dl)
        else:
            diter = enumerate(tqdm(dl))
        for _, (imgs, label) in diter:
            N,_, H, W = label.shape
            label = label.squeeze(1).cuda()
            size = label.size()[-2:]  #get original size label
            probs = torch.zeros((N, n_classes, H, W), dtype= torch.float32).cuda().detach()
            
            for scale in self.scales:
                sH, sW = int(scale* H), int (scale *W)
                im_sc = F.interpolate(imgs, size=(sH, sW), mode='bilinear', align_corners=True)
                im_sc = im_sc.cuda()
                logits = net(im_sc)
                logits = F.interpolate(logits, size=size, mode='bilinear', align_corners=True)
                probs += torch.softmax(logits, dim=1)
                if self.flip:
                    im_sc = torch.flip(im_sc, dims=(3, ))
                    logits = net(im_sc)
                    logits = torch.flip(im_sc, dim=(3,))
                    logits = F.interpolate(logits, size=size, mode='bilinear', align_corners=True)
                    probs += torch.softmax(logits, dim=1)
                    
            preds = torch.argmax(probs, dim=1)
            keep = label !=self.ignore_lb
            hist +=torch.bincount(label[keep]*n_classes + preds[keep], minlength= n_classes**2).view(n_classes, n_classes)
        if dist.is_initialized():
            dist.all_reduce(hist, dist.ReduceOp.SUM)
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        miou = ious.mean()
        return miou.item()
                
@torch.no_grad()
def eval_model(net,dl):
    net.eval()
    m_ious = []
    single_scale = MscEvalV0((1.,), False)
    mIOU = single_scale(net, dl, 19)
    m_ious.append(mIOU)
    return np.mean(m_ious)

            