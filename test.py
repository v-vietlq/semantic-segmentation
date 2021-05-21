from utils.metrics import Evaluator
from utils.cityscapes import get_data_loader
from utils.evaluate import eval_model
import torch
import torch.nn as nn
from models.bisenetv2 import BiSeNetV2
import numpy as np
import torch.distributed as dist

from torch.optim.lr_scheduler import ExponentialLR

from train import validate_model


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
np.random.seed(50)
torch.manual_seed(50)

if torch.cuda.is_available():
    torch.cuda.manual_seed(50)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_check_point(pretrained_pth, net, optimizer,scheduler, device):
    checkpoint = torch.load(pretrained_pth, map_location=device)
    
    model_state_dict = checkpoint['model_state_dict']
    
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    
    epoch = checkpoint['epoch']
    
    max_miou = checkpoint['max_miou']
    
    net.load_state_dict(model_state_dict)
    net.to(device)
    
    optimizer.load_state_dict(optimizer_state_dict)
    
    scheduler = checkpoint['scheduler']

    is_dist = dist.is_initialized()
    if is_dist:
        local_rank = dist.get_rank()
        net = nn.parallel.DistributedDataParallel(
            net,
            device_ids=[local_rank, ],
            output_device=local_rank
        )
    return net, optimizer, scheduler, epoch, max_miou


if __name__== "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    import torchvision.transforms as T
    from PIL import Image

    dl = get_data_loader(datapth='data/cityscapes',annpath='data/cityscapes/val.txt',batch_size=2,mode='val')
    
   
    net = BiSeNetV2(19)
    
    optimizer = torch.optim.Adam(net.parameters(),5e-4,(0.9, 0.999), eps=1e-08, weight_decay=5e-4)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    
    net.load_state_dict(torch.load('model_final_v2.pth'))
    net.cuda()

    is_dist = dist.is_initialized()
    if is_dist:
        local_rank = dist.get_rank()
        net = nn.parallel.DistributedDataParallel(
            net,
            device_ids=[local_rank, ],
            output_device=local_rank
        )
    miou = eval_model(net,dl)
    print(miou)
    