from utils.metrics import Evaluator
from utils.cityscapes import get_data_loader
from utils.evaluate import eval_model
import torch
import torch.nn as nn
from models.bisenetv2 import BiSeNetV2
import numpy as np
import torch.distributed as dist

from utils.lr_scheduler import WarmupPolyLrScheduler

from train import validate_model
from train import get_check_point


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
np.random.seed(50)
torch.manual_seed(50)

if torch.cuda.is_available():
    torch.cuda.manual_seed(50)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



if __name__== "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    import torchvision.transforms as T
    from PIL import Image

    dl = get_data_loader(datapth='cityscapes',annpath='cityscapes/val.txt',batch_size=8,mode='val')
    
   
    net = BiSeNetV2(19)
    
    optimizer = torch.optim.Adam(net.parameters(),5e-4,(0.9, 0.999), eps=1e-08, weight_decay=5e-4)

    lr_schdr = WarmupPolyLrScheduler(optimizer, power=0.9,
    max_iter=150000, warmup_iter=1000,
    warmup_ratio=0.1, warmup='exp', last_epoch=-1,)
    
    
    
    net, optimizer,lr_schdr,epoch, max_miou = get_check_point(
        './pretrained_models/BiSeNetv2_epoch_343_acc_0.5937.pt',
        net,
        optimizer,
        lr_schdr,
        device
    )
    
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
    