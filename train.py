import imp
import torch 
import torch.nn as nn
import random
import numpy as np
import models
from utils.cityscapes import get_data_loader
from utils.meters import TimeMeter, AvgMeter
from utils.loss import OhemCELoss
from tqdm import tqdm
from utils.metrics import Evaluator

torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.deterministic = True


# def parse_args():
#     parse = argparse.ArgumentParser()
#     parse.add_argument('--model', dest='model', type=str, default='bisenetv1')
    
# args = parse_args()
# cfg = cfg_factory[args.model]


    
def train_per_epoch(model, criterion, optimizer, dataloader, device):
    model.train()
    model.to(device)
    for _, (image, target) in tqdm(enumerate(dataloader)):
        image , target = image.to(device, dtype=torch.float), target.to(device)
        target = torch.squeeze(target, 1)
        output = model(image)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return 

@torch.no_grad()
def validate_model(model, criterion, valid_loader, device):

    model.eval()
    evaluator = Evaluator(19)
    ious = []
    for _,(image, target) in enumerate(valid_loader):
        
        # 2.1. Get images and groundtruths (i.e. a batch), then send them to 
        # the device every iteration
        image , target = image.to(device, dtype=torch.float), target.to(device)
        target = torch.squeeze(target, 1)
        output = model(image)

        # 2.2. Perform a feed-forward pass
           
        # 2.3. Compute the batch loss
        loss = criterion(output, target)
        seg_map = torch.argmax(output, dim=1)
        seg_map = seg_map.cpu().detach().numpy()
        target      = target.cpu().detach().numpy()
        evaluator.add_batch(target,seg_map)
        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
                
    return Acc, Acc_class, mIoU, FWIoU