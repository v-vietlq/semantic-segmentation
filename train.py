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
from utils.evaluate import eval_model
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.metrics import IoU, Fscore, Recall, Precision, Accuracy
from segmentation_models_pytorch.utils.meter import AverageValueMeter

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


# def set_meters():
#     time_meter = TimeMeter(cfg.max_iter)
#     loss_meter = AvgMeter('loss')
#     loss_pre_meter = AvgMeter('loss_prem')
#     loss_aux_meters = [AvgMeter('Loss_aux{}'.format(i) for i in range(cfg.num_aux_heads))]
    
#     return time_meter, loss_meter, loss_pre_meter, loss_aux_meters

# def set_model():
#     net =model_factory[cfg.model_type](19)
#     if not args.finetune_from is None:
#         net.load_state_dict(torch.load(args.finetune_from, map_location='cpu'))
#     net.cuda()
#     net.train()
#     criteria_pre = OhemCELoss(0.7)
#     criteria_aux= [OhemCELoss(0.7) for _ in range(cfg.num_aux_heads)]
#     return net, criteria_pre, criteria_aux

    
def train_per_epoch(model, criterion, optimizer, dataloader, device):
    model.train()
    model.to(device)
    for _, (image, target) in tqdm(enumerate(dataloader)):
        image , target = image.to(device, dtype=torch.float), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        seg_map = torch.argmax(output, dim=1)
        seg_map = seg_map.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
 
    m_ious1= eval_model(models, 4, './datasets/cityscapes', './datasets/cityscapes/val.txt')
    
    return m_ious1

def validate_model(model, criterion, valid_loader, device):

    model.eval()

    ious = []
    for _,(image, target) in enumerate(valid_loader):
        
        # 2.1. Get images and groundtruths (i.e. a batch), then send them to 
        # the device every iteration
        image , target = image.to(device, dtype=torch.float), target.to(device)
        output = model(image)
        print(output.shape)

        # 2.2. Perform a feed-forward pass
           
        # 2.3. Compute the batch loss
        loss = criterion(output, target)
        seg_map = torch.argmax(output, dim=1)
        seg_map = seg_map.cpu().detach().numpy()
        target      = target.cpu().detach().numpy()
        iou = IoU(output, target)
        ious.append(iou)
                
    return np.mean(ious)