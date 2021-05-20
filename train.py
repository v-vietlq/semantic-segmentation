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

from utils.metrics import Evaluator
from utils.cityscapes import get_data_loader
from utils.evaluate import eval_model
import torch
from models.bisenetv2 import BiSeNetV2
import numpy as np
import torch.distributed as dist


    
def train_per_epoch(model,optimizer, dataloader, device):
    model.train()
    criteria_pre = OhemCELoss(0.7)
    criteria_aux = [OhemCELoss(0.7) for _ in range(4)]
    for _, (image, target) in tqdm(enumerate(dataloader)):
        image , target = image.to(device, dtype=torch.float), target.to(device)
        target = torch.squeeze(target, 1)
        logits, *logits_aux = model(image)
        loss_pre = criteria_pre(logits, target)
        loss_aux = [crit(lgt, target) for crit, lgt in zip(criteria_aux, logits_aux)]
        loss = loss_pre + sum(loss_aux)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return 

@torch.no_grad()
def validate_model(model, valid_loader, device):

    model.eval()
    evaluator = Evaluator(19)
    for _,(image, target) in enumerate(valid_loader):
        
        # 2.1. Get images and groundtruths (i.e. a batch), then send them to 
        # the device every iteration
        image , target = image.to(device, dtype=torch.float), target.to(device)
        target = torch.squeeze(target, 1)
        output = model(image)[0]

        # 2.2. Perform a feed-forward pass
           
        # 2.3. Compute the batch loss
        seg_map = torch.argmax(output, dim=1)
        seg_map = seg_map.cpu().detach().numpy()
        target      = target.cpu().detach().numpy()
        evaluator.add_batch(target,seg_map)
        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
                
    return Acc, Acc_class, mIoU, FWIoU




def get_check_point(pretrained_pth, net, device):
    checkpoint = torch.load(pretrained_pth, map_location=device)
    net.load_state_dict(checkpoint)
    net.to(device)

    is_dist = dist.is_initialized()
    if is_dist:
        local_rank = dist.get_rank()
        net = nn.parallel.DistributedDataParallel(
            net,
            device_ids=[local_rank, ],
            output_device=local_rank
        )
    return net


if __name__== "__main__":
    from tqdm import tqdm
    import torchvision.transforms as T
    from PIL import Image
    import torch
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(50)
    torch.manual_seed(50)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(50)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    
    train_transform = T.Compose([
        T.ToPILImage(),
        # T.RandomResizedCrop((512,1024),scale=(0.25, 2.)),
        T.RandomHorizontalFlip(),
        # T.ColorJitter(
        #     brightness=0.4,
        #     contrast=0.4,
        #     saturation=0.4
        # ),
    ])
    
    val_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((512,1024), interpolation=Image.NEAREST)
    ])
    num_epochs = 50
    max_acc = 0
    patience = 10
    not_improved_count = 0
    batch_size = 2
    
    val_loader = get_data_loader(datapth='data/cityscapes',annpath='data/cityscapes/val.txt',trans_func=val_transform,batch_size=4,mode='val')
    train_loader = get_data_loader(datapth='data/cityscapes',annpath='data/cityscapes/train.txt',trans_func=train_transform,batch_size=batch_size,mode='train')
    
    net = BiSeNetV2(n_classes= 19)
    net = get_check_point('model_final_v2.pth', net, device)
    
    # criterion = OhemCELoss(thresh=0.7)
    optimizer = torch.optim.Adam(net.parameters(),5e-4,(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    for epoch in range(num_epochs):
        train_per_epoch(net,optimizer, train_loader, device)
        val_iou= eval_model(net, val_loader)

        print('Epoch: {}'.format(epoch))
        print('Valid_iou: {:.4f}'.format(val_iou))

        if val_iou > max_acc:
            torch.save(net.state_dict(), './pretrained_model/BiSeNetv2_epoch_' + str(epoch) + '_acc_{0:.4f}'.format(val_iou)+'.pt')
            max_acc = val_iou
            not_improved_count = 0
        else:
            not_improved_count+=1
        
        if not_improved_count >=patience:
            break

    


