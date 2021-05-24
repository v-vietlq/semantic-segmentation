
import torch 
import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler
from utils.cityscapes import get_data_loader
from utils.meters import TimeMeter, AvgMeter
from utils.loss import OhemCELoss
from tqdm import tqdm
from utils.metrics import Evaluator
import torch.nn.functional as F
import torchvision.transforms as T
from utils.cityscapes import get_data_loader
from utils.evaluate import eval_model
import segmentation_models_pytorch as smp
from models.bisenetv2 import BiSeNetV2
import numpy as np
import torch.distributed as dist
from utils.lr_scheduler import WarmupPolyLrScheduler
from torch.utils.tensorboard.writer import SummaryWriter


    
def train_per_epoch(model, criterion, optimizer, scheduler, dataloader, device):
    
    # total_iter = len(dataloader) * dataloader.batch_size * epoch
    
    model.train()
    
    criteria_pre = criterion
    criteria_aux = [criterion for _ in range(4)]
    
    
    for it, (image, target) in tqdm(enumerate(dataloader)):
        image , target = image.to(device, dtype=torch.float), target.to(device)
        target = torch.squeeze(target, 1)
        logits, *logits_aux = model(image)
        loss_pre = criteria_pre(logits, target)
        loss_aux = [crit(lgt, target) for crit, lgt in zip(criteria_aux, logits_aux)]
        loss = loss_pre + sum(loss_aux)
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
        torch.cuda.synchronize()
        
        scheduler.step()
        
        # total_iter += dataloader.batch_size
        
        
    print(optimizer.param_groups[0]['lr'])
    
    return 


def validate_model(model, valid_loader, device):

    model.eval()
    evaluator = Evaluator(19)
    evaluator.reset()
    for _,(image, target) in enumerate(tqdm(valid_loader)):
        
        # 2.1. Get images and groundtruths (i.e. a batch), then send them to 
        # the device every iteration
        image , target = image.to(device, dtype=torch.float), target.to(device)
        target = torch.squeeze(target, 1)
        
        image = T.Resize((512,1024))(image)
        
        with torch.no_grad():
            output = model(image)[0]
            
        seg_map = F.interpolate(output, size=(1024,2048),
                            mode='bilinear', align_corners=True)

           
        # 2.3. Compute the batch loss
        seg_map = torch.argmax(seg_map, dim=1)

        seg_map = seg_map.cpu().detach().numpy()
        target      = target.cpu().detach().numpy()
        evaluator.add_batch(target,seg_map)
    Acc = evaluator.Pixel_Accuracy()
    mIoU = evaluator.Mean_Intersection_over_Union()
    f1 = evaluator.F1_score()
                
    return mIoU, Acc, f1




def get_check_point(pretrained_pth, net, optimizer,scheduler, device):
    checkpoint = torch.load(pretrained_pth, map_location=device)
    
    model_state_dict = checkpoint['model_state_dict']
    
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    
    epoch = checkpoint['epoch']
    
    max_miou = checkpoint['max_miou']
    
    net.load_state_dict(model_state_dict)
    net.to(device)
    
    optimizer.load_state_dict(optimizer_state_dict)
    
    scheduler.load_state_dict(checkpoint['scheduler'])

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
    import torchvision.transforms as T
    import torch
    import matplotlib.pyplot as plt
    from torch.utils.tensorboard.writer import SummaryWriter
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(50)
    torch.manual_seed(50)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(50)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    num_epochs = 1000
    max_acc = 0
    patience = 100
    not_improved_count = 0
    batch_size = 8
    
    
    val_loader = get_data_loader(datapth='cityscapes',annpath='cityscapes/val.txt',batch_size=batch_size,mode='val')
    train_loader = get_data_loader(datapth='cityscapes',annpath='cityscapes/train.txt',batch_size=batch_size,mode='train')
    
    net = BiSeNetV2(n_classes= 19).to(device)
    
    criterion = OhemCELoss(thresh=0.7)
    
    optimizer = torch.optim.SGD(net.parameters(),lr = 1e-2,momentum=0.9)

    lr_schdr = WarmupPolyLrScheduler(optimizer, power=0.9,
    max_iter=150000, warmup_iter=1000,
    warmup_ratio=0.1, warmup='exp', last_epoch=-1,)
    
    
    
    net, optimizer,lr_scheduler,epoch, max_miou = get_check_point(
        './pretrained_models/BiSeNetv2_epoch_371_acc_0.5997.pt',
        net,
        optimizer,
        lr_schdr,
        device
    )
    
    
    writer = SummaryWriter('experiment')

    for epoch in range(epoch+1, num_epochs):
        train_per_epoch(net, criterion, optimizer, lr_schdr, train_loader, device)
        # val_iou= eval_model(net, val_loader)
        val_iou, val_f1, val_acc = validate_model(net, val_loader, device)

        print('Epoch: {}'.format(epoch))
        print('Valid_f1: {}'.format(val_f1))
        print('Valid_iou: {:.4f}'.format(val_iou))
        
        writer.add_scalar("mIoU", val_iou, epoch)
        writer.add_scalar("mDice",val_f1,epoch)
        
        if val_iou > max_miou:
            
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': net.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'max_miou': val_iou,
                'scheduler': lr_schdr.state_dict()
            }
            
            path = './pretrained_models/BiSeNetv2_epoch_' + str(epoch) + '_acc_{0:.4f}'.format(val_iou)+'.pt'
            
            
            torch.save(best_checkpoint, path)
            
            max_miou = val_iou
            
            not_improved_count = 0
        else:
            not_improved_count+=1
        
        if not_improved_count >=patience:
            break
        
    writer.close()

    


