from utils.metrics import Evaluator
from utils.cityscapes import get_data_loader
from utils.evaluate import eval_model
import torch
from models.bisenetv2 import BiSeNetV2
import numpy as np


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
np.random.seed(50)
torch.manual_seed(50)

if torch.cuda.is_available():
    torch.cuda.manual_seed(50)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_check_point(pretrained_pth, net, device):
    checkpoint = torch.load(pretrained_pth,map_location=device)
    multigpus = True
    for key in checkpoint:  # check if the model was trained in multiple gpus",
        if 'module' in key:
            multigpus = multigpus and True
        else:
            multigpus = False
    if multigpus:
        net = torch.nn.DataParallel(net)
    net.load_state_dict(checkpoint)
    net.to(device)
    net.eval()
    return net


if __name__== "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    import torchvision.transforms as T
    from PIL import Image
    val_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((1024,2048), interpolation=Image.NEAREST)
    ])
    dl = get_data_loader(datapth='data/cityscapes',annpath='data/cityscapes/val.txt',trans_func=val_transform,batch_size=4,mode='val')
    model = BiSeNetV2(n_classes= 19)
    model = get_check_point('model_final_v2.pth', model, device)
    miou = eval_model(model,dl)
    print(miou)
    