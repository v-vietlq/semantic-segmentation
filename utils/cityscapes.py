from torch.utils.data import DataLoader
import numpy as np


from utils.load_dataset import BaseDataset
import torchvision.transforms as transforms
import utils.augment as T



labels_info = [
    {"hasInstances": False, "category": "void", "catid": 0, "name": "unlabeled", "ignoreInEval": True, "id": 0, "color": [0, 0, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "ego vehicle", "ignoreInEval": True, "id": 1, "color": [0, 0, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "rectification border", "ignoreInEval": True, "id": 2, "color": [0, 0, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "out of roi", "ignoreInEval": True, "id": 3, "color": [0, 0, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "static", "ignoreInEval": True, "id": 4, "color": [0, 0, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "dynamic", "ignoreInEval": True, "id": 5, "color": [111, 74, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "ground", "ignoreInEval": True, "id": 6, "color": [81, 0, 81], "trainId": 255},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "road", "ignoreInEval": False, "id": 7, "color": [128, 64, 128], "trainId": 0},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "sidewalk", "ignoreInEval": False, "id": 8, "color": [244, 35, 232], "trainId": 1},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "parking", "ignoreInEval": True, "id": 9, "color": [250, 170, 160], "trainId": 255},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "rail track", "ignoreInEval": True, "id": 10, "color": [230, 150, 140], "trainId": 255},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "building", "ignoreInEval": False, "id": 11, "color": [70, 70, 70], "trainId": 2},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "wall", "ignoreInEval": False, "id": 12, "color": [102, 102, 156], "trainId": 3},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "fence", "ignoreInEval": False, "id": 13, "color": [190, 153, 153], "trainId": 4},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "guard rail", "ignoreInEval": True, "id": 14, "color": [180, 165, 180], "trainId": 255},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "bridge", "ignoreInEval": True, "id": 15, "color": [150, 100, 100], "trainId": 255},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "tunnel", "ignoreInEval": True, "id": 16, "color": [150, 120, 90], "trainId": 255},
    {"hasInstances": False, "category": "object", "catid": 3, "name": "pole", "ignoreInEval": False, "id": 17, "color": [153, 153, 153], "trainId": 5},
    {"hasInstances": False, "category": "object", "catid": 3, "name": "polegroup", "ignoreInEval": True, "id": 18, "color": [153, 153, 153], "trainId": 255},
    {"hasInstances": False, "category": "object", "catid": 3, "name": "traffic light", "ignoreInEval": False, "id": 19, "color": [250, 170, 30], "trainId": 6},
    {"hasInstances": False, "category": "object", "catid": 3, "name": "traffic sign", "ignoreInEval": False, "id": 20, "color": [220, 220, 0], "trainId": 7},
    {"hasInstances": False, "category": "nature", "catid": 4, "name": "vegetation", "ignoreInEval": False, "id": 21, "color": [107, 142, 35], "trainId": 8},
    {"hasInstances": False, "category": "nature", "catid": 4, "name": "terrain", "ignoreInEval": False, "id": 22, "color": [152, 251, 152], "trainId": 9},
    {"hasInstances": False, "category": "sky", "catid": 5, "name": "sky", "ignoreInEval": False, "id": 23, "color": [70, 130, 180], "trainId": 10},
    {"hasInstances": True, "category": "human", "catid": 6, "name": "person", "ignoreInEval": False, "id": 24, "color": [220, 20, 60], "trainId": 11},
    {"hasInstances": True, "category": "human", "catid": 6, "name": "rider", "ignoreInEval": False, "id": 25, "color": [255, 0, 0], "trainId": 12},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "car", "ignoreInEval": False, "id": 26, "color": [0, 0, 142], "trainId": 13},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "truck", "ignoreInEval": False, "id": 27, "color": [0, 0, 70], "trainId": 14},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "bus", "ignoreInEval": False, "id": 28, "color": [0, 60, 100], "trainId": 15},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "caravan", "ignoreInEval": True, "id": 29, "color": [0, 0, 90], "trainId": 255},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "trailer", "ignoreInEval": True, "id": 30, "color": [0, 0, 110], "trainId": 255},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "train", "ignoreInEval": False, "id": 31, "color": [0, 80, 100], "trainId": 16},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "motorcycle", "ignoreInEval": False, "id": 32, "color": [0, 0, 230], "trainId": 17},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "bicycle", "ignoreInEval": False, "id": 33, "color": [119, 11, 32], "trainId": 18},
    {"hasInstances": False, "category": "vehicle", "catid": 7, "name": "license plate", "ignoreInEval": True, "id": -1, "color": [0, 0, 142], "trainId": -1}
]

class CityScapes(BaseDataset):
    def __init__(self, dataroot, annpath, trans_func, mode='train') -> None:
        super(CityScapes, self).__init__(dataroot, annpath, trans_func=trans_func, mode=mode)
        self.n_cat = 19
        self.lb_ignore = 255
        self.lb_map=np.arange(256).astype(np.uint8)
        for el in labels_info:
            self.lb_map[el['id']] = el['trainId']
            
def transform_train():
    composed_transforms = transforms.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop((512,1024), scale=(0.25, 2.)),
        T.RandomGaussianBlur(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
        ),
        T.ToTensor()
    ])
    return composed_transforms

def transform_val():
    compose_transforms = transforms.Compose([
        T.FixedResize((512,1024)),
        T.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
            ),
        T.ToTensor()
    ])
    return compose_transforms
    
def transform_test():
    compose_transforms = transforms.Compose([
        T.FixedResize((512,1024)),
        T.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
            ),
        T.ToTensor()
    ])
    return compose_transforms
        

def get_data_loader(datapth, annpath,batch_size = 4, mode='train'):
    if mode=='train':
        trans_func = transform_train()
        shuffle = True
        drop_last = True
    elif mode =='val':
        trans_func = transform_val()
        shuffle = False
        drop_last = False
    else:
        trans_func = transform_test()
        shuffle = False
        drop_last = False
        
        
    ds = CityScapes(datapth, annpath,trans_func=trans_func,mode=mode)
    
    dl = DataLoader(
        ds, 
        batch_size= batch_size, 
        shuffle= shuffle, 
        drop_last= drop_last,
        num_workers= 4, 
        pin_memory=True)
    
    return dl
    
    
if __name__ == '__main__':

    
    import matplotlib.pyplot as plt    
    from utils import decode_segmap
    
    
    
    dl = get_data_loader(datapth='data/cityscapes',annpath='data/cityscapes/train.txt',batch_size=4,mode='train')
    img , gt = dl.dataset.__getitem__(0)
    
    
    print(img.shape, gt.shape)
    print(np.unique(gt))
    
    
    segmap = np.array(gt).astype(np.uint8)
    segmap = decode_segmap(segmap)
    img_tmp = np.transpose(img, axes=[1, 2, 0]).numpy()
    img_tmp *= (0.229, 0.224, 0.225)
    img_tmp += (0.485, 0.456, 0.406)
    img_tmp *= 255.0
    img_tmp = img_tmp.astype(np.uint8)
    fig = plt.figure()
    plt.title('display')
    plt.subplot(211)
    plt.imshow(img_tmp)
    plt.subplot(212)
    plt.imshow(segmap)
    fig.savefig('sample.png')
    