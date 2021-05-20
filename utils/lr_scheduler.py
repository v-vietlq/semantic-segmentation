from re import S
from torch.optim import _LRScheduler
class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_epoch, power=0.9, last_epoch= -1) -> None:
        super(PolyLR, self).__init__(optimizer, last_epoch)
        self.max_epoch = max_epoch
        self.power = power
        
    def get_lr(self):
        return [ base_lr*((1.0 - float(self.last_epoch)/float(self.max_epoch))**self.power) for base_lr in self.base_lrs]
        
    
    