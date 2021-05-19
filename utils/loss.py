import torch
import torch.nn as nn
import torch.nn.functional as F


class OhemCELoss(nn.Module):
    def __init__(self, thresh, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()

        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        n_min = labels[labels != self.ignore_lb].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:n_min]
        return torch.mean(loss)
    
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None) -> None:
        super().__init__()
        self.loss = nn.NLLLoss(weight)
        
    def forward(self, outputs, targets):
        outputs = outputs
        return self.loss(F.log_softmax(outputs, dim=1), targets)
    