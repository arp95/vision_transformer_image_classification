# header files
import torch
import torch.nn as nn
import torchvision
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import numpy as np
import skimage
from skimage import io, transform
import glob
import csv
from PIL import Image
import time
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from PIL import ImageFile
torch.backends.cudnn.benchmark = True

    
# class: Cross-Entropy loss with Label Smoothing
class CrossEntropyLabelSmoothingLoss(torch.nn.Module):
    """Cross-Entropy loss with Label Smoothing
    Arguments:
        smoothing: the smoothing factor lies between 0 and 1
    """
    
    
    def __init__(self, smoothing=0.0):
        super(CrossEntropyLabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        log_prob = torch.nn.functional.log_softmax(pred, dim=-1)
        weight = input.new_ones(pred.size()) * (self.smoothing/(pred.size(-1)-1.))
        weight.scatter_(-1, target.unsqueeze(-1), (1.-self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss
    
    
class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
