import torch
import torchvision
from torch import nn


class CIFAR_NET(nn.Module):
    def __init__(self, regularization = None):
        super().__init__(CIFAR_NET, self)
        """
        
        input = 32 by 32 
         H = (H_in - dilation * (Kernel - 1) -1 )/stride + 1
        
        
        """
        if regularization == "BATCH_NORM":
            self.first_pert = nn.Sequential(nn.Conv2d(kernel_size=3, in_channels=3, out_channels=16),nn.ReLU(), nn.Conv2d(kernel_size=3, in_channels=16, out_channels=64),nn.ReLU(), regularization(64), nn.MaxPool2d(kernel_size=2, stride=2),nn.Conv2d(kernel_size=2, in_channels=64, out_channels=256), nn.ReLU(), regularization(256), )