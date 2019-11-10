from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


class BaselineAMT(nn.Module):

    def __init__(self):
        super(BaselineAMT,self).__init__()
        self.conv1 = nn.Conv2D(1,64,kernel_size=3)
        self.conv2 = nn.Conv2D(64,64,kernel_size=3)
        self.conv3 = nn.Conv2D(64,64,kernel_size=3)
        return

    def forward(x):
        

