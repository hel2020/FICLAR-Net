import torch.nn as nn
import torch
from DCNV2 import *
import torch.nn.functional as F
from utils import DoubleConv2,ResidualBlock

class Attention(nn.Module):

    def __init__(self, in_channels, **kwargs):
        super(Attention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(1024,1024, 1, bias=False)
        self.fc2 = nn.Conv2d(1024,1024, 1, bias=False)
        self.sig = nn.Sigmoid()
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.res=ResidualBlock(1024,1024)

    def forward(self,x,mix):

        #print(mix.shape)
        m = self.res(x)
        p=self.softmax(self.res(m))*x
        chan=self.sig(self.fc2(self.fc1(self.avg_pool(x))))
        e =chan * p+x  # nchw

        return e



