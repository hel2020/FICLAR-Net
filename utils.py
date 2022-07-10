import h5py
import torch
import shutil
import torch
from torch import nn
from DCNV2 import *
from convlstm import *
from torchvision import models

def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())
def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():        
            param = torch.from_numpy(np.asarray(h5f[k]))         
            v.copy_(param)
            
def save_checkpoint(state,is_best, task_id, filename='checkpoint.pth.tar'):
    torch.save(state, task_id+filename)
    if is_best:
        shutil.copyfile(task_id+filename, task_id+'model_best.pth.tar')



class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1),nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True),
                                 nn.Conv2d(out_ch, out_ch, 3, padding=1),nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True))
    def forward(self, input):
        return self.conv(input)

class DoubleConv1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv1, self).__init__()
        self.conv = nn.Sequential(DeformConv2D(in_ch, out_ch, 3, padding=1),nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True),
                                             nn.Conv2d(out_ch, out_ch, 3, padding=1),nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True))
    def forward(self, input):
        return self.conv(input)

class DoubleConv2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv2, self).__init__()
        self.conv = nn.Sequential(DeformConv2D(in_ch, out_ch),nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                                  DeformConv2D(out_ch, out_ch),nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x):
        return self.conv(x)

class DConv(nn.Module):
    def __init__(self, in_ch, out_ch,s1,s2,s3):
        super(DConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3,stride=s1,dilation=1, padding=1),nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True),
                                  nn.Conv2d(out_ch, out_ch, 3,stride=s2, dilation=2,padding=2),nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True),
                                  nn.Conv2d(out_ch, out_ch, 3,stride=s3, dilation=3,padding=3), nn.BatchNorm2d( out_ch),nn.ReLU(inplace=True),
                                  nn.Conv2d(out_ch, out_ch, 3,stride=1,dilation=1, padding=1),nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True)
                                  )
    def forward(self, input):
        return self.conv(input)

class DConv2(nn.Module):
    def __init__(self, in_ch, out_ch, s1, s2, s3):
        super(DConv2, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, stride=s1, dilation=1, padding=1), nn.BatchNorm2d(out_ch),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_ch, out_ch, 3, stride=s2, dilation=2, padding=2),
                                  nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                                  nn.Conv2d(out_ch, out_ch, 3, stride=s3, dilation=3, padding=3),
                                  nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),

                                  nn.Conv2d(out_ch, out_ch, 3, stride=1, dilation=1, padding=1), nn.BatchNorm2d(out_ch),
                                  nn.ReLU(inplace=True)
                                  )

    def forward(self, input):
        return self.conv(input)

class DConv3(nn.Module):
    def __init__(self, in_ch, out_ch, s1, s2, s3):
        super(DConv3, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, stride=s1, dilation=1, padding=1), nn.BatchNorm2d(out_ch),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_ch, out_ch, 3, stride=s2, dilation=2, padding=2),
                                  nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),

                                  nn.Conv2d(out_ch, out_ch, 3, stride=1, dilation=1, padding=1), nn.BatchNorm2d(out_ch),
                                  nn.ReLU(inplace=True)
                                  )

    def forward(self, input):
        return self.conv(input)

class DConv4(nn.Module):
    def __init__(self, in_ch, out_ch, s1, s2, s3):
        super(DConv4, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, stride=s1, dilation=1, padding=1), nn.BatchNorm2d(out_ch),
                                  nn.ReLU(inplace=True),

                                  nn.Conv2d(out_ch, out_ch, 3, stride=1, dilation=1, padding=1), nn.BatchNorm2d(out_ch),
                                  nn.ReLU(inplace=True)
                                  )

    def forward(self, input):
        return self.conv(input)

class conv(nn.Module):
    def __init__(self, in_ch, out_ch,stride=1,dil=1,pad=0,kernel_size = 1):
        super(conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch,stride=stride,kernel_size = kernel_size, dilation = dil, padding = pad),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_1_init(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size = 3, stride = 1, padding = 1):
        super(conv_1_init, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activate='lrelu'):
        super(ResidualBlock, self).__init__()
        self.conv_shortcut = nn.Sequential(nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1,padding=1),
                                          nn.BatchNorm2d(output_dim))
        self.conv = nn.Sequential(nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
                                             nn.BatchNorm2d(output_dim),nn.ReLU(),
                                nn.Conv2d(in_channels=output_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
                                            nn.BatchNorm2d(output_dim))
        self.activate=activate
        if self.activate == 'relu':
            self.nonlinear = nn.ReLU()
        else:
            self.nonlinear = nn.LeakyReLU()

    def forward(self, inputs):

        shortcut = self.conv_shortcut(inputs)
        x=self.conv(inputs)
        return self.nonlinear(x + inputs)

class concat_pool(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size = 3, stride = 2, padding = 1):
        super(concat_pool, self).__init__()
        self.conv_pool = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace = True)
        )

    def forward(self, conv, pool):
        conv = self.conv_pool(conv)
        cat = torch.cat([conv, pool], dim = 1)
        return cat

