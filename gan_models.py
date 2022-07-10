import torch
from torch import nn
import torch.functional as F
from utils import ResidualBlock


class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        
        self.layers1 = self.layers(32, 64, layer="first_layer")
        self.layers2 = self.layers(64,128)
        self.layers3 = self.layers(128,256)
        self.layers4 = self.layers(256,512)
        self.layers5 = self.layers(512, 1024)
        self.layers6 = self.layers(1024,2048,layer="last_layer")

    def layers(self, inchannel, outchannel, kernel_size=3, stride=2, padding=1, layer=None):
        if layer == "first_layer":

            x = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size, 1, padding, bias=False),
                              nn.LeakyReLU(0.2, inplace=True))
            return x
        
        elif layer == "last_layer":
            
            x = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size, 1, padding, bias=False),
                              nn.Sigmoid())
            return x
        
        else:
            x = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size, stride, padding, bias=False),
                              nn.BatchNorm2d(outchannel),
                              nn.LeakyReLU(0.2, inplace=True))
            return x

    def forward(self,x,without_noise=None):
        
        x = self.layers6(self.layers5(self.layers4(self.layers3(self.layers2(self.layers1(x))))))
        
        return x

class Discriminator(nn.Module):
    def __init__(self, input_dim=1024):
        super(Discriminator, self).__init__()

        self.layers1 = self.layers(2048,1024)
        self.layers2 = self.layers(1024,512)
        self.layers3 = self.layers(512,256)
        self.layers4 = self.layers(256,128)
        self.layers5 = self.layers(128,64,layer="last_layer")
        self.layers6 = self.layers(64,1,layer="last_layer")

    def layers(self, in_channel, out_channel, kernel_size=3, stride=2, padding=1, layer=None):

        if layer == "last_layer":

            x = nn.Sequential(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, 1, padding, bias=False),
                              nn.Sigmoid())
            return x
        else:

            x = nn.Sequential(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding, bias=False),
                              nn.BatchNorm2d(out_channel),
                              nn.ReLU(True))
            return x

    def forward(self, x):

        x = self.layers6(self.layers5(self.layers4(self.layers3(self.layers2(self.layers1(x))))))

        return x
