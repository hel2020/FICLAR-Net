import torch
from torch import nn
from torchvision import datasets, transforms
from torchvision import models
from DCNV2 import *
from attention import Attention
from utils import *



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.concat_pool1 = concat_pool(32, 32, kernel_size=3, stride=2)

        self.conv_fusion1 = conv_1_init(64, 256, kernel_size=1, padding=0)
        self.conv2 = DoubleConv(256, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.concat_pool12 = concat_pool(32, 64, stride=4)
        self.concat_pool22 = concat_pool(64, 64, stride=2)
        self.conv_fusion2 = conv_1_init(64 * 3, 512, kernel_size=1, padding=0)

        self.conv3 = DoubleConv(512, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.concat_pool13 = concat_pool(32, 128, stride=8)
        self.concat_pool23 = concat_pool(64, 128, stride=4)
        self.concat_pool33 = concat_pool(128, 128)
        self.conv_fusion3 = conv_1_init(128 * 4, 1024, kernel_size=1, padding=0)

        self.conv4 = DoubleConv(1024, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.concat_pool14 = concat_pool(32, 256, stride=16)
        self.concat_pool24 = concat_pool(64, 256, stride=8)
        self.concat_pool34 = concat_pool(128, 256, stride=4)
        self.concat_pool44 = concat_pool(256, 256, stride=2)
        self.conv_fusion4 = conv_1_init(256 * 5, 2048, kernel_size=1, padding=0)

        self.conv5 = DoubleConv(2048, 512)
        self.D_conv=DoubleConv2(2048,1024)
        self.ff2 = DConv2(64, 512, 2, 2, 2)
        self.ff3 = DConv3(128, 512, 2, 2, 1)
        self.ff4 = DConv4(256, 512, 2, 1, 1)
        self.ff5 = nn.Conv2d(512, 512, 1)
        self.conv55 = nn.Conv2d(2048, 1024, 1)
        self.D_conv=DoubleConv2(2048,1024)



    def forward(self, x):
            c1 = self.conv1(x)
            p1 = self.pool1(c1)
            concat_pool11 = self.concat_pool1(c1, p1)  # [8, 64, 128, 128]
            fusion1 = self.conv_fusion1(concat_pool11)

            c2 = self.conv2(fusion1)
            p2 = self.pool2(c2)
            # print(p2.shape, c1.shape)
            concat_pool12 = self.concat_pool12(c1, p2)
            concat_pool22 = self.concat_pool22(c2, concat_pool12)
            fusion2 = self.conv_fusion2(concat_pool22)

            c3 = self.conv3(fusion2)
            p3 = self.pool3(c3)
            concat_pool13 = self.concat_pool13(c1, p3)
            concat_pool23 = self.concat_pool23(c2, concat_pool13)
            concat_pool33 = self.concat_pool33(c3, concat_pool23)
            fusion3 = self.conv_fusion3(concat_pool33)

            c4 = self.conv4(fusion3)
            p4 = self.pool4(c4)
            concat_pool14 = self.concat_pool14(c1, p4)
            concat_pool24 = self.concat_pool24(c2, concat_pool14)
            concat_pool34 = self.concat_pool34(c3, concat_pool24)
            concat_pool44 = self.concat_pool44(c4, concat_pool34)
            fusion4 = self.conv_fusion4(concat_pool44)
            # print(fusion4.shape) # 8, 2048, 16, 16

            c5 = self.conv5(fusion4)
            c5 = torch.cat([self.ff2(c2), self.ff3(c3), self.ff4(c4), self.ff5(c5)], dim=1)

            return c5,c1,c2,c3,c4


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.D_conv=DoubleConv2(2048,1024)
        self.ff2 = DConv2(64, 512, 2, 2, 2)
        self.ff3 = DConv3(128, 512, 2, 2, 1)
        self.ff4 = DConv4(256, 512, 2, 1, 1)
        self.ff5 = nn.Conv2d(512, 512, 1)
        self.conv55 = nn.Conv2d(2048, 1024, 1)
        self.att = Attention(1024)

        self.skip_conv1 = conv(32, 64)
        self.skip_conv2 = conv(64, 128)
        self.skip_conv3 = conv(128, 256)
        self.skip_conv4 = conv(256, 512)

        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)

        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)

        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)

        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)

        self.conv10 = nn.Conv2d(64, 1, 1)

    def forward(self, c5,mix,c1,c2,c3,c4):

        # c5_x4=self.att(c5_x4)
        c5 = self.D_conv(mix)
        c5 = self.att(c5,mix)

        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, self.skip_conv4(c4)], dim=1)
        merge6= self.conv6(merge6)
        c6 = merge6
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, self.skip_conv3(c3)], dim=1)
        merge7= self.conv7(merge7)
        c7 = merge7
        up_8 = self.up8(c7)  # 64
        merge8 = torch.cat([up_8, self.skip_conv2(c2)], dim=1)
        merge8= self.conv8(merge8)
        c8 = merge8
        up_9 = self.up9(c8)  # 128
        merge9 = torch.cat([up_9, self.skip_conv1(c1)], dim=1)
        merge9= self.conv9(merge9)
        c9 = merge9
        c10 = self.conv10(c9)
        c10 = torch.sigmoid(c10)

        return c10


class Single_Conv(nn.Module):
    def __init__(self):
        super(Single_Conv, self).__init__()
        self.conv1 = nn.Conv2d(2048,2048,stride=1,kernel_size = 1)
        self.conv2 = nn.Conv2d(4096,2048,stride=1,kernel_size = 1)

    def forward(self, mid,fake_mid):
        mid= self.conv1(mid)
        mix= torch.cat([fake_mid, mid],dim=1)
        x = self.conv2(mix)
        return x









