import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

class Residual(nn.Module):
    def __init__(self,input_channels,num_channels,use_1X1conv = False,strides = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels,num_channels,kernel_size=3,padding=1,stride=strides)
        self.conv2 = nn.Conv2d(num_channels,num_channels,kernel_size=3,padding=1)
        if use_1X1conv:
            self.conv3 = nn.Conv2d(input_channels,num_channels,kernel_size=1,stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
    def forward(self,X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y = Y + X
        return F.relu(Y)
# 输出和输入形状一样的情况
print('输出和输入形状一样的情况')
blk1 = Residual(3,3)
X = torch.rand(4,3,6,6)
Y = blk1(X)
print(Y.shape)
# 输出和输入情况不一样的情况:增加输出通道数，减半输出的高和宽
print('输出和输入情况不一样的情况:增加输出通道数，减半输出的高和宽')
blk2 = Residual(3,6,use_1X1conv=True,strides=2)
print(blk2(X).shape)

##下面开始构建ResNet模型 （使用搭建好的ResNet块）
print('下面开始构建ResNet模型 （使用搭建好的ResNet块）')
b1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=64,kernel_size=7,stride=2,padding=3),
                   nn.BatchNorm2d(64),nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

def resnet_block(input_channels,num_channels,num_residuals,first_block=False):
    '''
    :param input_channels:输入通道数
    :param num_channels:  输出通道数
    :param num_residuals: Residual块的数目
    :param first_block:  False or True
    :return:
    '''
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels,num_channels,use_1X1conv=True,strides=2))
        else:
            blk.append(Residual(num_channels,num_channels))
    return blk
b2 = nn.Sequential(*resnet_block(input_channels=64,num_channels=64,num_residuals=2,first_block=True))
b3 = nn.Sequential(*resnet_block(input_channels=64,num_channels=128,num_residuals=2))
b4 = nn.Sequential(*resnet_block(input_channels=128,num_channels=256,num_residuals=2))
b5 = nn.Sequential(*resnet_block(input_channels=256,num_channels=512,num_residuals=2))
net = nn.Sequential(b1,b2,b3,b4,b5,nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(512,10))
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)