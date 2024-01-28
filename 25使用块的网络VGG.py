import torch
from torch import nn
import pandas as pd
import numpy as np
def vgg_block(num_convs,in_channels,out_channels):##定义VGG的块
    '''
    :param num_convs:卷积层的数量
    :param in_channels:输入通道的数量
    :param out_channels:输出通道的数量
    :return:返回最后的那个网络network
    '''
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    #卷积部分
    for (num_convs,out_channels) in conv_arch:
        '''
        num_convs表示卷积层个数
        out_channels表示输出通道数
        '''
        conv_blks.append(vgg_block(num_convs,in_channels,out_channels))
        in_channels = out_channels
    return nn.Sequential(*conv_blks,nn.Flatten(),
           #  全连接部分
           nn.Linear(out_channels*7*7,4096) , nn.ReLU() , nn.Dropout(p = 0.5),
           nn.Linear(4096,4096) , nn.ReLU() , nn.Dropout(p =0.5),
           nn.Linear(4096,10))

#conv_arch中的第一个参数表示输入通道数，第二个参数表示输出通道数
#conv_arch(num_convs,out_channels)
conv_arch = ((1,64),(1,128),(2,256),(2,512),(2,512))
net = vgg(conv_arch)
X = torch.randn(size=(1,1,224,224))
print('将X经过net后的结果输出出来')
net(X)
print(net(X))
print('输出VGG卷积神经网络框架的每一层结构')
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)