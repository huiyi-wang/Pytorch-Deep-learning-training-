##网络中的网络，NiN
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import tensor
def nin_block(in_channels,out_channels,kernel_size,strides,padding):
    '''
    :param in_channels:  输入通道数目
    :param out_channels: 输出通道数目
    :param kernel_size:  卷积核的窗口大小
    :param strides:      步幅大小
    :param padding:      填充大小
    :return: 返回值：NiN 网络块
    '''
    return nn.Sequential(
           nn.Conv2d(in_channels,out_channels,kernel_size,strides,padding),nn.ReLU(),
           nn.Conv2d(out_channels,out_channels,kernel_size=1),nn.ReLU(),
           nn.Conv2d(out_channels,out_channels,kernel_size=1),nn.ReLU())

net =  nn.Sequential(
       nin_block( in_channels = 1 , out_channels = 96 , kernel_size = 11 , strides = 4,padding = 0),
       nn.MaxPool2d( kernel_size = 3 , stride = 2 ),
       nin_block( in_channels = 96 , out_channels = 256 , kernel_size = 5 , strides = 1 , padding = 2 ),
       nn.MaxPool2d( kernel_size = 3 , stride = 2 ),
       nin_block( in_channels = 256 , out_channels = 384 , kernel_size = 3 , strides = 1 , padding = 1),
       nn.MaxPool2d( kernel_size = 3 , stride = 2),
       nn.Dropout(p=0.5),
       #类别标签为10
       nin_block(in_channels = 384,out_channels = 10 , kernel_size = 3 , strides = 1 , padding = 1),
       nn.AdaptiveAvgPool2d((1,1)),#平均池化层
       nn.Flatten())

X = torch.rand(size = (1,1,224,224))
print(net(X))
print('*'*100)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)





