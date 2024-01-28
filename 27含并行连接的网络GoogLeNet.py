import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class Inception(nn.Module):
    # c1-c4为每条路径的输出通道数
    def __init__(self,in_channels,c1,c2,c3,c4,**kwargs):
        super(Inception,self).__init__(**kwargs)
        # 线路一：单卷积层
        self.p1_1 = nn.Conv2d(in_channels,c1,kernel_size=1)
        # 线路二：1 X 1 卷积层后 接3 X 3卷积层
        self.p2_1 = nn.Conv2d(in_channels,c2[0],kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0],c2[1],kernel_size=3,padding=1)
        # 线路三：1 X 1卷积层后接 5 X 5卷积层
        self.p3_1 = nn.Conv2d(in_channels,c3[0],kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0],c3[1],kernel_size=5,padding=2)
        # 线路四：3 X 3最大汇聚层后接1X1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.p4_2 = nn.Conv2d(in_channels,c4,kernel_size=1)
    def forward(self,X):
        p1 = F.relu(self.p1_1(X))
        p2 = F.relu(self.p2_2(F.relu_(self.p2_1(X))))
        p3 = F.relu(self.p3_2(F.relu_(self.p3_1(X))))
        p4 = F.relu(self.p4_2(self.p4_1(X)))
        #在通道维度上连接输出
        return torch.cat((p1,p2,p3,p4),dim=1)

'现在开始逐一实现GoogLeNet每一个模块'
#第一个模块：使用64个通道，7X7卷积层
b1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=64,kernel_size=7,stride=2,padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
#第二个模块：使用两个卷积层：第一个卷积层是64个通道，1X1卷积层；第二个卷积层使用将通道数量增加三倍的3X3的卷积层。
b2 = nn.Sequential(nn.Conv2d(64,64,kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64,192,kernel_size=3,padding=1),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
#第三个模块：
b3 = nn.Sequential(Inception(192,64,(96,128),(16,32),32),
                   Inception(256,128,(128,192),(32,96),64),
                   nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
#第四个模块：
b4 = nn.Sequential(Inception(480,192,(96,208),(16,48),64),
                   Inception(512,160,(112,224),(24,64),64),
                   Inception(512,128,(128,256),(24,64),64),
                   Inception(512,112,(114,288),(32,64),64),
                   Inception(528,256,(160,320),(32,128),128),
                   nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
#第五个模块：
b5 = nn.Sequential(Inception(832,256,(160,320),(32,128),128),
                   Inception(832,384,(192,384),(48,128),128),
                   nn.AdaptiveAvgPool2d((1,1)),
                   nn.Flatten())

net = nn.Sequential(b1,b2,b3,b4,b5,nn.Linear(1024,10))
X = torch.rand(size=(1, 1, 96, 96))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)

