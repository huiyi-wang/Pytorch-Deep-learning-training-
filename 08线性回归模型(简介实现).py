import torch
import pandas as pd
import numpy as np
from torch.utils import data
from torch import nn   #nn是神经网络的缩写
import random

#1.人为构造数据集
def synthetic_data(w,b,num_examples):
    " 生成 y = Xw + b +噪声 "
    X = torch.normal(0,1,size = (num_examples,len(w)))  #随机生成自变量X
    Y = torch.matmul(X,w) + b                           #生成应变量y
    Y = Y + torch.normal(0,0.01,size = Y.shape)
    return X , Y.reshape((-1,1))
true_w = torch.tensor([2,3.4])
true_b = 4.2
features ,labels = synthetic_data(true_w,true_b,1000)

##2.定义一个data_iter函数，该函数接收批量大特征矩阵和标签向量作为输入，生成大小为batch_size的小批量
def load_array(data_arrays, batch_size, is_train=True):
    "构建一个pytorch数据迭代器"
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size ,shuffle=is_train)
batch_size = 10
data_iter = load_array((features,labels),batch_size)


net = nn.Sequential(nn.Linear(2,1)) #使用框架的预定义好的层

net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)
loss = nn.MSELoss()#损失函数

#调用优化函数
trainer = torch.optim.SGD(net.parameters(),lr=0.03)


##训练过程
num_epoch = 3
for epoch in range(num_epoch):
    for x,y in data_iter:
        l = loss(net(x),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features),labels)
    print(f'epoch{epoch+1},loss{l:f}')

