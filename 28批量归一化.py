import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import math
from matplotlib.pyplot import plot as plt

'''
批量归一化的具体从零实现
'''
print('批量归一化的从零实现')
def batch_norm(X,gamma,beta,moving_mean,moving_var,eps,momentum):
    '''
    :param X: 输入矩阵X
    :param gamma: 缩放参数
    :param beta:  移位参数
    :param moving_mean: 全局均值
    :param moving_var:  全局方差
    :param eps: 扰动
    :param momentum: 更新参数
    :return:
    '''
    #通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():
        #如果是在预测模式下直接使用传入的移动平均所得到的均值与方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2,4)
        #   2表示是全连接层归一化，4表示是卷积层归一化
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值与方差
            mean = X.mean(dim=0) # 按行求均值
            var  = ((X - mean) **2).mean(dim=0)
        else:
            #使用二维卷积层的情况，计算通道维上（axis = 1）的均值与方差
            #这里我们需要保持X的形状，以便之后可以做广播计算
            mean = X.mean(dim=(0,2,3),keepdim=True)
            var  = ((X - mean)**2).mean(dim=(0,2,3),keepdim=True)
        #训练模式下，用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值与方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean # 移动平均进行均值的更新
        moving_var  = momentum * moving_var  + (1.0 - momentum) * var  #移动平均进行方差的更新
    Y = gamma * X_hat + beta #缩放与移位
    return Y , moving_mean.data , moving_var.data


##创建一个正确的BatchNorm的层
class BatchNorm(nn.Module):
    def __init__(self,num_features,num_dims):
        '''
        :param num_features: 完全连接层的输出数量或卷积层的输出通道数
        :param num_dims:2表示完全连接层，4表示卷积层
        '''
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            shape = (1,num_features)
        else:
            shape = (1,num_features,1,1)
        #参与梯度和迭代的拉伸和偏移参数，分别初始化为1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta  = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var  = torch.ones(shape)

    def forward(self,X):
        #如果X不在内存上，将 moving_mean 和 moving_var 复制到 X 所在的显存上
        # if self.moving_mean.device != X.device:
        #     self.moving_mean = self.moving_mean.to(X.device)
        #     self.moving_var  = self.moving_var.to(X.device)
        # 保存更新过的 moving_mean 和 moving_var
        Y,  self.moving_mean,  self.moving_var = batch_norm(X, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9)
        return Y

##使用批量归一化层的LeNet
net1 =nn.Sequential(
     nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5) , BatchNorm(6,num_dims=4) , nn.Sigmoid(),
     nn.AvgPool2d(kernel_size=2,stride=2),
     nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5) , BatchNorm(16,num_dims=4) , nn.Sigmoid(),
     nn.AvgPool2d(kernel_size=2,stride=2),
     nn.Flatten(),
     nn.Linear(16*21*21,120), BatchNorm(120,num_dims=2) , nn.Sigmoid(),
     nn.Linear(120,84), BatchNorm(84,num_dims=2) , nn.Sigmoid(),
     nn.Linear(84,10))

X = torch.rand(size=(1, 1, 96, 96))
for layer in net1:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
print('*'*100)
'''
批量归一化的简介实现
'''
net2 = nn.Sequential(
       nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5) ,nn.BatchNorm2d(6), nn.Sigmoid(),
       nn.AvgPool2d(kernel_size=2,stride=2),
       nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5) ,nn.BatchNorm2d(16) ,nn.Sigmoid(),
       nn.AvgPool2d(kernel_size=2,stride=2) ,
       nn.Flatten(),
       nn.Linear(7056,120), nn.BatchNorm1d(120), nn.Sigmoid(),
       nn.Linear(120,84) ,nn.BatchNorm1d(84),nn.Sigmoid(),
       nn.Linear(84,10))


X = torch.rand(size=(1, 1, 96, 96))
for layer in net2:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)



