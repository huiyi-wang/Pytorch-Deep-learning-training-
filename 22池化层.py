import torch
import pandas as pd
import numpy as np
from torch import nn
'''
最大值池化和均值池化
'''
def pool2d(X,pool_size,mode='max'):
    p_h , p_w = pool_size
    Y = torch.zeros((X.shape[0]-p_h+1,X.shape[1]-p_w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode =='max':
                Y[i,j] = X[i : i + p_h , j : j + p_w].max()
            elif mode == 'avg':
                Y[i,j] = X[i : i + p_h , j : j + p_w].mean()
    return Y

X = torch.tensor([[0.0,1.0,2.0],[3.0,4.0,5.0],[6.0,7.0,8.0]])
print(pool2d(X,(2,2),'max'))
print(pool2d(X,(2,2),'avg'))

'''
填充和步幅
'''
Z = torch.arange(16,dtype=torch.float32).reshape((1,1,4,4))
print(Z)
##深度学习框架中的步幅与池化窗口的大小相同
pool2d1 = nn.MaxPool2d(3)
print(pool2d1(Z))

##手动设置填充和步幅
pool2d1 = nn.MaxPool2d(3,padding=1,stride=2)
print(pool2d1(Z))

##池化层在每个通道上单独运算
ZZ = torch.cat((Z , Z + 1),1)
print(ZZ)
pool2d2 = nn.MaxPool2d(3,padding=1,stride=2)
print(pool2d2(ZZ))