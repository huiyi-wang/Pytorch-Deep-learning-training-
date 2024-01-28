import torch
import pandas as pd
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import math

'填充操作'
print('下面开始进行填充操作')
def comp_conv2d(conv2d,X):
    X = X.reshape((1,1)+X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])

# kernel_size:表示卷积核的大小, padding:表示使用1来填充,输入和输出的通道数都为1
conv2d = nn.Conv2d(1,1,kernel_size=3,padding=1) #函数功能：实现填充。并且实现卷积操作
X = torch.rand(size=(8,8))
print(comp_conv2d(conv2d,X).shape)
print('*'*100)
'步幅操作'
print('下面进行步幅操作')
#将高度和宽度的步幅都设置为2
print('将高度和宽度的步幅都设置为2')
conv2d1 = nn.Conv2d(1 , 1  ,kernel_size=3,padding=1,stride=2)
comp_conv2d(conv2d1,X)
print(comp_conv2d(conv2d1,X).shape)
#一个稍微复杂的例子
print('一个稍微复杂的例子')
conv2d2 = nn.Conv2d(1,1,kernel_size=(3,5),padding=(0,1),stride=(3,4))
print(comp_conv2d(conv2d2,X).shape)




