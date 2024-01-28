import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
from IPython import display

'构建LeNet网络'
net = nn.Sequential(
      nn.Conv2d( 1 , 6 , kernel_size = 5 , padding = 2 ) , nn.Sigmoid(),
      nn.AvgPool2d( kernel_size = 2 , stride = 2),
      nn.Conv2d( 6 , 16 , kernel_size = 5) , nn.Sigmoid(),
      nn.AvgPool2d( kernel_size = 2 , stride = 2 ),
      nn.Flatten(),
      nn.Linear( 16 * 5 * 5 , 120 ) , nn.Sigmoid(),
      nn.Linear( 120 ,  84 ) , nn.Sigmoid(),
      nn.Linear( 84 , 10))

X = torch.rand(size=(1,1,28,28),dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)

'模型训练'
batch_size = 256







