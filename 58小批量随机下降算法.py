'matplotlib inline'
import numpy as np
import torch
from torch import nn
import time

A = torch.zeros(256,256)
B = torch.randn(256,256)
C = torch.randn(256,256)
#逐个元素计算A = BC
start1 = time.time()
for i in range(256):
    for j in range(256):
        A[i,j] = torch.dot(B[i,:],C[:,j])
end1 = time.time()
time1 = end1 - start1
print('time1',time1)
#逐列计算A = BC
start2 = time.time()
for j in range(256):
    A[:,j] = torch.mv(B,C[:,j])
end2 = time.time()
time2 = end2 - start2
print('time2',time2)
#一次性计算A = BC
start3 = time.time()
A = torch.mm(B,C)
end3 = time.time()
time3 = end3 - start3
print('time3',time3)

def sgd(params,states,hyperparams):
    for p in params:
        p.data.sub_(hyperparams['lr'] * p.grad)
        p.grad.data.zero_()
def train_ch11(trainer_fn,states,hyerparams,data_iter,feature_dim,num_epochs = 2):
    #初始化模型
    w = torch.normal(mean=0.0,std=0.01,size=(feature_dim,1),requests_grad=True)
    b = torch.zeros((1),requires_grad=True)
    return


