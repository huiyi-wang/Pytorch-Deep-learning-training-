import pandas as pd
import numpy as np
import torch

A = torch.arange(20).reshape(4,5)
print('原矩阵',A)
print('转置矩阵',A.T)

B = torch.tensor([[1,2,3],[2,0,4],[3,4,5]])
print(B==B.T)

X = torch.arange(24).reshape(2,3,4)
print(X)

print(A+A)
print(A*A)


a = 2
print(X+a)



xx = torch.arange(4,dtype=torch.float32)
print(xx)
print(xx.sum())
print(xx.shape)

Aa = torch.arange(2*20,dtype=torch.float32).reshape(2,5,4)
print(Aa)

print(Aa.sum())
Aa_sum_axis0 = Aa.sum(axis=0)
print(Aa_sum_axis0)
Aa_sum_axis01= Aa.sum(axis=[0,1])
print(Aa_sum_axis01)
print(Aa.mean())



