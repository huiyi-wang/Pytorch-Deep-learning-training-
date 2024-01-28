import math

import numpy as np
import pandas as pd
import torch
from torch import nn
import d2l

max_degree = 20#多项式最大阶数
n_train = 100 #训练集大小
n_test = 100  #测试集大小
true_w = np.zeros(n_train+n_test).reshape(-1,n_train+n_test) #定义大量的空间
true_w[:,0:4] = np.array([5,1.2,-3.4,5.6])
features = np.random.normal(size=(n_train+n_test,1))
np.random.shuffle(features)
poly_features = np.power(features,np.array(max_degree).reshape(1,-1))
for i in range(max_degree):
    poly_features /= math.gamma(i+1)
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)   #添加一个噪声

true_w, features, poly_features, labels = [torch.tensor(x, dtype=torch.float32) for x in [true_w, features, poly_features, labels]]

print(features[:2])
print(poly_features[:2, :])
print(labels[:2])







