import pandas as pd
import numpy as np
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from numpy import ndarray as nd
from d2l import torch as d2l
import d2l
from d2l import torch as d2l
from mxnet.gluon import data as gdata
import sys
import time
from torch.utils import data
######第一部分：图像分类数据集
trans = transforms.ToTensor()##将图片转换为一个Tensor数据集合
mnist_train = torchvision.datasets.FashionMNIST(root="../data",train=True,transform=trans,download=True)
mnist_test  = torchvision.datasets.FashionMNIST(root="../data",train=False,transform=trans,download=False)
# print(len(mnist_train))
# print(len(mnist_test))
# print(mnist_train[0][0].shape)

def get_fashion_labels(labels):
    "返回Fashion-MNIST数据集的文本标签"
    text_labels = ['t-shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']
    return [text_labels[int(i)] for i in labels]

#展示图片效果，及定义一个可以在一行里画出多张图像和对应标签的函数

def show_fashion_mnist(images,labels):
    d2l.use_svg_display()
    _,figs = d2l.plt.subplots(1,len(images),figsize=(12,12))
    for f , img , lbl in zip(figs,images,labels):
        f.imshow(img.reshape((28,28)).asnumpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)

X , y =mnist_train[0:9]
show_fashion_mnist(X , get_fashion_labels(y))









