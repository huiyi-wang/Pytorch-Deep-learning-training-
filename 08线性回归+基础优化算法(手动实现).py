##线性回归模型
import numpy as np
import torch
import pandas as pd
from  matplotlib import pyplot as plt
import random

###1.人为构造数据集
def synthetic_data(w,b,num_examples):
    " 生成 y = Xw + b +噪声 "
    X = torch.normal(0,1,size = (num_examples,len(w)))
    Y = torch.matmul(X,w) + b
    Y = Y + torch.normal(0,0.01,size = Y.shape)
    return X , Y.reshape((-1,1))

###2.定义一个data_iter函数，该函数接收批量大特征矩阵和标签向量作为输入，生成大小为batch_size的小批量
def data_iter(batch_size,features,labels):
    num_examples =  len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)#表示这些样本是随机读取的，没有固定的顺序
    for i in range(0 , num_examples , batch_size):
        batch_indices = torch.tensor(  indices[i:min(i + batch_size,num_examples)] )
        yield features[batch_indices] , labels[batch_indices]

###3.定义模型
def linreg(X,w,b):
    "线性回归模型"
    return torch.matmul(X,w) + b

###4.定义损失函数
def squared_loss(y_hat,y):
    "均方根误差"
    return (y_hat-y.reshape(y_hat.shape))**2 / 2

###5、定义优化算法
def sgd(params,lr,batch_size):
    "小批量随机梯度下降"
    with torch.no_grad():
        for param in params:
            ##注意这里千万不能写成
            # param = param - lr * param.grad /batch_size
            # 猜测是因为默认的tensor变量的requires_grad的值默认为False，不对梯度进行跟踪
            param -=  lr * param.grad /batch_size
            param.grad.zero_()



true_w = torch.tensor([2,-3.4])
true_b = 4.2
features,labels = synthetic_data(true_w,true_b,1000)

##1、画图显示一下
plt.plot(features[:,1].numpy(),labels.numpy(),'o')
plt.show()
##2、
batch_size = 10
##测试代码
for x , y in data_iter(batch_size,features,labels):
    print(x,'\n',y)


##3、定义：模型初始化参数
w = torch.normal(0,0.01,size = (2,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)

##4、训练的过程
lr = 0.4
num_epochs = 100
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for x , y in data_iter(batch_size,features,labels):
        l = loss(net(x,w,b),y)
        l.sum().backward()
        sgd([w,b],lr,batch_size)

    ##计算已经完成，开始输出一下迭代的损失
    with torch.no_grad():
        train_l = loss(net(features,w,b),labels)
        print('epoch %d, loss %f' %(epoch + 1,train_l.mean().numpy()))



