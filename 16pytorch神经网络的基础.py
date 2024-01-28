import pandas as pd
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F

'''
5.1代码练习：神经网络的层和块
'''
print('5.代码练习：神经网络的层和块')
'''
第一节：使用多层感知机模拟实现一个小网络
'''
net = nn.Sequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
X = torch.rand(2,20)
net(X)
print('第一节：使用多层感知机模拟实现一个小网络')
print(net(X))
print('*'*100)
'''
第二节：自定义一个块
'''
class MLP(nn.Module):
    #用模型参数来申明层。这里我们声明了两个全连接层
    def __init__(self):
        #调用MLP的父类Modele的构造函数来执行必要的初始化。
        #这样，在类实例化时，也可以指定其他函数，例如模型参数params（之后在进行介绍）
        super().__init__()
        self.hidden = nn.Linear(20,256)    #隐含层数
        self.out = nn.Linear(256,10)       #输出层数

    #定义模型的前向传播：即如何根据输入x返回所需要的模型输出
    def forward(self, X):
        #注意，这里我们使用ReLU的函数版本，其在nn.functional模块中去定义
        return self.out(F.relu(self.hidden(X)))

Net = MLP()
print('第二节：自定义一个块')
print(Net(X))
print('*'*100)

'''
第三部分：顺序块
#我们可以更仔细地理解Sequential类是如何工作的， Sequential的设计是为了把其他模块串起来。
'''
class Mysequential(nn.Module):
    def __init__(self,*args):
        super().__init__()
        for idx,module in enumerate(args):
            '''
            dix表示数组args的索引
            module表示第idx个网络模型
            '''
            #这里，module 是 Module 子类的一个实例，我们把它保存在'Module'类的成员
            # _modules的主要优点是： 在模块的参数初始化过程中， 系统知道在_modules字典中查找需要初始化参数的子块。
            # 这里 _modules中。 _module的类型是OrderedDict
            self._modules[str(idx)] = module
    def forward(self,X):
        # OrderedDict 保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X

net11 = Mysequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
net11(X)
print('顺序块')
print(net11(X))
print('*'*100)
'''
第四部分：在前向传播函数中执行代码
'''
class FixHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        #不计算梯度的随机权重参数，因此其在训练期间保持不变
        self.rand_weight = torch.rand((20,20),requires_grad = False)
        self.linear = nn.Linear(20,20)

    def forward(self,X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu函数和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1 )
        # 复用全连接层，这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X = X/2
        return X.sum()

print('第四部分：在前向传播函数中执行代码')

net22 = FixHiddenMLP()
net22(X)
print(net22(X))
print('*'*100)

'''
第五部分：混合搭配各种组合块的方法。进行块的嵌套。
'''
##多个网络进行不同层次的嵌套工作，并进行结果的输出的工作
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        self.linear = nn.Linear(32,16)

    def forward(self,X):
        return self.linear(self.net(X))
chimera = nn.Sequential(NestMLP(),nn.Linear(16,20),FixHiddenMLP())
chimera(X)
print('*'*100)
print(chimera(X))
print('*'*200)

'''
5.2代码练习：神经网络的参数管理
'''
print('5.2代码练习：神经网络的参数管理')
'''
第一部分：具有单隐藏层的多层感知机
'''
import torch
from torch import nn
net12 = nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,1))
XY = torch.rand(size=(2,4))
print('第一部分：具有单隐藏层的多层感知机')
# 下面的代码从第二个全连接层（即第三个神经网络层）提取偏置， 提取后返回的是一个参数类实例，并进一步访问该参数的值。
print(net12(XY))
print(len(net12))
print(net12[2].state_dict())
'''
第一部分：目标参数
'''
print('第一部分：目标参数')
print(type(net12[2].bias))
print(net12[2].bias)
print(net12[2].bias.data)
print(net12[2].weight.grad)
print('*'*100)
'''
第二部分：一次性访问所有的参数
'''
print('第二部分；一次性访问所有的参数')
print(*[(name,param.shape) for name,param in net12[0].named_parameters()])
print(*[(name,param.shape) for name,param in net12.named_parameters()])
print('使用遍历进行参数读取')
for name ,param in net12.named_parameters():
    print(name)
    print(param)
print('*'*100)
'''
第四部分：从嵌套块中收集参数
'''
print('第四部分：从嵌套块中收集参数')
def block1():
    return nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,4),nn.ReLU())
def block2():
    net = nn.Sequential()
    for i in range(4):
        #这里开始嵌套
        net.add_module(f'block{i}',block1())
    return net

rgnet = nn.Sequential(block2(),nn.Linear(4,1))
rgnet(XY)
print(rgnet(XY))
print(rgnet)

'''
第五部分：参数初始化
'''
print('第五部分：参数初始化')
#首先调用内置的初始化器
def init_normal(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight , mean = 0 , std = 0.01)
        nn.init.zeros_(m.bias)
net12.apply(init_normal)#apply的含义就是将所有 net 的模组去调用init_normal函数
print('使用内置函数进行参数初始化')
print(net12[2].weight.data[0])
print(net12[0].bias.data[0])
#将所有的参数初始化为给定的常数，比如初始化为1
def init_constant(m):
    if type(m)  ==nn.Linear:
        nn.init.constant_(m.weight,1)#nn.init.constant_()功能:使用val的值来填充输入的Tensor
        nn.init.zeros_(m.bias)
print('使用给定的常数进行参数的初始化')
net12.apply(init_constant)
print(net12[0].weight.data[0])
print(net12[0].bias.data[0])

#对某些块应用不同的初始化方法
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight,42)
net12[0].apply(init_xavier)
net12[2].apply(init_42)
print('对某些块应用不同的初始化方法')
print('第一层的参数权重')
print(net12[0].weight.data[0])
print('第二层的参数权重')
print(net12[2].weight.data)
#自定义初始化
def my_init(m):
    if type(m) == nn.Linear:
        print('Init',*[(name,param.shape) for name ,param in m.named_parameters()][0])
        nn.init.uniform_(m.weight,-10,10)
        m.weight.data *= m.weight.data.abs() >= 5  #保留绝对值大于等于5的权重，如果不符合条件的话，就将其设置为0
print('自定义初始化')
net12.apply(my_init)
print(net12[0].weight)
#参数绑定
print('参数绑定')
shared = nn.Linear(8,8)
net32 = nn.Sequential(nn.Linear(4,8),nn.ReLU(),shared,nn.ReLU(),shared,nn.ReLU(),nn.Linear(8,1))
print(net32)
#检查参数是否相同
print(net32[2].weight.data[0] == net32[4].weight.data[0])
net32[2].weight.data[0,0] = 100
print(net32[2].weight.data[0] == net32[4].weight.data[0])
print('*'*200)
'''
5.3延后初始化
'''
'''
5.4 自定义层
'''
print('5,4 自定义层')
'''
第一部分:定义一个不带参数的层
'''
print('第一部分：定义一个不带参数的层')
import torch
from torch import nn
import torch.nn.functional as F
class CenteredLayer(nn.Module):
    def __init__(self):
        super(CenteredLayer, self).__init__()
    def forward(self,X):
        return X-X.mean()
layer = CenteredLayer()
print(layer(torch.FloatTensor([1,2,3,4,5])))

nettt = nn.Sequential(nn.Linear(8,128),CenteredLayer())
YYY = nettt(torch.rand(4,8))
print(YYY.mean())
print(('*'*100))
'''
第二部分：带参数的层
'''
print('第二部分：带参数的层')
class Mylinear(nn.Module):
    def __init__(self,in_units,units):
        super(Mylinear, self).__init__()
        self.weight = nn.Parameter(torch.rand(in_units,units))
        self.bias   = nn.Parameter(torch.rand(units,))
    def forward(self,X):
        linear = torch.matmul(X,self.weight.data) + self.bias.data
        return F.relu(linear)

linear = Mylinear(5,3)
print(linear.weight)
a = linear(torch.rand(2,5))
print(a)
net1111 = nn.Sequential(Mylinear(64,8),Mylinear(8,1))
net1111(torch.rand(2,64))
print(net1111(torch.rand(2,64)))
print(net1111)

