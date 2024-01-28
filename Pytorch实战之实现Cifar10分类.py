import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import warnings
import numpy as np
# CIFAR10 dataset
train_dataset = dsets.CIFAR10(root='/ml/pycifar',               #选择数据的根目录
                              train=True,                       #选择训练集
                              transform=transforms.ToTensor(),  #转换为Tensor变量
                              download=True)                    #从网上下载数据集
test_dataset  = dsets.CIFAR10(root='ml/pycifar',                #选择数据的根目录
                              train=False,                      #选择测试集
                              transform=transforms.ToTensor(),  #转换为Tensor变量
                              download=True)                    #从网上下载数据集
#加载数据
batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader  = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
#神经网络定义
class Net(nn.Module):
    def __init__(self,input_size,hidden_size1,hidden_size2,num_classes):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.num_classes  = num_classes
        self.layer1 = nn.Linear(self.input_size,self.hidden_size1)
        self.layer2 = nn.Linear(self.hidden_size1,self.hidden_size2)
        self.layer3 = nn.Linear(self.hidden_size2,self.num_classes)
    def forward(self,x):
        out = torch.relu(self.layer1(x))
        out = torch.relu(self.layer2(out))
        out = self.layer3(out)
        return  out
input_size = 3072
hidden_size1 = 500
hidden_size2 = 200
num_classes  = 10
num_epoch = 30
batch_size = 100
learning_rate = 1e-1
net = Net(input_size = input_size,
          hidden_size1 = hidden_size1,
          hidden_size2 = hidden_size2,
          num_classes = num_classes)

# print(net)
# print(net.parameters())
# paras = list(net.parameters())
# for num,para in enumerate(paras):
#     print('number:',num)
#     print(para)
#     print('_____________________________')
#训练模型
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr = learning_rate)
for epoch in range(num_epoch):
    print('current epoch = %d ' % epoch)
    for i ,(images,labels) in enumerate(train_loader): #利用 enumerate 取出一个可迭代对象的内容
        images = Variable(images.view(images.size(0), -1))
        labels = Variable(labels)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('current loss = %.5f' % loss.item())
print('Finish training')
#测试集准确度测试
total = 0
correct = 0
for images ,labels in test_loader:
    images = Variable(images.view(images.shape[0],-1))
    outputs = net(images)
    _ , predicts = torch.max(outputs.data,1)
    total = total + labels.size(0)
    correct = correct + (predicts == labels).sum()
print('Accuracy = %.2f ' % (100 * correct / total))