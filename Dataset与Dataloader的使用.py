import numpy as np
import pandas as pd
import torch
from torch import tensor
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import warnings
warnings.filterwarnings("ignore")

'''
深度学习布置
    1、Prepare dataset
        tools：Dataset and Dataloader
    2、Design model using Class
        inherit from nn.Module
    3、Contruct loss and otimizer
        using Pytorch API
    4、Training cycle
        foward，backward，update
'''
class MyDataset(Dataset):
    '''
    处理数据的两种方法：
        1、All Data load to Memory(结构化数据)
        2、定义一个列表，把每一个sample路径放到一个列表，标签放到另一个列表里面，避免一次性全部加载到内存中（非结构化数据）
    '''
    def __init__(self,filepath):
        '''
        :param filepath:读取数据集的路径
        '''
        data_train = np.loadtxt(filepath ,dtype=np.float32)
        self.len = data_train.shape[0]
        self.train_x = torch.from_numpy(data_train[:,:-1])
        self.train_y = torch.from_numpy(data_train[:,[-1]])
        # print('数据已经准备好了')
    def __getitem__(self, index): #为了支持下标操作，即索引dataset[index]
        return self.train_x[index] , self.train_y[index]
    def __len__(self): #为了方便使用len(dataset)
        return  self.len

file = '深度学习训练数据集.txt'
mydataset = MyDataset(file)
train_loader =DataLoader(dataset=mydataset,batch_size=32,shuffle=True,num_workers=4)#dataloader一次性创建num_worker个工作进程

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(24,24)
        self.linear2 = nn.Linear(24,12)
        self.linear3 = nn.Linear(12,1)
        self.sigmod  = nn.Sigmoid()
    def forward(self,x):
        x = self.sigmod(self.linear1(x))
        x = self.sigmod(self.linear2(x))
        x = self.sigmod(self.linear3(x))
        return x


model = Model()
criterion = nn.BCELoss(size_average=True)               #损失函数
optimizer =torch.optim.SGD(model.parameters(),lr = 0.1) #优化器
paras = list(model.parameters())
for num,para in enumerate(paras):
    print('number:',num)
    print(para)
    print('_____________________________')
##开始进行函数的迭代训练
if __name__ == "__main__":#防止在被其他文件导入时显示多余的程序主体部分。
    for epoch in range(10):
        for i,data in enumerate(train_loader,0):
            # 1、准备数据
            inputs,labels = data
            # 2、前向传播
            y_pred =model(inputs)
            loss = criterion(y_pred,labels)
            print('epoch'+str(epoch),'i'+str(i),loss.item())
            # 3、后向传播
            optimizer.zero_grad()
            loss.backward()
            # 4、更新
            optimizer.step()






