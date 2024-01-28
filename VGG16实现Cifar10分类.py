import os.path
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import math
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
'第一步：载入数据集'
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])  #将 [0，1] 归一化到 [-1,-1])
trainset = torchvision.datasets.CIFAR10(root='/ml/pycifar',    # 表示的是Cifar10 的数据存放目录
                                        train=True,                               # 选择训练集
                                        download=True,                            # 表示下载数据
                                        transform=transform)                      # 对数据进行操作

testset  = torchvision.datasets.CIFAR10(root='/ml/pycifar',    # 表示的是Cifar10 的数据存放目录
                                        train=False,                              # 选择测试集
                                        download=True,                            # 表示下载数据
                                        transform=transform)                      # 对数据进行操作

trainloader = torch.utils.data.DataLoader(trainset,batch_size = 100,shuffle = True,num_workers = 2)
testloader  = torch.utils.data.DataLoader(testset, batch_size = 100,shuffle = True,num_workers = 2)
cifar10_classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

# #查看训练数据
# if __name__ == '__main__':
#     dataiter = iter(trainloader)      #从训练数据中随机取出一些数据
#     images,labels = dataiter.next()
#     print(images.shape)
#     torchvision.utils.save_image(images[1],'test.jpg')
#     print(cifar10_classes[labels[1]])

'第二步：构建卷积神经网络'
cfg = {'VGG16':[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M']}
class VGG(nn.Module):
    def __init__(self,net_name):
        super(VGG, self).__init__()
        #构建网络的卷积层和池化层，最终输出命名为features，原因是通常认为经过这些操作的输出为包含图像空间信息的特征层
        self.features = self._make_layers(cfg[net_name])

        #构建卷积层之后的全连接层以及分类器
        self.classifier = nn.Sequential(nn.Dropout(),
                                        nn.Linear(512,512),    #fc1
                                        nn.ReLU(True),
                                        nn.Dropout(),
                                        nn.Linear(512,512),    #fc2
                                        nn.ReLU(True),
                                        nn.Linear(512,10))     #fc3            最终cifar10的输出为 10 类
    #初始化权重
    ###pytorch中的实现方法：(pytorch默认使用kaiming正态分布初始化卷积层参数。，所以不用自己去手动初始化，因此常被人所遗忘的知识点)权重是用的0均值高斯分布，偏置是0均值0方差的均匀分布。
    ## 进行权值初始化  如果不自己初始化，则使用的默认方法 init.kaiming_uniform_  0均值的正态分布
    # def __initialize_weight(self):
    #     for m in self.modules():  #作用：将整个模型的所有构成（包括包装层、单独的层、自定义层等）由浅入深依次遍历出来, 直到最深处的单层
    #         if isinstance(m,nn.Conv2d):   #表示如果卷积网络中的层为二维卷积层，就执行下面的操作
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0,math.sqrt(2./n))
    #             m.bias.data.zero_()

    def forward(self,x):
        #前向传播
        x = self.features(x)
        x = x.view(x.size(0),-1)   #卷积层到全连接层需要reshape
        x = self.classifier(x)
        return x

    def _make_layers(self,cfg):
        #构建 VGG 块
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
            else:
                layers.append(nn.Conv2d(in_channels,v,kernel_size=3,padding=1))
                layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(inplace=True))
                in_channels = v
        return nn.Sequential(*layers)

if __name__ == '__main__':
    net = VGG('VGG16')
    '第三步：定义损失函数和优化方法'
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),lr = 0.001,momentum=0.9)
    '第四步：卷积神经网络的训练'
    num_epoch = 1
    for epoch in range(num_epoch):
        train_loss = 0.0
        print('current epoch = ' + str(epoch+1))
        for batch_idx,data in enumerate(trainloader,0):
            #初始化
            inputs,labels =data          #获取数据
            optimizer.zero_grad()        #先将梯度设置为0
            #优化过程
            outputs = net(inputs)              #将数据输入网络，得到第一轮网络前向传播的预测结果outputs
            loss = criterion(outputs,labels)   #预测结果outputs和labels通过之前定义的交叉熵计算损失、
            loss.backward()                    #误差反向传播
            optimizer.step()                   #随机梯度下降方法（之前定义的）优化权重
            #查看网络的训练状态
            loss.item()
            if batch_idx % 100 == 0:       #每迭代2000个batch打印一次以查看当前网络的收敛情况
                print('[%d,%5d] loss: %.3f' % (epoch+1,batch_idx,loss.item()))
            state = {'net':net.state_dict(),'epoch':epoch+1}
    print('Finish Training')


    #测试集准确度测试
    total = 0
    correct = 0
    for images ,labels in testloader:
        images = Variable(images.view(images.shape[0],-1))
        outputs = net(images)
        _ , predicts = torch.max(outputs.data,1)
        total = total + labels.size(0)
        correct = correct + (predicts == labels).sum()
    print('Accuracy = %.2f ' % (100 * correct / total))













