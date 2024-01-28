import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore")
batch_size = 100
#MNIST dataset
train_dataset = dsets.MNIST(root='/pymnist',                      #选择数据的根目录
                            train=True,                           #选择数据的训练集
                            transform=transforms.ToTensor(),      #转换为Tensor变量
                            download=True)                        #从网上下载图片

test_dataset  = dsets.MNIST(root='/pymnist',                      #选择数据的根目录
                            train=False,                          #选择数据的测试集
                            transform=transforms.ToTensor(),      #转换为Tensor张量
                            download=True)                        #从网上下载图片
#加载数据
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, #使用批量数据
                                           shuffle=True)          #将数据进行打乱
test_loader  = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

#原始数据如下
print('train_data:',train_dataset.train_data.size())
print('train_labels:',train_dataset.train_labels.size())
print('test_data:',test_dataset.test_data.size())
print('test_labels:',test_dataset.test_labels.size())
#数据打乱小批次
print('小批次的尺寸：',train_loader.batch_size)
print('load_train_data:',train_loader.dataset.train_data.shape)
print('load_train_labels:' ,train_loader.dataset.train_labels.shape)


#定义神经网络模型
input_size = 784 #像素为 28 * 28           #输入为784个数据
hidden_size = 500
num_classes = 10                          # 输出为 10 个类别分别对应 0 - 9

#创建神经网络模型
class Neural_net(nn.Module):
    #初始化函数,接受自定义的输入特征维数，隐含层特征维数以及输出特征维数
    def __init__(self,input_num,hidden_size,out_put):
        super(Neural_net, self).__init__()
        self.layer1 = nn.Linear(input_num,hidden_size) #从输入到隐藏层的线性处理
        self.layer2 = nn.Linear(hidden_size,out_put)   #从隐藏层到输出层的线性处理
    def forward(self,x):
        out = self.layer1(x)     #输入层到隐藏层的线性计算
        out = torch.relu(out)    #隐藏层被激活
        out = self.layer2(out)   #输出层，注意，输出层直接被loss
        return out

net = Neural_net(input_num=input_size,hidden_size=hidden_size,out_put=num_classes)
print(net)

#开始进行模型的训练过程
# optimization
from torch.autograd import Variable
learning_rate = 1e-1
num_epoch = 20
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr = learning_rate)  # 使用随机梯度下降法
for epoch in range(num_epoch):
    print('current epoch = %d ' % epoch)
    for i,(images,labels) in enumerate(train_loader): #利用 enumerate 取出一个可迭代对象得内容
        '''
        (tensor变成variable之后才能进行反向传播求梯度。用变量.backward()进行反向传播之后,var.grad中保存了var的梯度)
        '''
        images = Variable(images.view(-1,28 * 28))    # view() 相当于 numpy.reshape 的作用，重新定义矩阵的形状
        labels = Variable(labels)

        outputs = net(images)                     #将数据集传入网络做前向计算
        loss = criterion(outputs,labels)          #计算loss
        optimizer.zero_grad()                     #在做反向传播之前先清除下网络得状态
        loss.backward()                           #lOSS 反向传播
        optimizer.step()                          #更新参数
        if i % 100 == 0:
            print('current loss = %.5f ' % loss.item())
print('finished training')

# 测试集准确度测试
total = 0
correct = 0
for images,labels in test_loader:
    images = Variable(images.view(-1 , 28 * 28))
    output = net(images)
    _ , predicets = torch.max(output.data,1)         #第一个是值的张量，第二个是序号的张量 (每一张图片都会给出所有的类别，然后计算出概率最大的类别)
    total = total + labels.size(0)                   #累计一共有多少数据量
    correct = correct + (predicets == labels).sum()
print('Accuracy = %.2f' % (100 * correct / total))












