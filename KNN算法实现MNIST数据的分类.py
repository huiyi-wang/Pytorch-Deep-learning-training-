import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import warnings
import matplotlib.pyplot as plt
import numpy as np
import operator
warnings.filterwarnings("ignore")
import os

def KNN_classify(k,dis,X_train,Y_train,X_test):
    # assert dis == 'E' or dis == 'M'   'dis must E or M ; E代表欧式距离，M代表曼哈顿距离'
    num_test = X_test.shape[0]   #表示测试数据集的个数
    labellist = []
    '''
    使用欧式距离公式作为距离度量
    '''
    if (dis == 'E'):
        for i in range(num_test):
            #实现欧式距离的计算公式
            distances = np.sqrt(np.sum(((X_train - np.tile(X_test[i],(X_train.shape[0],1))) ** 2) , axis=1))
            nearest_k = np.argsort(distances) #距离由小到大进行排序，并返回 index 值
            topk = nearest_k[:k] # 选取前 K 个距离的点
            classCount = {}
            for j in topk:       # 依次遍历这 K 个点
                classCount[Y_train[j]] = classCount.get(Y_train[j],0) + 1
            # operator.itemgetter(1) 表示取出对象维度为 1 的数据
            sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
            labellist.append(sortedClassCount[0][0])
        return np.array(labellist)
    '''
    使用曼哈顿距离公式作为距离度量
    '''
    if (dis == 'M'):
        for i in range(num_test):
            #实现曼哈顿距离的计算公式
            distances = np.sum(np.abs(X_train - np.tile(X_test[i],(X_train.shape[0],1))),axis=1)
            nearset_k = np.argsort(distances)  #距离由大到小进行排序，并返回 index 值
            topk = nearset_k[:k] # 选取前 K 个距离的点
            '''
            下面开始进行类别计数，并对类别号进行排序，输出类别数目最多的作为该点的类别
            '''
            classCount = {}  # 定义了一个字典
            for j in topk:       # 依次遍历这 K 个点
                # 注意此处的 get 函数，因为没有设置 Y_train[j] 这个键的值，因此默认输出为 0
                classCount[Y_train[j]] = classCount.get(Y_train[j],0) + 1  #类别计算器，用来计算类别
            sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)  #对类别计数器进行排序，选择类别最多的一个作为该测试集的类别
            labellist.append(sortedClassCount[0][0])
        return np.array(labellist)


batch_size = 100
# MNIST dataset
train_dataset = dsets.MNIST(root='/m1/pymnist',      #选择数据的根目录
                            train=True,              #选择训练集
                            transform=None,          #不考虑使用任何数据预处理
                            download=True)           #从网络上下载图片

test_dataset  = dsets.MNIST(root='/m1/pymnist',      #选择数据的根目录
                            train=False,             #选择测试集
                            transform=None,          #不考虑使用任何数据预处理
                            download=True)           #从网络上下载图片

def standardization(data):#标准化
    mean = np.mean(data)   #均值
    sigma = np.std(data)   #方差
    return (data - mean)/sigma

#加载数据集
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)           #将数据集打乱重组
test_loader  = torch.utils.data.DataLoader(dataset=test_dataset ,
                                           batch_size=batch_size,
                                           shuffle=True)           #将数据集打乱重组

print('train_data:',train_dataset.train_data.size())
print('train_labels:',train_dataset.train_labels.size())
print('test_data:',test_dataset.test_data.size())
print('test_labels:',test_dataset.test_labels.size())

# digit = train_loader.dataset.train_data[1]
# plt.imshow(digit,cmap=plt.cm.binary)
# plt.show()
# print(train_loader.dataset.train_labels[1])

#验证 KNN 在 MNIST 上的效果
X_train = train_loader.dataset.train_data.numpy() #需要转为numpy矩阵
X_train = standardization(X_train)
X_train = X_train.reshape(X_train.shape[0],28*28) #需要reshape后才可以放入 KNN 分类器
# print(np.shape(X_train))
y_train = train_loader.dataset.train_labels.numpy()
X_test  = test_loader.dataset.test_data[:1000].numpy()
X_test  = standardization(X_test)
X_test  = X_test.reshape(X_test.shape[0],28*28)
y_test  = test_loader.dataset.test_labels[:1000].numpy()
num_test = y_test.shape[0]
y_test_pred = KNN_classify(5,'M',X_train,y_train,X_test)
num_correct = np.sum(y_test_pred==y_test)
accuracy = float(num_correct) / num_test
print('KNN实现MNIST数据集的正确率为'+str(accuracy*100)+'%')








#实现 Knn 算法的封装
class Knn():
    def __init__(self,X_train,y_train):    # X_train 代表的是训练数据集，y_train 表示测试数集
        self.X_train = X_train
        self.y_train = y_train
    def predict(self,k,dis,X_test):
        num_test = X_test.shape[0]
        labellist = []
        # 使用欧式距离作为距离度量公式
        if (dis == 'E'):
            for i in range(num_test):
                distances = np.sqrt(np.sum(((self.X_train - np.file( X_test[i],(self.X_train.shape[0],1)))**2),axis=1))
                nearlist_K = np.argsort(distances)
                top_K  = nearlist_K[:k]
                classCounter = {}
                for j in top_K:
                    classCounter[self.y_train[j]] = classCounter.get(self.y_train[j],0) + 1
                    sortedclassCounter = sorted(classCounter.items(),key=operator.itemgetter(1),reverse=True)
                    labellist.append(sortedclassCounter[0][0])
            return np.array(labellist)
        # 使用曼哈顿距离作为距离度量公式
        if (dis == 'M'):
            for i in range(num_test):
                distances = np.sum(np.abs(self.X_train - np.file( X_test[i],(self.X_train.shape[0],1))),axis=1)
                nearlist_K = np.argsort(distances)
                top_K = nearlist_K[:k]
                classCounter = {}
                for j in top_K:
                    classCounter[self.y_train[j]] = classCounter.get(self.y_train[j],0) + 1
                    sortedclassCounter =sorted(classCounter.items(),key=operator.itemgetter(1),reverse=True)
                    labellist.append(sortedclassCounter[0][0])
            return np.array(labellist)




