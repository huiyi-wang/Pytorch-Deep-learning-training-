import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import operator
import warnings
warnings.filterwarnings('ignore')

batch_size = 100
# Cifar10 dataset
train_dataset = dsets.CIFAR10(root='/ml/pycifar',        # 选择数据的根目录
                              train=True,                # 选择训练集
                              download=True)             # 从网上下载图片

test_dataset  = dsets.CIFAR10(root='/ml/pycifar',        # 选择数据的根目录
                              train=False,               # 选择测试集
                              download=True)             # 从网上下载图片

#加载数据集
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader  = torch.utils.data.DataLoader(dataset=test_dataset ,batch_size=batch_size,shuffle=True)

classes = ('plane','car','bird','cat','deer','dog','flog','horse','ship','truck')
digit = train_loader.dataset.data[20000]
# plt.imshow(digit,cmap=plt.cm.binary)
# plt.show()
# print(classes[train_loader.dataset.targets[20000]])

def getmean(X_train):
    X_train = np.reshape(X_train,(X_train.shape[0],-1))     # 将图片从二维展开为一维
    mean_image = np.mean(X_train,axis=0)                    # 求解出训练集中所有图片每个像素位置上的平均值
    return mean_image

def centralized(X_test,mean_image):
    X_test = np.reshape(X_test,(X_test.shape[0],-1))        # 将图片从二维展开为一维
    X_test = X_test.astype(np.float)                        # 将数据类型转为浮点型
    X_test = X_test - mean_image                            # 减去均值图像，实现零均值化
    return X_test

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


X_train = train_loader.dataset.data
mean_image1 = getmean(X_train)
X_train = centralized(X_train,mean_image1)
y_train = train_loader.dataset.targets

X_test  = test_loader.dataset.data[:100]
mean_image2 = getmean(X_test)
X_test  = centralized(X_test,mean_image2)
y_test  = test_loader.dataset.targets[:100]

num_test = len(y_test)
y_test_pred = KNN_classify(6,'E',X_train,y_train,X_test)  #使用没有封装好的类
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct)/num_test
print('KNN实现Cifar10数据分类得精确度为'+str(accuracy*100)+'%')

