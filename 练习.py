import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import warnings
import operator
import torchvision.datasets as dsets
warnings.filterwarnings('ignore')
'''
第一步：构建KNN分类器
'''
class KNN:
    def __init__(self):
        pass
    def fit(self,X_train,Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self,k,dis,X_test):
        num_test = X_test.shape[0]
        labeslist = []
        if dis == 'E':
            for i in range(num_test):
                distances = np.sqrt(np.sum(((self.X_train - np.tile(X_test[i],(self.X_train.shape[0],1))))**2,axis=1))
                nearlist = np.argsort(distances)
                top_k = nearlist[:k]
                classCounter = {}
                for j in top_k:
                    classCounter[self.Y_train[j]] = classCounter.get((self.Y_train[j]),0) + 1
                    sortedclassCounter = sorted(classCounter.items(),key=operator.itemgetter(1),reverse=True)
                    labeslist.append(sortedclassCounter[0][0])
            return np.array(labeslist)

        if dis == 'M':
            for i in range(num_test):
                distances = np.sum(np.abs((self.X_train - np.tile(X_test[i],(self.X_train.shape[0],1)))),axis=1)
                nearlist = np.argsort(distances)
                top_k = nearlist[:k]
                classCounter = {}
                for j in top_k:
                    classCounter[self.Y_train[j]] = classCounter.get((self.Y_train[j]),0) + 1
                    sortedclassCounter = sorted(classCounter.items(),key=operator.itemgetter(1),reverse=True)
                    labeslist.append(sortedclassCounter[0][0])
            return np.array(labeslist)

def getmean(X_train):
    X_train = np.reshape(X_train,(X_train.shape[0],-1))
    mean_image = np.mean(X_train,axis=0)
    return mean_image

def centralized(X_test,mean_image):
    X_test = np.reshape(X_test,(X_test.shape[0],-1))        # 将图片从二维展开为一维
    X_test = X_test.astype(np.float)                        # 将数据类型转为浮点型
    X_test = X_test - mean_image                            # 减去均值图像，实现零均值化
    return X_test

'''
第二步：数据准备
'''
batch_size = 100
# Cifar10 dataset
train_dataset = dsets.CIFAR10(root='/ml/pycifar',        # 选择数据的根目录
                              train=True,                # 选择训练集
                              download=True)             # 从网上下载图片

test_dataset  = dsets.CIFAR10(root='/ml/pycifar',        # 选择数据的根目录
                              train=False,               # 选择测试集
                              download=True)             # 从网上下载图片

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader  = torch.utils.data.DataLoader(dataset=test_dataset ,batch_size=batch_size,shuffle=True)
X_train = train_loader.dataset.data
X_train = X_train.reshape(X_train.shape[0],-1)
mean_image = getmean(X_train)
X_train = centralized(X_train,mean_image)

y_train = train_loader.dataset.targets
y_train = np.array(y_train)

X_test = test_loader.dataset.data[:100]
X_test = X_test.reshape(X_test.shape[0],-1)
X_test = centralized(X_test,mean_image)

y_test = test_loader.dataset.targets[:100]
y_test = np.array(y_test)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

'''
第三部分：实现K折交叉验证
'''
num_fold = 5     #表示进行 5 折的交叉验证
X_train_fold = []
y_train_fold = []
K_choice = [1,3,6,12,16,5]
indice = np.array_split(np.array(X_train.shape[0]),indices_or_sections=num_fold)
for i in indice:#对整个数组进行遍历，改变原数组的格式
    X_train_fold.append(X_train[i])
    y_train_fold.append(y_train[i])

for k in K_choice:
    for i in range(num_fold):
        # 将第 i 个数据作为交叉验证的测试集，将剩余其他的作为交叉验证的训练集
        test_x = X_train_fold[i]     #测试集的特征x
        test_y = y_train_fold[i]     #测试集的特征y
        x = X_train_fold[0:i] + X_train_fold[ i + 1 :]
        x = np.concatenate(x , axis = 0)      #训练集的特征x
        y = y_train_fold[0:i] + y_train_fold[ i + 1 :]
        y = np.concatenate(y , axis = 0)      #测试集的特征y





























