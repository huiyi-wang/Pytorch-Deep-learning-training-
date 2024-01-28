import numpy as np
import operator
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.datasets as dsets
import warnings
warnings.filterwarnings('ignore')

'''
第一步：构建函数和KNN分类器
'''

class Knn:
    def __init__(self):
        pass

    def fit(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self,k,dis,X_test):
        num_test = X_test.shape[0]  # 测试样本的数量
        labellist = []
        # 使用欧式距离公式作为距离度量
        if (dis == 'E'):
            for i in range(num_test):
                distances = np.sqrt(np.sum(((self.X_train - np.tile(X_test[i],(self.X_train.shape[0],1))) ** 2) , axis=1))
                nearlist_k = np.argsort(distances)
                top_k = nearlist_k[:k]
                classCounter = {}
                for j in top_k:
                    classCounter[self.y_train[j]] = classCounter.get(self.y_train[j],0) + 1
                sortedclassCounter = sorted(classCounter.items(),key=operator.itemgetter(1),reverse=True)
                labellist.append(sortedclassCounter[0][0])
            return np.array(labellist)

        # 使用曼哈顿距离公式作为距离度量
        if (dis == 'M'):
            for i in range(num_test):
                distances = np.sum(np.abs(self.X_train - np.tile(X_test[i],(self.X_train.shape[0],1))),axis=1)
                nearlist_k = np.argsort(distances)
                top_k = nearlist_k[:k]
                classCounter = {}
                for j in top_k:
                    classCounter[self.y_train[j]] = classCounter.get(self.y_train[j],0) + 1
                    sortedclassCounter = sorted(classCounter.items(),key=operator.itemgetter(1),reverse=True)
                    labellist.append(sortedclassCounter[0][0])
            return np.array(labellist)

def getmean(X_train):
    X_train = np.reshape(X_train,(X_train.shape[0],-1))     # 将图片从二维展开为一维
    mean_image = np.mean(X_train,axis=0)                    # 求解出训练集中所有图片每个像素位置上的平均值
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
第三步，将训练数据集分成5个部分，每个部分轮流作为验证集
'''
num_folds = 5     # 表示将数据集分为 5 折，进行 5 折交叉验证
k_choices = [ 1 , 3 , 5 , 8 , 10 , 12 , 15 , 20 ]
num_training = X_train.shape[0]
X_train_fold = []
y_train_fold = []
indices = np.array_split(np.arange(num_training),indices_or_sections=num_folds)
for i in indices:   # 对整个列表进行遍历，构造批量化数据集
    X_train_fold.append(X_train[i])
    y_train_fold.append(y_train[i])

k_to_accuracies = {}
for k in k_choices:
    '''
    对不同的 k 值进行计算，选择出 k 值最好的那个kNN算法
    '''
    # 进行 k 折交叉验证
    acc = []
    for i in range(num_folds):
        #将第 i 个数据作为交叉验证的验证集，其他的数据集作为交叉验证的训练集
        x = X_train_fold[0:i] + X_train_fold[i+1:]    #训练集的特征中不包含验证集
        x = np.concatenate(x,axis=0)                  #使用 concatenate 将之前分开的四个训练集拼接在一起，构成新的数据集 x
        y = y_train_fold[0:i] + y_train_fold[i+1:]    #训练集的标签中不包含验证集
        y = np.concatenate(y,axis=0)                  #使用 concatenate 将之前分开的四个训练集拼接在一起，构成新的数据集 y
        test_x = X_train_fold[i]                      #单独拿出验证集
        test_y = y_train_fold[i]

        classifier = Knn()                            #定义module
        classifier.fit(x,y)                           #读入训练集
        # dist = classifier.computer_distances_no_loops(test_x)
        y_pred = classifier.predict(k,'M',test_x)     #预测结果
        accuracy = np.mean(y_pred==test_y)            #准确率
        acc.append(accuracy)

k_to_accuracies[k] = acc                              #计算交叉验证的平均准确率

# 输出准确率
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f ' % ( k , accuracy ))

#下面使用代码图形化展示 k 的选取与准确度趋势
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.sactter([k] * len(accuracies) , accuracies)

# plt the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std  = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices,accuracies_mean,yerr=accuracies_std)
plt.title('Cross-validation on k ')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()














