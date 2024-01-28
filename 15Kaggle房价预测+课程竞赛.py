import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


##第一步：读取数据
path_train = './house-prices-advanced-regression-techniques/train.csv'  #训练数据集地址
path_test  = './house-prices-advanced-regression-techniques/test.csv'   #测试数据集地址
train_data = pd.read_csv(path_train)  #训练数据集
print(train_data)
test_data  = pd.read_csv(path_test)   #测试数据集
print(test_data)
##第二步：预处理数据集
all_features = pd.concat([train_data.iloc[:,1:-1],test_data.iloc[:,1:]],axis=0)
print(all_features)
# print(all_features)
index_Missing_data = all_features.isnull()##全部确实值的索引
numeric_featrues = all_features.dtypes[all_features.dtypes != 'object'].index ##返回所有标签不是object的索引
# print(all_features[numeric_featrues])##输出所有列索引不为object的数据，表示这些列的数据可以用数值表示，即：可以观察到
all_features[numeric_featrues] = all_features[numeric_featrues].apply(lambda x: (x-x.mean())/(x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_featrues] = all_features[numeric_featrues].fillna(0)

# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features, dummy_na=True)

#从Dataframe数据库中读取数据并将其转化为tensor形式
n_train = len(train_data)
train_features = torch.tensor(all_features[:n_train].values,dtype = torch.float32)
test_features  = torch.tensor(all_features[n_train:].values,dtype = torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

'''
##代码测试：检测构成的小批量数据
from torch.utils.data import DataLoader, TensorDataset
train_iter = DataLoader(TensorDataset(train_features, train_labels), batch_size=64,shuffle=True)
for X,Y in train_iter:
    print(X)
    print(Y)

#画一下原始的房子销售价格，并分析一下数据，选择使用的预测方法（本节理解即可）
a = train_labels.numpy().reshape(-1)
b = len(a)

plt.plot(np.arange(100), a[0:100])
plt.show()
'''

###第三部：开始模型的训练
#定义平方损失函数
loss = nn.MSELoss()
#定义一个简单的线性回归模型
in_features = train_features.shape[1]

def get_net():
    net = nn.Sequential(nn.Linear(in_features,1))
    return net

#定义评级模型的指标：对数均方根误差
def log_RMSE(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    RMSE = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return RMSE.item()

#定义训练模型的过程
def train(net,train_features,train_labels,test_features,test_labels,num_epochs,learning_rate,weight_decay,batch_size):
    '''
    net:定义的网络模型
    train_features :训练的特征值
    train_labels   :训练的标签
    test_features  :测试的特征值
    test_labels    :测试的标签
    num_epochs     :训练迭代的次数
    learning_rate  :学习率
    weight_decay   :权重值
    batch_size     :小批量大小
    return         :train_ls表示训练数据集的损失, test_ls表示测试数据集的损失
    '''

    from torch.utils.data import DataLoader, TensorDataset
    ##使用Dataloader进行小批量读取数据
    train_iter = DataLoader(TensorDataset(train_features, train_labels), batch_size=batch_size,shuffle=True)
    train_ls = [] # 用来保存训练集的对数均方根误差
    test_ls  = [] # 用来保存测试集的对数均方跟误差
    optimizer = torch.optim.Adam(net.parameters(),  lr = learning_rate , weight_decay = weight_decay)#优化函数
    for epoch in range(num_epochs):
        for X,Y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X),Y)
            l.backward()
            optimizer.step()
        train_ls.append(log_RMSE(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_RMSE(net, test_features, test_labels))
    return train_ls, test_ls

# 第三部分：K折交叉验证
def get_k_fold_data(k, i, X, y):
    '''
    函数功能：将训练集数据进行 k 折 切 分
    k:表示将数据进行 k 次切分
    i:
    X:表示训练集的数据输入
    y:表示训练集的数据输出
    return:函数返回切分好的数据集
    '''
    assert k > 1  #只有当k大于1时，后面的代码才会开始运行
    fold_size = X.shape[0] // k # 读取训练数据的第一维的数据（行），并将测试的数据集进行相对的切分
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)   #对数据开始切片，并返回切片位置的索引
        X_part = X[idx, :]  # 进行训练集的输入数据切分
        y_part = y[idx]     # 进行训练集的输出数据切分
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,batch_size):
    '''
    k: 进行 k 折交叉检验
    X_train: 训练数据集的输入数据
    y_train: 训练数据集的输出的数据
    num_epochs:迭代的次数
    learning_rate: 学习率
    weight_decay: 权重
    batch_size: 小批量大小
    return: 返回函数的
    '''
    train_l_sum = 0   #用来累计训练数据集的损失
    valid_l_sum = 0   #用来累计测试数据集的损失
    for i in range(k):
        '''
        依次从K个交叉验证集中选取第i个数据集
        '''
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,weight_decay, batch_size)
        train_l_sum += train_ls[-1] #累加返回训练的损失
        valid_l_sum += valid_ls[-1] #累加返回测试的损失
        '输出第一次的损失函数的图像'
        if i :
            plt.plot(list(range(1, num_epochs + 1)), train_ls)
            plt.plot(list(range(1, num_epochs + 1)), valid_ls)
            plt.xlim([1, num_epochs])
            plt.xlabel('epoch')
            plt.ylabel('RMSE')
            plt.legend(['train', 'valid'])
            plt.yscale('log')
            plt.show()

        print(f'k折交叉检验，折{i + 1}，训练log RMSE {float(train_ls[-1]):f}, 'f'验证log RMSE {float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k

k = 5
num_epochs = 100
lr = 5
weight_decay = 0
batch_size =64

train_l, valid_l = k_fold (k, train_features, train_labels, num_epochs, lr,weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log RMSE: {float(train_l):f}, 'f'平均验证log RMSE: {float(valid_l):f}')

##第四部分： 开始进行测试集的预测，并提交预测结果
def train_and_pred(train_features, test_features, train_labels, test_data,num_epochs, lr, weight_decay, batch_size):
    '''
    :param train_features:  训练集的特征
    :param test_features:   测试集的特征
    :param train_labels:    训练集的标签
    :param test_data:       测试集的标签
    :param num_epochs:      迭代次数
    :param lr:              学习率
    :param weight_decay:    权重
    :param batch_size:      小批量大小
    :return:
    '''
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,num_epochs, lr, weight_decay, batch_size)
    plt.plot(np.arange(1, num_epochs + 1), train_ls)
    plt.xlabel('epoch')
    plt.ylabel('log RMSE')
    plt.xlim([1, num_epochs])
    plt.yscale('log')
    plt.show()
    print(f'训练log RMSE：{float(train_ls[-1]):f}')

    # 将网络应用于测试集，并输出预测的数据
    preds = net(test_features).detach().numpy()##开始进行预测

    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('./house-prices-advanced-regression-techniques/submission.csv', index=False)

train_and_pred(train_features, test_features, train_labels, test_data,num_epochs, lr, weight_decay, batch_size)







