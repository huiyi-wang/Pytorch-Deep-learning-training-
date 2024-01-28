import torch
from torch import nn

'''
互相关运算
'''
print('互相关运算')
def corr2d(X,K):
    #计算二维互相关运算
    h , w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1 , X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i + h,j:j + w] * K).sum()
    return Y
X = torch.tensor([[0.0,1.0,2.0],[3.0,4.0,5.0],[6.0,7.0,8.0]])
K = torch.tensor([[0.0,1.0],[2.0,3.0]])
print(X)
print(K)
a = corr2d(X,K)
print(a)
'''
卷积层
'''
print('卷积层')
class Conv2D(nn.Module):
    def __in1t__(self,kernel_size):
        super(Conv2D, self).__in1t__()
        self.weight =  nn.Parameter(torch.rand(kernel_size))##表示卷积窗口中的内容
        self.bias   =  nn.Parameter(torch.zeros(1))
    def forward(self,X):
        return corr2d(X,self.weight) + self.bias

##图像边缘的检测
X = torch.ones((6,8))
X[:,2:6] = 0
print(X)


K1 = torch.tensor([[1.0,-1.0]])
Y = corr2d(X,K1)
print(Y)
Y2 = corr2d(X.t(), K1)
print(Y2)
print('*'*100)
###学习卷积核
print('学习卷积核')
#构建一个二维卷积层，它具有一个输出通道和形状为（1，2）的卷积核
conv2d = nn.Conv2d(1,1,kernel_size=(1,2),bias=False)
print(conv2d)
X = X.reshape(1,1,6,8)
print(X)
Y = Y.reshape((1,1,6,7))
print(Y)
lr = 3e-2 #学习率
epoch = 10
for i in range(epoch):
    Y_hat = conv2d(X)
    l = (Y_hat-Y)**2   ## 损失函数
    conv2d.zero_grad() ## 梯度归0
    l.sum().backward() ## 反向求解梯度
    #迭代卷积核
    conv2d.weight.data[:] =  conv2d.weight.data[:] - lr * conv2d.weight.grad  #权重更新
    print(f'epoch{i+1},loss{l.sum():.3f}')



