import torch
'''
实现多输入通道的互相关运算
'''
print('实现多输入通道的互相关运算')
def corr2d(X,K):
    #计算二维互相关运算
    h , w = K.shape # 表示二维卷积核的 长 和 宽
    Y = torch.zeros((X.shape[0] - h + 1 , X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i + h,j:j + w] * K).sum()
    return Y

def corr2d_multi_in(X,K):
    return sum(corr2d(x,k) for x,k in zip(X,K))

X = torch.tensor([[[0.0,1.0,2.0],[3.0,4.0,5.0],[6.0,7.0,8.0]],[[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]])
print(X)
K = torch.tensor([[[0.0,1.0],[2.0,3.0]],[[1.0,2.0],[3.0,4.0]]])
print(K)
corr2d_multi_in(X, K)

print(corr2d_multi_in(X,K))
print('*'*100)
'''
实现多输出通道的运算
'''
print('实现多输出通道的运算')
def corr2d_multi_out(X,K):
    return torch.stack([corr2d_multi_in(X,k) for k in K],0)
K = torch.stack((K,K+1,K+2),0) ##使用3维的卷积核
print(K)
print(corr2d_multi_out(X,K))
print('*'*100)
'''
验证1*1的卷积核等价于全连接网络
'''
print('1X1卷积')
def corr2d_multi_in_out_1X1(X,K):
    c_i , h , w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i , h * w))
    K = K.reshape((c_o,c_i))
    Y = torch.matmul(K,X)
    return Y.reshape((c_o,h,w))
X1 = torch.normal(0,1,(3,3,3))
K1 = torch.normal(0,1,(2,3,1,1))
Y1 = corr2d_multi_in_out_1X1(X1,K1)
Y2 = corr2d_multi_out(X1,K1)
print('Y1',Y1)
print('Y2',Y2)
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6