import torch
import datetime
import numpy as np
import torch.nn as nn
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import plot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False

'第一步：设置全局变量'
num_time_steps = 16 # 训练时间窗的步长
input_size = 3      # 输入数据的维度
hidden_size = 16    # 隐含层的维度
output_size = 3     # 输出层的维度
num_layers = 1      # 隐含层的个数
lr = 0.01           # 学习率

'第二步：定义RNN类'
class Net(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        super(Net, self).__init__()
        self.rnn = nn.RNN(input_size = input_size,
                          hidden_size = hidden_size,
                          num_layers = num_layers,
                          batch_first = True)

        for p in self.rnn.parameters():   #RNN层进行参数初始化
            nn.init.normal_(p , mean = 0.0 , std = 0.001)

        self.linear = nn.Linear(hidden_size , output_size)

    def forward(self , x , hidden_prev):
        out , hidden_prev = self.rnn( x , hidden_prev)
        # [ b , seq , h ]
        out = out.view(-1 , hidden_size)  # 将RNN层的输出维度进行重新构建，为了使得之后的线性输出层的输入维度与之相匹配
        out = self.linear(out)
        out = out.unsqueeze(dim=0)        # unsqueeze函数为升维函数，表示在哪一个地方增加一个维度
        return out , hidden_prev

'第三步：构造初始化训练集'

def getdate():
    x1 = np.linspace(1,10,30).reshape(30,1)
    y1 = ( np.zeros_like(x1)+2 ) + np.random.rand(30,1)*0.1
    z1 = ( np.zeros_like(x1)+2 ).reshape(30,1)
    tr1 = np.concatenate((x1,y1,z1),axis=1)      #表示全部的数据集tr1（data）
    return tr1

'第四步：开始训练模型'
num_epoch = 3000
def train_RNN(data):
    model = Net(input_size , hidden_size , num_layers)
    print('model:\n',model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr)
    # 初始化h
    hidden_prev = torch.zeros(1,1,hidden_size)
    l = []
    # 训练3000次
    for epoch in range(num_epoch):
        # loss = 0
        start = np.random.randint(10,size=1)[0]
        end = start + 15
        x = torch.tensor(data[start:end]).float().view(1 , num_time_steps - 1 , 3)
        # 在data里面随机选择15个点作为输入，预测第16个数
        y = torch.tensor(data[start + 5 : end + 5 ]).float().view(1 , num_time_steps -1 , 3)
        output , hidden_prev = model( x , hidden_prev )
        hidden_prev = hidden_prev.detach()
        loss = criterion(output,y)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0 :
            print("Iteration:{}loss{}".format(iter,loss.item()))
            l.append(loss.item)
    ##########绘制损失函数############
    # plt.plot(l,'r')
    # plt.xlabel('训练次数')
    # plt.ylabel('loss')
    # plt.title('RNN损失函数下降曲线')
    return hidden_prev,model

'第五步：开始预测'
def RNN_pre(model,data,hidden_prev):
    data_test = data[19:29]
    data_test = torch.tensor(np.expand_dims(data_test,axis=0),dtype=torch.float32)
    pred1,h1 = model(data_test,hidden_prev)
    print('pred1.shape:',pred1.shape)
    pred2,h2 = model(pred1,hidden_prev)
    print('pred2.shape:',pred2.shape)
    pred1 = pred1.detach().numpy().reshape(10,3)
    pred2 = pred2.detach().numpy().reshape(10,3)
    predictions = np.concatenate((pred1,pred2),axis=0)
    print('predictions.shape:',predictions.shape)

    ##########预测可视化#############
    # fig = plt.figure(figsize=(9,6))
    # ax  = Axes3D(fig)
    # ax.scatter3D(data[:,0],data[:,1],c='red')
    # ax.scatter3D(predictions[:,0],predictions[:,1],predictions[:,2],c='y')
    # ax.set_xlabel('X')
    # ax.set_xlim(0,8.5)
    # ax.set_ylabel('Y')
    # ax.set_ylim(0,10)
    # ax.set_zlabel('Z')
    # ax.set_zlim(0,4)
    # plt.title("RNN航迹预测")
    # plt.show()

'第六步：定义主函数mian'
def main():
    data = getdate()
    start = datetime.datetime.now()
    hidden_pre,model = train_RNN(data)
    end = datetime.datetime.now()
    print('The training time : %s' %str(end - start))
    # plt.show()
    RNN_pre(model,data,hidden_pre)

if __name__ == '__main__':
    main()

