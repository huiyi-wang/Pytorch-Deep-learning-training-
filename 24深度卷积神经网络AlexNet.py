import torch
from torch import nn

net = nn.Sequential(
      # 这里我们使用一个11*11的更大窗口来捕捉对象
      # 同时，步幅为4，以减少输出的高度和宽度
      # 另外，输出通道的数目远远大于LeNet
      nn.Conv2d( 1 , 96 , kernel_size = 11 , stride = 4 , padding = 1 ) , nn.ReLU(),
      nn.MaxPool2d( kernel_size = 3 , stride = 2 ),
      # 减小卷积的窗口，使用填充为2来使得输入和输出的高和宽一致，且增大输出通道的数目
      nn.Conv2d( 96 , 256 , kernel_size = 5 , padding = 2 ) , nn.ReLU(),
      nn.MaxPool2d( kernel_size = 3 , stride = 2),
      # 使用三个连续的卷积层和较小的卷积窗口
      # 除了最后的卷积层，输出通道的数目进一步增加
      # 在前两个卷积层后，汇聚层不用于减少输入的高度和宽度
      nn.Conv2d( 256 , 384 , kernel_size = 3 , padding = 1 ) , nn.ReLU(),
      nn.Conv2d( 384 , 384 , kernel_size = 3 , padding = 1 ) , nn.ReLU(),
      nn.Conv2d( 384 , 256 , kernel_size = 3 , padding = 1 ) , nn.ReLU(),
      nn.MaxPool2d( kernel_size = 3 , stride = 2),
      nn.Flatten(),
      # 这里，全连接层的输出数量是LeNet中的好多倍，使用dropout层来减轻过拟合的情况
      nn.Linear( 6400 , 4096 ) , nn.ReLU(),
      nn.Dropout( p = 0.5 ),
      nn.Linear( 4096 , 4096 ) , nn.ReLU(),
      nn.Dropout( p = 0.5 ),
      # 最后是输出层，由于这里使用的是Fashion-MNIST，所以类别数为10，而非论文中的1000
      nn.Linear( 4096 , 10 ))

X = torch.randn( 1 , 1 , 224 , 224)
print(X)

for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)







