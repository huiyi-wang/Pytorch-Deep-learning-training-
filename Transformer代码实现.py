import torch
import numpy as np
import copy
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.nn import functional as F
'第一步：输入部分的实现'
# 构建 Embedding 类来实现文本嵌入层
class Embeddings(nn.Module):
    def __init__(self,d_model,vocab):
        # d_model : 词嵌入维度
        # vocab   : 词表大小
        '类初始化函数，有两个参数：d_model: 指词嵌入的维度，vocab：指词表的大小'
        # 接着就是使用 super 的方式指明继承 nn.Module 的初始化函数，我们自己实现的所有层都这样取做
        super(Embeddings,self).__init__()
        # 之后调用 nn 中预定义层 Embedding ，获得一个词嵌入对象 self.lut
        self.lut = nn.Embedding(vocab,d_model)
        # 最后就是将 d_model 传入类中
        self.d_model = d_model
    def forward(self,x):
        # 可以将其理解为该层的前向传播逻辑，所有层中都会有此函数
        # 当传给该类的实例化对象参数时，自动调用该函数
        # 参数x ：因为 Embedding 层是首层，所以代表输入给模型的文本通过词汇映射后的张量
        return self.lut(x) * math.sqrt(self.d_model)

# 构建 PositionalEncoding 类来实现 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout,max_len = 5000):
        '位置编码器类的初始化函数，共有三个参数，分别是 d_model:词嵌入维度，dropout:置0比例，max_len:每个句子的最大长度'
        super(PositionalEncoding,self).__init__()
        # 实例化 nn 中预定义的 Dropout 层，并将 dropout 传入其中，获得对象self.dropout
        self.dropout = nn.Dropout(p = dropout)

        # 初始化一个绝对位置编码矩阵，它是一个0矩阵，矩阵的大小为 max_len * d_model
        pe = torch.zeros(max_len,d_model)  # 60 * 512 -> 1 * 60 * 512

        # 初始化一个绝对位置矩阵，在这里，词汇的绝对位置就是用它的索引去表示
        # 所以我们首先使用arange方法获得一个连续自然数向量，然后再使用 unsqueeze 方法拓展向量维度使其成为矩阵
        # 又因为参数传的是 1 ，代表矩阵拓展的位置，会使得向量变成一个 max_len * 1 的矩阵
        position = torch.arange(0,max_len).unsqueeze(1)    # 在列的维度上加1

        # 绝对位置矩阵初始化之后，接下来就是考虑如何将这些位置信息加入到位置编码矩阵中
        # 最简单的思路就是先将 max_len * 1的绝对位置矩阵，变换成 max_len * d_model 的形状，然后覆盖原来的初始位置编码
        # 要做这种矩阵变换，就需要一个 1 * d_model 形状的变换矩阵 div_term，我们对这个变换矩阵的要求除了形状外。
        # 还希望它能够将自然数的绝对位置编码缩放成足够小的数字，有助于在之后的梯度下降过程中更快的收敛
        # 首先使用arange获得一个自然数矩阵，但是细心的同学们会发现，我们这里并没有按照预计的一样初始化一个 1 * d_model
        # 而是有了一个跳跃，只初始化了一半即 1 * d_model/2 的矩阵，为什么是一半呢 ，其实这里不是真正意义上的初始化了一半矩阵
        # 我们可以把它看作是初始化了两次，而每次初始化的变化矩阵会做不同的处理，第一次初始化的变换矩阵分布在正弦波上，第二次初始化的变换矩阵
        # 并把这两个矩阵分别填充在位置编码矩阵的偶数和奇数位置上，组成最终的位置编码矩阵
        div_term = torch.exp(torch.arange(0,d_model,2) * -(math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin( position * div_term )
        pe[:,1::2] = torch.cos( position * div_term )

        # 这样我们就得到了位置编码矩阵pe，pe现在还只是一个二维矩阵，要想和embedding的输出
        # 就必须要扩展一个维度，所以这里使用unsqueeze扩展维度
        pe = pe.unsqueeze(0)  # 在行的维度上加1

        # 最后把pe位置编码矩阵注册成模型的buffer，什么是buffer呢
        # 我们把它认为是对模型效果有帮助的，但是却不是模型结构中超参数或者参数，不需要随着优化步骤进行更新的增益对象
        # 注册之后我们就可以在模型保存后重加载时和模型结构与参数一同被加载
        self.register_buffer('pe',pe)

    def forward(self,x):
        'forward函数的参数是 x ，表示文本序列中词嵌入表示'
        # 在相加之前我们对 pe 做一些适配工作，将这个三维张量的第二维也就是句子最大长度的那一维将切片到与输入的x的第二维相同
        # 因为我们默认max_len为5000一般来讲实在太大了，很难有一条句子包含5000个词汇，所以要进行与输入张量的适配
        # 最后使用Variable进行2封装，使其与x的样式相同，但是它是不需要进行梯度求解的，因此把requires_grad设置为False
        x = x + Variable(self.pe[:,:x.size(1)],requires_grad = False)  # 此处使用了 pytorch 的广播机制

        # 最后使用 self.dropout 对象进行’丢弃‘操作，并返回结果
        return self.dropout(x)

'第二部分:编码器部分的实现'
# 掩码张量
def subsequent_mask(size):
    "生成向后遮掩的掩码张量，参数size是掩码张量最后两个维度的大小，它的最后两维形成一个方阵"
    # 在函数中，首先定义掩码张量的形状
    attn_shape = (1,size,size)

    # np.triu : 然后使用 np.ones 方法向这个形状中添加1元素，形成上三角矩阵，最后为了节约空间
    # 再使其中的数据类型变为无符号 8 位整形 unit8
    subsequent_mask = np.triu(np.ones(attn_shape),k = 1).astype('uint8')

    # 最后将numpy类型转换为torch中的tensor类型，内部做一个 1 - 的操作
    # 在这个其实就是做了一个三角矩阵的反转，subsequent_mask 中的每一个元素都会被 1 减
    # 如果是 0 ，subsequent_mask 中的该位置由 0 变为 1
    # 如果是 1 ，subsequent_mask 中的该位置有 1 变为 0
    return torch.from_numpy(1 - subsequent_mask)

# 注意力机制
def attention(query,key,value,mask = None , dropout = None):
    '注意力机制的实现，分别输入的是 query ， key ， value ， mask: 掩码张量 ， dropout 是 nn.Droupout 层的实例化对象，默认为None'
    # 在函数中，首先获取query的最后一维的大小，一般情况下就等同于我们的词嵌入维度，命名为d_k
    d_k = query.size(-1)

    # 按照注意力公式，将 query 与 key 转置相乘，这里面key是将最后两个维度进行转置，再除以 缩放系数math.sqrt(d_k) ,得到注意力得分张量scores
    scores = torch.matmul(query,key.transpose(-2,-1)) / math.sqrt(d_k)

    # 接着判断是否使用掩码张量
    if mask is not None:
        # 使用tensor的mask_fill方法，将掩码张量和scores张量的每一个位置一一比较，
        # 则对应的scores张量用-1e9这个值来替换，如下演示：
        scores = scores.masked_fill( mask == 0 ,-1e9 )

    # 对scores的最后一维进行softmax操作，使用 F.softmax 方法，第一个参数是softmax对象，第二个是dim变量
    # 这样获得最终的注意力张量
    p_attn = F.softmax(scores,dim = -1)

    # 之后判断是否使用 dropout 进行随机置 0
    if dropout is not None:
        # 将p_attn传入dropout对象中进行丢弃处理
        p_attn = dropout(p_attn)

    # 最后，根据公式将p_attn与value张量相乘获得最终的query注意力表示，同时返回注意力张量
    return torch.matmul( p_attn , value ) , p_attn

# 克隆函数
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# 多头自注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self,head,d_model,dropout = 0.1):
        super(MultiHeadAttention,self).__init__()
        assert d_model % head == 0  # 判断多头注意力的头数是否能被词向量维度整除
        self.d_k = d_model // head
        self.head = head
        self.linears = clones(nn.Linear(d_model,d_model),4)
        self.attn = None
        self.dropout = nn.Dropout(p = dropout)
    def forward(self,query,key,value,mask = None):
        if mask is not None:
            # 使用squeeze将掩码张量进行维度扩充，代表多头中的第n个头
            mask = mask.unsqueeze(1)
        # 得到batch_size的大小
        n_batchs = query.size(0)

        # 首先使用zip将网络和输入数据连接到一起，模型的输出利用view和transpose进行维度和形状的变换
        query,key,value = [l(x).view(n_batchs,-1,self.head,self.d_k).transpose(1,2) for l,x in zip(self.linears,(query,key,value))]

        # 将每个头的输出传入到注意力层
        x , self.attn = attention(query,key,value , mask = mask ,dropout = self.dropout)

        # 得到每个头的计算结果是4维张量，需要进行形状的转换
        # 前面已经将1，2两个维度进行过转置，再这里需要重新转置回来
        # 注意:经历了transpose()方法后,必须要使用contiguous，不然无法使用view()方法
        x = x.transpose(1,2).contiguous().view(n_batchs,-1,self.head * self.d_k)

        # 最后将x输入线性层列表中的最后一个线性层中进行处理，得到最终的多头注意力结构输出
        return self.linears[-1](x)

# 前馈全连接层
# 通过类PositionwiseFeedForward
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.1):
        '初始化函数有三个输入参数分别是d_model,d_ff 和 droppout = 0.1， 第一个是线性层的输入维度，因为我们希望输入通道通过前馈全连接层后输入和输出二点维度保持不变；第二个参数 d_ff 就是第二个线性层的输出维度；最后一个是 dropout 置 0 比率'
        # features : 代表词嵌入的维度
        # eps : 一个足够小的正数 ， 用来再规范化计算公式的分母中，防止除零操作
        super(PositionwiseFeedForward,self).__init__()

        # 首先按照我们预期使用nn实例化了两个线性层对象，self.w1 和 self.w2
        # 它们的参数分别是d_model , d_ff 和 d_ff ， d_model
        self.w1 = nn.Linear(d_model,d_ff)
        self.w2 = nn.Linear(d_ff,d_model)

        # 然后使用nn的 Dropout 实例化了对象self.dropout
        self.dropout =nn.Dropout(dropout)

    def forward(self,x):
        '输入第一个参数为 x ，代表来自上一层的输出'
        # 首先经过第一个线性层，然后使用Funtional中relu函数进行激活
        # 之后再使用 dropout 进行随机置 0 ，最后通过第二个线性层w2，返回最终结果
        return self.w2( self.dropout( F.relu( self.w1(x) ) ) )

# 规划化层
# 通过LayerNorm实现规范化层的类
class LayerNorm(nn.Module):
    def __init__(self,feature,eps = 1e-6):
        '初始化函数有两个参数，一个是features，表示词嵌入的维度，另一个是eps它是一个足够小的数，在规范化公式的分母中出现，防止分母为0，默认值是1e-6'
        super(LayerNorm,self).__init__()
        # 根据features的形状初始化两个参数张量 a2 和 b2 ，第一个初始化为 1 张量
        # 也就是里面的元素都是 1 ，第二个初始化为 0 的张量，也就是里面的元素都是 0 ，这两个张量都是规范化层的参数
        # 因为直接对上一层得到的结果做规范化公式计算，将改变结果的正常表征，因此就需要有参数作为调节因子
        # 使其即能满足规范化要求，又不能改变针对目标的表征，最后使用nn.parameter封装，代表他们的模型的参数
        self.a2 = nn.Parameter(torch.ones(feature))  # 缩放系数
        self.b2 = nn.Parameter(torch.zeros(feature)) # 位移系数

        # 把 eps 传到类中
        self.eps = eps

    def forward(self,x):
        '输入参数 x 代表来自上一层的输出'
        # 在函数中，首先对输入变量 x 求其最后一个维度的均值，并保持输出维度与输入维度一致
        # 接着再求最后一个维度的标准差，然后就是根据规范化公式，用x减去均值除以标准差来获得规范化的结果
        # 最后对结果乘以我们的缩放系数，即a2 ， * 表示同型点乘，即对应位置进行乘法操作，加上位移参数b2,返回即可
        mean = x.mean(-1,keepdim = True)
        std = x.std(-1,keepdim = True)
        return self.a2 * (x - mean)  / (std + self.eps) + self.b2

# 子层连接
# 使用SublayerConnection来实现子层连接结构的类
class SublayerConnection(nn.Module):
    def __init__(self,size,dropout = 0.1):
        '它的输入参数有两个，size以及dropout，size一般是都是词嵌入维度的大小'
        'dropout本身是对模型结构中的节点数进行随机抑制的比率'
        '又因为节点被抑制等效果就是该节点的输出都是0，因此也可以把dropout看作是对输出矩阵的随机丢弃'
        super(SublayerConnection,self).__init__()
        # 实例化了规范化对象 self.norm
        self.norm = LayerNorm(size)

        # 又使用nn中预定义的dropout实例化了一个dropout对象
        self.dropout = nn.Dropout(p = dropout)

    def forward(self,x,sublayer):
        '前向逻辑函数中: 接受上一个层或者子层的输入作为第一个参数'
        '将该子层连接中的子层函数作为第二个参数'

        # 我们首先对输出进行规范化，然后将结果传给子层处理，之后在对子层进行dropout操作
        # 随机停止一些网络中神经元的作用，来防止过拟合，最后还有一个add操作
        # 因为存在跳跃连接，所以是将输入x与dropout后的子层输出结果相加作为最终的子层连接输出
        return x + self.dropout(sublayer(self.norm(x)))

# 使用 Encoderlayer 类实现编码器层
class Encoderlayer(nn.Module):
    def __init__(self,size,self_attn,feed_forward,dropout):
        '它的初始化函数的参数有四个，分别是'
        # size:其实是词嵌入维度的大小，它作为我们编码器的输入
        # self_attn:之后我们传入多头自注意力层实例化对象，并且是自注意力机制
        # feed_forward: 之后我们传入前馈全连接层实例化对象
        # dropout: 置0比例
        super(Encoderlayer,self).__init__()

        # 首先将self_attn和feed_forward传入其中
        self.self_attn = self_attn
        self.feed_forward = feed_forward

        # 编码器层中有两个子层连接结构，所以使用clones函数进行克隆
        self.sublayer = clones(SublayerConnection(size,dropout),2)

        # 把size传入其中
        self.size = size

    def forward(self,x,mask):
        'forward函数中有两个参数：x和mask，分别代表上一层的输出和掩码张量mask'
        # 首先经过第一个子层连接结构，其中包含多头自注意力子层
        # 然后通过第二个子层连接结构，其中包含前馈全连接子层，最后返回结果
        x = self.sublayer[0](x,lambda x : self.self_attn(x,x,x,mask))
        x = self.sublayer[1](x,self.feed_forward)
        return x

# 使用Encoder类来实现编码器
class Encoder(nn.Module):
    def __init__(self,layer,N):
        '初始化函数的两个参数分别代表编码器层和编码器层的个数'
        super(Encoder,self).__init__()
        # 首先使用克隆函数clones克隆N个编码器放在self.layers中
        self.layers = clones(layer,N)

        # 在初始化一个规范化层，它将用在编码器的最后面
        self.norm = LayerNorm(layer.size)
    def forward(self,x,mask):
        'forward函数的输入和编码器相同，x代表上一层的输出，mask代表掩码张量'

        # 首先就是对我们的克隆函数的编码器层进行循环，每次都会得到一个新的x
        # 这个循环过程，就是相当于输出的x经过了N个编码器层的处理
        # 最后再通过规范化层的对象self.norm进行处理，最后返回结果
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)

'第三部分:使用DecoderLayer的类实现解码器层'
class Decoderlayer(nn.Module):
    def __init__(self,size,self_attn,src_attn,feed_forward,dropout):
        # size : 代表词嵌入的维度
        # self_attn : 代表多头自注意力机制的对象
        # src_attn  : 代表常规的注意力机制的对象
        # feed_forward : 代表前馈全连接层的对象
        # dropout : 代表Dropout的置0比率
        super(Decoderlayer,self).__init__()

        # 将参数传入类中
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = dropout

        # 按照解码器层的结构图，使用clones函数克隆3个子层连接对象
        self.sublayer = clones(SublayerConnection(size,dropout),3)
    def forward(self,x,memory,source_mask,target_mask):
        # x 代表上一层输入的张量
        # memory : 代表编码器的语义存储张量
        # source_mask : 原数据的掩码张量
        # target_mask : 目标数据的掩码张量
        m = memory

        # 第一步让 x 经历第一个子层，多头自注意力机制的子层
        # 采用target_mask,为了将掩码时未来的信息进行遮掩，比如模型解码第二个字符，只能看见第一个字符的信息
        x = self.sublayer[0](x,lambda x:self.self_attn(x,x,x,target_mask))

        # 第二步让 x 经过第二个子层，常规的注意力机制的子层，此时Q ！= K = V
        # 采用source_mask ,为了遮掩掉对结果信息无用的数据
        x = self.sublayer[1](x,lambda x: self.src_attn(x,m,m,source_mask))

        # 第三次让 x 经过第三个子层，前馈全连接层
        x = self.sublayer[2](x,self.feed_forward)
        return x

# 实现解码器层Decoder
class Decoder(nn.Module):
    def __init__(self,layer,N):
        '初始化参数有两个，第一个就是解码器层layer，第二个就是解码器层的个数N'
        # layer:代表解码器层的对象
        # N : 代表将layer进行几层的拷贝
        super(Decoder,self).__init__()
        # 首先使用clones方法克隆了N个layer，然后实例化一个规范化层
        # 因为数据走过了所有的解码器层最后要做规范化处理
        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.size)

    def forward(self,x,memory,source_mask,target_mask):
        'forward函数中的参数有4个，x代表目标数据的嵌入表示，memory是编码器层的输出，source_mask,target_mask代表源数据和目标数据的掩码张量'
        # 然后就是对每一层进行循环，当然这个循环就是变量x通过每一个层的处理，得到最后的结果，再进行一次规范化返回即可
        for layer in self.layers:
            x = layer(x,memory,source_mask,target_mask)
        return self.norm(x)

'第四部分:输出部分'

# 将线性层和softmax计算层一起实现，因为二者的共同目标是生成最后的结构
# 因此把类的名字叫做Generator，生成器类
class Generator(nn.Module):
    def __init__(self,d_model,vocab_size):
        '初始化函数的输入参数有两个，d_model代表词嵌入维度，vocab_size代表词表的大小'
        super(Generator,self).__init__()
        # 首先就是使用nn中的预定义线性层进行实例化，得到最后一个对象self.project等待使用
        # 这个线性层的参数有两个，就是初始化函数传进来的两个参数:d_model,vocab_size
        self.project = nn.Linear(d_model,vocab_size)

    def forward(self,x):
        '前向逻辑函数中输入的上一层的输出张量 x '
        # 在函数中，首先使用上一步得到的self.project对 x 进行线性变换
        # 然后使用F中已经实现的log_softmax进行的softmax处理
        # 在这里之所以使用log_softmax，是因为和我们这个pytorch版本的损失函数实现有关，在其他版本上没有这个问题
        # log_softmax就是softmax的结果又取了对数，因为这个对数是单调递增函数
        # 因此对最终我们取最大值的概率值没有影响，最后返回结果就可以了

        return F.log_softmax(self.project(x),dim=-1)

'第五部:使用EncoderDecoder类来实现编码器-解码器结构'
class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder,source_embed,target_embed,generator):
        '初始化函数中有5个参数：分别是编码器对象，解码器对象，源数据嵌入函数，目标数据嵌入函数，以及输出部分的类别生成器对象'
        super(EncoderDecoder,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator

    def forward(self,source,target,source_mask,target_mask):
        '在forward函数中，有四个参数，source代表源数据，target代表目标数据，source_mask 和 target_mask代表对应的掩码张量'
        # 在函数中，将source，source_mask传入编码函数中，得到结果后，与source_mask,target 和 target_mask一同传给解码函数
        return self.decoder(self.encoder(source,source_mask),source_mask,target,target_mask)

    def encoder(self,source,source_mask):
        '编码函数，以 source 和 source_mask 为参数'
        # 使用src_embed对source做处理，然后和source_mask一起传给self.encoder
        return self.encoder(self.src_embed(source),source_mask)

    def decoder(self,memory,source_mask,target,target_mask):
        '解码函数，以 memory 即解码器的输出，source_mask,target,target_mask 为参数 '
        # memory : 代表经历编码器编码后的输出张量
        # 使用tgt_embed对target做处理，然后和source_mask,target_mask,memory一起传给self.decoder
        return self.decoder(self.tgt_embed(target),memory,source_mask,target_mask)

def make_model(source_vocab, target_vocab, N = 6, d_model = 512, d_ff = 2048, head = 8, dropout = 0.1):
    '该函数用来构建模型，有7个参数，分别是源数据特征（词汇）总数，目标数据特征（词汇总数），编码器和解码器堆叠数。词向量映射维度，前馈全连接网络中变换矩阵的维度。多头注意力结构中的多头数，以及置零比率dropout'
    # source_vocab:源数据特征（词汇）总数
    # target_vocab:目标数据特征（词汇）总数
    # N = 6 ，编码器和解码器堆叠数
    # d_model : 词向量映射维度
    # d_ff : 前馈全连接网络中变换矩阵的维度
    # head = 8 ,多头注意力结构中的多头数
    # dropout : 置0比率

    # 首先得到一个深度拷贝命令，接下来很多结构都需要进行深度拷贝，来保证他们彼此之间相互独立，不受干扰
    c = copy.deepcopy

    # 实例化了多头注意力类，得到对象attn
    attn = MultiHeadAttention(head,d_model)

    # 然后实例化前馈全连接类，得到对象ff
    ff = PositionwiseFeedForward(d_model,d_ff,dropout)

    # 实例化位置编码类，得到对象position
    position = PositionalEncoding(d_model,dropout)

    # 根据结构图，最外层是EncoderDecoder，在EncoderDecoder中，分别是编码器层，解码器层，源数据Embedding层和位置编码组成的有序结构
    # 目标数据Embedding层和位置编码组成的有序结构，以及类别生成器层
    # 在编码器层中有attention子层以及前馈全连接子层
    # 在解码器层中两个attention子层以及前馈全连接层
    model = EncoderDecoder(
        Encoder(Encoderlayer(d_model,c(attn),c(ff),dropout),N),
        Decoder(Decoderlayer(d_model,c(attn),c(attn),c(ff),dropout),N),
        nn.Sequential(Embeddings(d_model,source_vocab),c(position)),
        nn.Sequential(Embeddings(d_model,target_vocab),c(position)),
        Generator(d_model,target_vocab))

    # 模型结构完成后，接下来就是初始化模型中的参数，比如线性层中的变换矩阵
    # 这里一但判断参数的维度大于1，则会将其初始化成一个服从均匀分布的矩阵

    for p in model.parameters():   # 遍历模型中所有的参数
        if p.dim() > 1 :  # 判断是高维的张量
            nn.init.xavier_uniform_(p)

    return model

if __name__ == '__main__':
    # # 词嵌入维度是512维
    # d_model = features = 512
    #
    # # 字典大小
    # voacb = 1000
    #
    # # 置 0 比率为0.1
    # dropout = 0.1
    #
    # # 句子最大长度
    # max_len = 60
    #
    # # 线性变化的维度
    # d_ff = 64
    #
    # # 多头注意力的头数
    # head = 8
    #
    # x = Variable(torch.LongTensor([[100,2,421,508],[491,998,1,221]]))   # 词表
    # emb = Embeddings(d_model,voacb)                    # 编码层
    # embr = emb(x)
    # # print('embr:',embr)
    # # print(embr.shape)
    #
    # pe = PositionalEncoding(d_model,dropout,max_len)   # 位置编码
    # pe_result = pe(embr)
    # # print('pe_result',pe_result)
    # # print(pe_result.shape)
    #
    # # query = key = value = pe_result
    # # mask = Variable(torch.zeros(2, 4, 4))
    # # attn,p_attn = attention(query,key,value,mask)      # 注意力机制
    # # print('attn:',attn)
    # # print(attn.shape)
    # # print('p_attn',p_attn)
    # # print(p_attn.shape)
    #
    # # query = key = value = pe_result
    # # mask = Variable(torch.zeros(2,4,4))
    # # mha = MultiHeadAttention(head,d_model,dropout = 0.2) # 多头注意力机制
    # # mha_result = mha(query,key,value,mask)
    # # print(mha_result)
    # # print(mha_result.shape)
    #
    # # x1 = mha_result
    # # ff = PositionwiseFeedForward(d_model,d_ff,dropout)  # 前馈传播层
    # # ff_result = ff(x1)
    # # # print(ff_result)
    # # # print(ff_result.shape)
    # #
    # # x2 = ff_result
    # # In = LayerNorm(features)                            # 规范化层
    # # In_result = In(x2)
    # # # print(In_result)
    # # # print(In_result.shape)
    # #
    # # x3 = pe_result
    # # mask = Variable(torch.zeros(2,4,4))
    #
    # # # 假设子层中装的是多头注意力层，实例化这个类
    # # self_attn = MultiHeadAttention(head,d_model)
    # #
    # # # 使用lambda获得一个函数类型的子层
    # # sublayer = lambda x3 : self_attn(x3,x3,x3,mask)
    # #
    # # sc = SublayerConnection(size = 512,dropout = 0.2)
    # # sc_result = sc(x3,sublayer)
    # # print(sc_result)
    # # print(sc_result.shape)
    #
    # size = 512
    #
    # d_model = 512
    #
    # x = pe_result
    # dropout = 0.2
    # self_attn = MultiHeadAttention(head,d_model)
    # ff = PositionwiseFeedForward(d_model,d_ff,dropout)
    # mask = Variable(torch.zeros(2,4,4))
    #
    # # print(el_result)
    # # print(el_result.shape)
    #
    # c = copy.deepcopy
    # layer = Encoderlayer(size,self_attn,ff,dropout)
    # N = 8
    # en = Encoder(layer,N)
    # en_result = en(x,mask)
    # print(en_result)
    # print(en_result.shape)
    #
    # self_attn = src_attn = MultiHeadAttention(head,d_model,dropout)
    # ff = PositionwiseFeedForward(d_model,d_ff,dropout)
    # x = pe_result
    # memory = en_result
    # source_mask = target_mask = mask
    # dl = Decoderlayer(size,self_attn,src_attn,ff,dropout)
    #
    # de = Decoder(dl,8)
    # de_result = de(x,memory,source_mask,target_mask)
    # print(de_result)
    source_vocab = 11
    target_vocab = 11
    N = 6
    res = make_model(source_vocab,target_vocab,N)
    print(res)
