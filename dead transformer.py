# 与NLP预测方式相同
# 通过利用数据的周期性预测
# 输入为101个死亡率，每次预测一个

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
np.set_printoptions(threshold=np.inf)
import datetime
from tkinter import _flatten
from numpy import genfromtxt
import random

device = torch.device("cuda:0")
print(device)
seed = 99
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

look_back =202
de_len = 2
dataset0 = pd.read_csv("D:/pycharm/PycharmProject/dead list/data/loge/1-2.csv", index_col='年份',encoding="gbk")
dataset = np.array(dataset0.astype('float32')).reshape(-1, 1)
train_size = int(len(dataset) * (51/67))
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size - look_back:, :]
train_x = dataset[0:train_size + 1, :]
train_mean, train_std = np.mean(train), np.std(train)

def norm(x):
    return (x - train_mean) / train_std

def create_dataset(data, n):
    dataX, dataY = [], []
    for i in range(len(data) - n):
        a = data[i:(i + n), :]
        dataX.append(a)
        dataY.append(data[i + n, :])
    return np.array(dataX), np.array(dataY)


dataset_norm = dataset
trainX, trainY = create_dataset((train), look_back)
testX, testY = create_dataset((test), look_back)
train_X, train_Y = create_dataset((train_x), look_back)
print(testX)
print(testY)
print(testY.shape)

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    def __init__(self, d_model, out_size):
        super(Generator, self).__init__()
        self.proj1 = nn.Linear(d_model, 1)  # d_model = 4
        self.out_size = out_size
        self.proj2 = nn.Linear(out_size, 1)

    def forward(self, x):
        return self.proj2(F.leaky_relu(self.proj1(x)).reshape(x.size(0), self.out_size))


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size  # size=d_model

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)


    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('float32')
    subsequent_mask = np.ones(attn_shape).astype('float32')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h  # h = 4
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:  # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # 升维:在第一维前面加一维
        nbatches = query.size(0)  # 32
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]  # [32,2,8,2]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.pff = nn.Sequential(nn.Linear(d_model, d_ff),
                                 nn.LeakyReLU(True), nn.Dropout(dropout),
                                 nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.pff(x)


class Embeddings1(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings1, self).__init__()
        self.lut = nn.Linear(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x)


class Embeddings2(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings2, self).__init__()
        self.lut = nn.Linear(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        x = x.to(device) + Variable(self.pe[:, :x.size(1)].to(device), requires_grad=True)
        return x


def make_model(src_vocab, tgt_vocab, N, d_model, d_ff, h, dropout):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings1(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings2(d_model, tgt_vocab), c(position)),
        Generator(d_model,look_back-de_len))

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class Batch:
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)  # [32,1,8]
        if trg is not None:
            self.trg = self.src[:, de_len:, :]
            self.trg_y = trg[:, 1:, :]
            self.trg_mask = self.make_std_mask(self.trg, pad)  # [32, 7, 7]

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad) # [32,1,7]
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-2)).type_as(tgt_mask.data))  # [32,7,7]
        return tgt_mask


minloss = 1.5
Loss = []
def run_epoch(data_iter, Model, loss_compute):
    total_loss = 0
    global minloss
    for j, batch in enumerate(data_iter):
        # [32,8,1]  [32,7,1] [32, 1, 8] [32, 7, 7]
        model.train(mode=True)
        out = Model.forward(batch.src, batch.trg,
                            batch.src_mask.reshape(batch_size, 1, look_back),
                            batch.trg_mask.reshape(batch_size,look_back-de_len, look_back-de_len))
        loss = loss_compute(out, batch.trg_y).to(device) # [32,7]
        total_loss += loss
    if (total_loss/batch_size).cpu().numpy() < minloss:
        minloss = (total_loss/batch_size).cpu().numpy()
        torch.save(model.state_dict(), "transformer_.pth")
    Loss.append((total_loss/batch_size).cpu().numpy())
    print("Epoch Step: %d Loss: %f  " % (epoch, total_loss/batch_size))


class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer  # torch.optim.Adam(model.parameters()...
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 101
        for p in self.optimizer.param_groups:
            p['lr'] = self.rate()  # lr=0.001
        self._rate = learning
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def data_gen(nbatches):
    for i in range((train_size-look_back) // nbatches):
        data1 = torch.from_numpy(X0[nbatches * i:nbatches * i + nbatches].reshape(nbatches, look_back,1)).to(device)
        data2 = torch.from_numpy(Y0[nbatches * i:nbatches * i + nbatches].reshape(nbatches, look_back,1)).to(device)
        src = Variable(data1, requires_grad=False).to(device)
        tgt = Variable(data2, requires_grad=False).to(device)
        yield Batch(src, tgt, 0)


class SimpleLossCompute:
    def __init__(self, generator, criterion, opt):
        self.generator = generator  # model.generator
        self.criterion = criterion  # nn.MSELoss
        self.opt = opt

    def __call__(self, x, y):
        x = self.generator(x)  # [batch_size,1]
        loss = self.criterion(x, y[:, -1])
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.data

X0 = trainX.reshape(trainX.shape[0], trainX.shape[1], 1).astype(np.float32)  # (967,9,1)
Y0 = train_X[1:]
Y0 = Y0.reshape(Y0.shape[0], Y0.shape[1], 1).astype(np.float32)  # (967,9,1)
V = 1
criterion = nn.MSELoss()
criterion = criterion.to(device)
learning = 0.001
model = make_model(V, V, N=1, d_model=32, d_ff=16, h=2, dropout=0.1).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning)
loss_compute = SimpleLossCompute(model.generator, criterion, optimizer)
batch_size = 202

for epoch in range(446):
    run_epoch(data_gen(batch_size), model, loss_compute)
plt.plot(list(range(len(Loss))), Loss, label='Loss',)
plt.xticks(rotation=60)
plt.legend()
plt.show()


def greedy_decode(transformer, src, src_mask):
    memory = transformer.encode(src, src_mask)  # src[1,8] src_mask[1,1,8] start_symbol=1 max_len=8
    ys = src[:, de_len:, :]
    out = transformer.decode(memory, src_mask, Variable(ys), Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
    prob = transformer.generator(out)
    return prob


tes = []
generate_train = []
for place in range(0, len(train) - look_back):
    X0 = trainX
    X0 = X0.reshape(X0.shape[0], X0.shape[1], 1).astype(np.float32)
    X0 = X0[place]  # [8,1]
    src = Variable(torch.Tensor(X0.reshape(1, -1, 1))).to(device)
    src_mask = Variable(torch.ones(1, 1, look_back)).to(device)
    pred = torch.squeeze(greedy_decode(model, src, src_mask).squeeze(0))
    generate_train.append(pred.detach().cpu().numpy() )
pred1 = []
generate_test = []
for place in range(0, len(test)-look_back):
    X0 = testX  # (234,8) testX[236,8]
    X0 = X0.reshape(X0.shape[0], X0.shape[1], 1).astype(np.float32)  # (234,8,1)
    X0 = X0[place]  # [8,1]
    src = Variable(torch.Tensor(X0.reshape(1, -1, 1))).to(device)
    src_mask = Variable(torch.ones(1, 1, look_back)).to(device)
    pred = torch.squeeze(greedy_decode(model, src, src_mask).squeeze(0))
    pred1.append(pred.detach().cpu().numpy())
    generate_test.append(pred.detach().cpu().numpy() )

array1 = np.asarray(pred1)
pred2 = np.array(pred1).T
pred2.reshape(1616)
print(array1.shape)


def MSE(target, prediction):
    error = []
    squaredError = []
    absError = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])
    for val in error:
        squaredError.append(val * val)  # target-prediction之差平方
        absError.append(abs(val))  # 误差绝对值
    print("MSE = ", sum(squaredError) / len(squaredError))
    print("RMSE = ", math.pow(sum(squaredError) / len(squaredError), 0.5))

def MAE(targrt, prediction):
    n = len(targrt)
    absError = []
    for i in range(n):
        absError.append(np.abs(targrt[i] - prediction[i]))
    mae = sum(absError) / n
    print("MAE=", mae)

def MAPE(targrt, prediction):
    n = len(targrt)
    absError = []
    for i in range(n):
        absError.append(np.abs((targrt[i] - prediction[i]) / targrt[i]))
    mape = sum(absError) / n * 100
    print("MAPE=", mape, "%")
# print(testY)
hh1=MAPE(testY,pred1)
hh2=MSE(testY,pred1)
hh3=MAE(testY,pred1)
print(pred2)


