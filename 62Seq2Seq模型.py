import math
import collections
import torch
from torch import nn
class Seq2SeqEncoder(nn.Module):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,dropout = 0,**kwargs):
        super(Seq2SeqEncoder,self).__init__()
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.rnn = nn.GRU(embed_size,num_hiddens,num_layers,dropout=dropout)

    def forward(self,X):
        # 输出 'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 在循环神经网络模型中,第一个轴对应时间步
        X = X.permute(1,0,2)
        # 如果未提及状态的画，则默认为 0
        output,state = self.rnn(X)
        # output 的形状为:(num_steps,batch_size,num_hiddens)
        # state 的形状为:(num_layers,batch_size,num_hiddens)
        return output,state

encoder = Seq2SeqEncoder(vocab_size=10,embed_size=8,num_hiddens=16,num_layers=2)
encoder.eval()
X = torch.zeros((4,7),dtype=torch.long)
output,state = encoder(X)



class Seq2SeqDecoder(nn.Module):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,dropout = 0,**kwargs):
        super(Seq2SeqDecoder,self).__init__()
        self.enbedding = nn.Embedding(vocab_size,embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens,num_hiddens,num_layers,dropout=dropout)
        self.dense = nn.Linear(num_hiddens,vocab_size)

    def init_state(self,enc_outputs,*args):
        return enc_outputs[1]

    def forward(self,X,state):
        # 输出'X'的形状:(batch_size,num_steps,embed_size)
        X = self.enbedding(X).permute(1,0,2)
        # 广播context,使其具有与X相同的num_steps
        context = state[-1].repeat(X.shape[0],1,1)
        X_and_context = torch.cat((X,context),2)
        output,state = self.rnn(X_and_context,state)
        output = self.dense(output).permute(1,0,2)
        # output 的形状：(batch_size,num_steps,vocab_size)
        # state 的形状： (num_layers,batch_size,num_hiddens)
        return output,state

decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,num_layers=2)
decoder.eval()
state = decoder.init_state(encoder(X))
output, state = decoder(X, state)
