import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
import math
import pdb

class Pretrain_MLP(nn.Module):
    def __init__(self,dropout=0.3,input_size=128,output_size=19):
        super().__init__()
        self.layer1 = nn.Linear(input_size,256)
        self.layer2 = nn.Linear(256,256)
        self.layer3 = nn.Linear(256,64)
        self.layer4 = nn.Linear(64,output_size)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        x = x.float()
        x = self.dropout(self.act(self.layer1(x)))
        x = self.dropout(self.act(self.layer2(x)))
        x = self.dropout(self.act(self.layer3(x)))
        x = self.layer4(x)
        return x


class Attention_Embedding_(nn.Module):
    def __init__(self, interval, embedding_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.ModuleList([
            nn.Embedding(interval[i], embedding_dim) for i in range(9)
        ])
        self.attn_score = nn.Parameter(torch.ones(9,1))
    def forward(self, data):
        embeddings = []
        pdb.set_trace()
        for i in range(9):
            embeddings.append(self.embeddings[i](data[:, i]))

        embeddings = torch.stack(embeddings, dim=1)
        result = (embeddings * self.attn_score).sum(dim=1)
        return result, self.attn_score

class Attention_Embedding(nn.Module):
    def __init__(self, interval, embedding_dim=128):
        super().__init__()
        max_num = sum(interval)
        self.embedding = nn.Embedding(max_num, embedding_dim)
        self.offset = torch.cumsum(torch.tensor([0]+interval,device="cuda"),0)[:-1]
        self.attn_score = nn.Parameter(torch.ones(9,1))
    def forward(self, data):
        data = data + self.offset
        embeddings = self.embedding(data)
        result = (embeddings * self.attn_score).sum(dim=1)
        return result, self.attn_score


class SelfAttention_Embedding(nn.Module):
    def __init__(self, interval, embedding_dim=128, heads=8, attn_dropout=0.3):
        super().__init__()
        max_num = sum(interval)
        self.embedding_dim = embedding_dim
        self.heads = heads
        assert embedding_dim % heads == 0
        self.d_k = embedding_dim // heads
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.embedding = nn.Embedding(max_num, embedding_dim)
        self.offset = torch.cumsum(torch.tensor([0]+interval,device="cuda"),0)[:-1]
        self.softmax = nn.Softmax(dim=-1)
        self.W_q = nn.Linear(embedding_dim,embedding_dim)
        self.W_k = nn.Linear(embedding_dim,embedding_dim)
        self.W_v = nn.Linear(embedding_dim,embedding_dim)

    def forward(self, data):
        data = data + self.offset
        # print(data[:,-1])
        embeddings = self.embedding(data)
        b = data.shape[0]
        m = data.shape[1]
        def shape(x): # [b, m, d_model] -> [b, heads, m, d_k]
            return x.view(b, -1, self.heads, self.d_k).transpose(1, 2)

        def unshape(x):	# 当乘以V之后，x再transpose就是不连续的了
            return x.transpose(1, 2).contiguous().view(b, -1, self.embedding_dim)
        q = self.W_q(embeddings)
        k = self.W_k(embeddings)
        v = self.W_v(embeddings)

        q = shape(q)
        k = shape(k)
        v = shape(v)

        scores = torch.matmul(q, k.transpose(2, 3))
        scores = scores / math.sqrt(self.d_k)

        attn = self.softmax(scores)
        drop_attn = self.attn_dropout(attn)
        context = torch.matmul(drop_attn, v)		# [b, heads, m, d_k]

        context = unshape(context)
        return torch.mean(context,dim=1), torch.mean(attn,dim=1)

class TSA(nn.Module):
    def __init__(self, interval, embedding_dim=128, output_embedding=False,output_size=19):
        super().__init__()
        self.pretrain_mlp = Pretrain_MLP(output_size=output_size)
        self.pretrain_embedding = SelfAttention_Embedding(interval,embedding_dim)
        self.output_embedding = output_embedding

    def forward(self, data):
        output, attn = self.pretrain_embedding(data)
        if self.output_embedding:
            return output, attn
        return self.pretrain_mlp(output), attn


class LPA(nn.Module):
    def __init__(self, interval, embedding_dim=128, output_embedding=False, old=False,output_size=19):
        super().__init__()
        self.pretrain_mlp = Pretrain_MLP(output_size=output_size)
        if old:
            self.pretrain_embedding = Attention_Embedding_(interval, embedding_dim)
        else:
            self.pretrain_embedding = Attention_Embedding(interval,embedding_dim)
        self.output_embedding = output_embedding

    def forward(self, data):
        output, attn = self.pretrain_embedding(data)
        if self.output_embedding:
            return output, attn

        return self.pretrain_mlp(output), attn