import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def attention(q, k, v, d_k, mask=None, attn_dropout=None):
    # 借鉴一下PCT
    # scores = torch.matmul(q, k.transpose(-2, -1))
    # socres = F.softmax(scores, dim=-1)
    # scores = scores / (1e-9 + scores.sum(dim=1, keepdims=True))
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if attn_dropout is not None:
        scores = attn_dropout(scores)

    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        self.trans_conv = nn.Linear(d_model, d_model)
        self.after_norm = nn.BatchNorm1d(d_model)
        self.act = nn.ReLU()


    def forward(self, q, k, v, mask=None):
        x = q # (1, n, d_model)
        
        bs = q.size(0) # q: 1 * n' * c
        # bs = 1
        
        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k) 
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_k

        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.attn_dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous() \
            .view(bs, -1, self.d_model) # (1, n, d_model)
        # 使用了point cloud transformer的offset attention计算效果可能更好？
        x_r = x - concat
        x_r = self.act(self.after_norm(self.trans_conv(x_r).transpose(-1, -2))).transpose(-1, -2)
        # output = self.out(concat)
        # output = self.proj_dropout(output)
        # x = x + x_r
        return x_r

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=64, dropout = 0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn_norm = Norm(d_model)
        self.ff_norm = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, d_ff=d_ff)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # trans-unet的结构图可以借鉴，写论文时可以用
        # x: 1 * n' * d_model
        q = k = v = x
        x = self.attn_norm(x) # norm，对channel进行归一化
        x2 = self.attn(q, k ,v ,mask) # multi-head attn
        x = x + x2 # add
        
        x2 = self.ff_norm(x) # norm
        x2 = self.ff(x2) # ffn
        x = x + x2 # add
        return x



class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, d_ff=d_ff).cuda()

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs,
                                           src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x

import copy
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, xyz):
        xyz1 = xyz.unsqueeze(1) ### N * 1 * 3
        xyz2 = xyz.unsqueeze(0) ### 1 * N * 3
        pairwise_dist =xyz1 - xyz2 ### N * N * 3
        return pairwise_dist


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, N, heads, d_ff, rel_pos=True):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.d_ff = d_ff
        self.N = N
        self.rel_pos = rel_pos
        self.pe = PositionalEncoding(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads, d_ff=d_ff), N)
        self.norm = Norm(d_model)
        self.position_linear = nn.Linear(3, d_model)

    def forward(self, xyz, feats, batch_ids):
        batch_size = batch_ids.max().item() + 1
        assert feats.size(1) == self.d_model
        output = torch.zeros_like(feats)
        for i in range(batch_size):
            batch_id = (batch_ids==i).nonzero().squeeze(dim=1)
            if batch_id.size(0) == 0:
                continue
            batch_xyz = xyz[batch_id].view(-1, 3) ###n' * 3
            batch_features = feats[batch_id].view(-1, self.d_model) ### n' * d_model
            if self.rel_pos:
                pairwise_dist = self.pe(batch_xyz) ### n' * n' * 3
                pairwise_dist = pairwise_dist.mean(dim=1) ### n' * 3
                position_embedding = self.position_linear(pairwise_dist) # n' * d_model
                x = (batch_features + position_embedding).unsqueeze(dim=0) # 1, n', d_model
            else:
                x = batch_features.unsqueeze(dim=0)
            for i in range(self.N):
                x = self.layers[i](x, mask=None)

            x = self.norm(x)
            x = x.squeeze(dim=0)
            output[batch_id] = x
        return output
