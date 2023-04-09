import codecs
import copy
import re
from locale import normalize

from matplotlib import pyplot as plt
from torch.autograd import Variable
import pandas as pd
import torch.nn.functional as F
import torch
import torch.nn as nn
import datetime
import time
from language import Lang
import random
import pickle
import numpy as np
from tqdm import tqdm
import os

"""
for p in data:
    p1 = []
    p2 = []
    cure = p[0]
    med = p[1]

    for i in cure:
        p1.append(cure_lang.id2word[i])
    for j in med:
        p2.append(med_lang.id2word[j])
    result1 = ''.join(c for c in p1)
    result2 = ' '.join(c for c in p2)
    with open("test.txt", "a", encoding='utf-8') as f:
        f.write(result1 + '\t\t')
        f.write(result2 + '\n')

with open('test.txt', 'r', encoding='utf-8') as f1:
    content = f1.read()
content = content.replace('EOS', '')
with open('../../英法翻译/test1.txt', 'w', encoding='utf-8') as f2:
    f2.write(content)
"""
ngpu = 1
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if (use_cuda and ngpu > 0) else "cpu")

data, test_data_text, med_lang, cure_lang = pickle.load(open('prescription_pairs_data6.pkl', 'rb'))
random.seed(2)
random.shuffle(data)
train_data = data[:int(len(data) * 0.98)]
# dev_data = data[int(len(data) * 0.9):int(len(data) * 0.95)]
dev_data = data[int(len(data) * 0.98):]


def padding(sequence, length):
    sequence = sequence[:length]
    while len(sequence) < length:
        sequence.append(0)
    return sequence


def make_batches(data, batch_size):
    batch_num = len(data) // batch_size if len(data) % batch_size == 0 else len(data) // batch_size + 1
    batches = []
    for batch in range(batch_num):
        mini_batch = data[batch * batch_size:(batch + 1) * batch_size]
        mini_batch = sorted(mini_batch, key=lambda t: len(t[0]), reverse=True)  # 对数据进行了降序排列，按照原来的长度
        en_max_len = max([len(p[0]) for p in mini_batch])
        de_max_len = max([len(p[1]) for p in mini_batch])
        en_mini_batch = [padding(p[0], en_max_len) for p in mini_batch]
        de_mini_batch = [padding(p[1], de_max_len) for p in mini_batch]
        batches.append((en_mini_batch, de_mini_batch))
    return batches


def get_angles(pos, i, d_model):
    # 2*(i//2)保证了2i，这部分计算的是1/10000^(2i/d)
    angle_rates = 1 / np.power(10000, 2 * (i // 2) / np.float32(d_model))  # => [1, 512]
    return pos * angle_rates  # [50,1]*[1,512]=>[50, 512]


# np.arange()函数返回一个有终点和起点的固定步长的排列，如[1,2,3,4,5]，起点是1，终点是5，步长为1
# 注意：起点终点是左开右闭区间，即start=1,end=6，才会产生[1,2,3,4,5]
# 只有一个参数时，参数值为终点，起点取默认值0，步长取默认值1。
def positional_encoding(position, d_model):  # d_model是位置编码的长度，相当于position encoding的embedding_dim？
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],  # [50, 1]
                            np.arange(d_model)[np.newaxis, :],  # [1, d_model=512]
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])  # 2i
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])  # 2i+2

    pos_encoding = angle_rads[np.newaxis, ...]  # [50,512]=>[1,50,512]
    return torch.tensor(pos_encoding, dtype=torch.float32)


pad = 0


def create_padding_mask(seq):  # seq [b, seq_len]
    # seq = torch.eq(seq, torch.tensor(0)).float() # pad=0的情况
    seq = torch.eq(seq, torch.tensor(pad)).float()  # pad!=0
    return seq[:, np.newaxis, np.newaxis, :]  # =>[b, 1, 1, seq_len]


# torch.triu(tensor, diagonal=0) 求上三角矩阵，diagonal默认为0表示主对角线的上三角矩阵
# diagonal>0，则主对角上面的第|diagonal|条次对角线的上三角矩阵
# diagonal<0，则主对角下面的第|diagonal|条次对角线的上三角矩阵
def create_look_ahead_mask(size):  # seq_len
    mask = torch.triu(torch.ones((size, size)), diagonal=1)
    # mask = mask.device() #
    return mask  # [seq_len, seq_len]


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    #计算注意力权重。
    q, k, v 必须具有匹配的前置维度。 且dq=dk
    k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
    #虽然 mask 根据其类型（填充或前瞻）有不同的形状，
    #但是 mask 必须能进行广播转换以便求和。
    #参数:
        q: 请求的形状 == (..., seq_len_q, depth)
        k: 主键的形状 == (..., seq_len_k, depth)
        v: 数值的形状 == (..., seq_len_v, depth_v)  seq_len_k = seq_len_v
        mask: Float 张量，其形状能转换成
              (..., seq_len_q, seq_len_k)。默认为None。
    #返回值:
        #输出，注意力权重
    """
    # matmul(a,b)矩阵乘:a b的最后2个维度要能做乘法，即a的最后一个维度值==b的倒数第2个纬度值，
    # 除此之外，其他维度值必须相等或为1（为1时会广播）
    matmul_qk = torch.matmul(q, k.transpose(-1, -2))  # 矩阵乘 =>[..., seq_len_q, seq_len_k]
    # 缩放matmul_qk
    dk = torch.tensor(k.shape[-1], dtype=torch.float32)  # k的深度dk，或叫做depth_k
    scaled_attention_logits = matmul_qk / torch.sqrt(dk)  # [..., seq_len_q, seq_len_k]
    # 将 mask 加入到缩放的张量上(重要！)
    if mask is not None:  # mask: [b, 1, 1, seq_len]
        # mask=1的位置是pad，乘以-1e9（-1*10^9）成为负无穷，经过softmax后会趋于0
        scaled_attention_logits += (mask * -1e9)

    # softmax 在最后一个轴（seq_len_k）上归一化
    attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)  # [..., seq_len_q, seq_len_k]

    output = torch.matmul(attention_weights, v)  # =>[..., seq_len_q, depth_v]
    return output, attention_weights  # [..., seq_len_q, depth_v], [..., seq_len_q, seq_len_k]


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0  # 因为输入要被（平均？）split到不同的head

        self.depth = d_model // self.num_heads  # 512/8=64，所以在scaled dot-product atten中dq=dk=64,dv也是64

        self.wq = torch.nn.Linear(d_model, d_model)
        self.wk = torch.nn.Linear(d_model, d_model)
        self.wv = torch.nn.Linear(d_model, d_model)

        self.final_linear = torch.nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):  # x [b, seq_len, d_model]
        x = x.view(batch_size, -1, self.num_heads,
                   self.depth)  # [b, seq_len, d_model=512]=>[b, seq_len, num_head=8, depth=64]
        return x.transpose(1, 2)  # [b, seq_len, num_head=8, depth=64]=>[b, num_head=8, seq_len, depth=64]

    def forward(self, q, k, v, mask):  # q=k=v=x [b, seq_len, embedding_dim] embedding_dim其实也=d_model
        batch_size = q.shape[0]

        q = self.wq(q)  # =>[b, seq_len, d_model]
        k = self.wk(k)  # =>[b, seq_len, d_model]
        v = self.wq(v)  # =>[b, seq_len, d_model]

        q = self.split_heads(q, batch_size)  # =>[b, num_head=8, seq_len, depth=64]
        k = self.split_heads(k, batch_size)  # =>[b, num_head=8, seq_len, depth=64]
        v = self.split_heads(v, batch_size)  # =>[b, num_head=8, seq_len, depth=64]

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        # => [b, num_head=8, seq_len_q, depth=64], [b, num_head=8, seq_len_q, seq_len_k]

        scaled_attention = scaled_attention.transpose(1, 2)  # =>[b, seq_len_q, num_head=8, depth=64]
        # 转置操作让张量存储结构扭曲，直接使用view方法会失败，可以使用reshape方法
        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)  # =>[b, seq_len_q, d_model=512]

        output = self.final_linear(concat_attention)  # =>[b, seq_len_q, d_model=512]
        return output, attention_weights  # [b, seq_len_q, d_model=512], [b, num_head=8, seq_len_q, seq_len_k]


# 点式前馈网络
def point_wise_feed_forward_network(d_model, dff):
    feed_forward_net = torch.nn.Sequential(
        torch.nn.Linear(d_model, dff),  # [b, seq_len, d_model]=>[b, seq_len, dff=2048]
        torch.nn.ReLU(),
        torch.nn.Linear(dff, d_model),  # [b, seq_len, dff=2048]=>[b, seq_len, d_model=512]
    )
    return feed_forward_net


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)  # 多头注意力（padding mask）(self-attention)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        # self.conv1 = torch.nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)  # 后续可以调整
        self.layernorm1 = torch.nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(normalized_shape=d_model, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(rate)
        self.dropout2 = torch.nn.Dropout(rate)

    # x [b, inp_seq_len, embedding_dim] embedding_dim其实也=d_model
    # mask [b,1,1,inp_seq_len]
    def forward(self, x, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # =>[b, seq_len, d_model]
        attn_output = self.dropout1(attn_output)
        conv_output = self.conv1(x.permute(0, 2, 1))
        conv_output = conv_output.permute(0, 2, 1)
        out1 = self.layernorm1(x + attn_output)  # 残差&层归一化 =>[b, seq_len, d_model]

        ffn_output = self.ffn(out1)  # =>[b, seq_len, d_model]
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  # 残差&层归一化 =>[b, seq_len, d_model]

        return out2  # [b, seq_len, d_model]


class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model,
                                       num_heads)  # masked的多头注意力（look ahead mask 和 padding mask）(self-attention)
        self.mha2 = MultiHeadAttention(d_model, num_heads)  # 多头注意力（padding mask）(encoder-decoder attention)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = torch.nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm3 = torch.nn.LayerNorm(normalized_shape=d_model, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(rate)
        self.dropout2 = torch.nn.Dropout(rate)
        self.dropout3 = torch.nn.Dropout(rate)

    # x [b, targ_seq_len, embedding_dim] embedding_dim其实也=d_model=512
    # look_ahead_mask [b, 1, targ_seq_len, targ_seq_len] 这里传入的look_ahead_mask应该是已经结合了look_ahead_mask和padding mask的mask
    # enc_output [b, inp_seq_len, d_model]
    # padding_mask [b, 1, 1, inp_seq_len]
    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x,
                                               look_ahead_mask)  # =>[b, targ_seq_len, d_model], [b, num_heads,
        # targ_seq_len, targ_seq_len]
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(x + attn1)  # 残差&层归一化 [b, targ_seq_len, d_model]

        # Q: receives the output from decoder's first attention block，即 masked multi-head attention sublayer
        # K V: V (value) and K (key) receive the encoder output as inputs
        attn2, attn_weights_block2 = self.mha2(out1, enc_output, enc_output,
                                               padding_mask)  # =>[b, targ_seq_len, d_model], [b, num_heads,
        # targ_seq_len, inp_seq_len]
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(out1 + attn2)  # 残差&层归一化 [b, targ_seq_len, d_model]

        ffn_output = self.ffn(out2)  # =>[b, targ_seq_len, d_model]
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(out2 + ffn_output)  # 残差&层归一化 =>[b, targ_seq_len, d_model]

        return out3, attn_weights_block1, attn_weights_block2
        # [b, targ_seq_len, d_model], [b, num_heads, targ_seq_len, targ_seq_len], [b, num_heads, targ_seq_len,
        # inp_seq_len]


class Encoder(torch.nn.Module):
    def __init__(self,
                 num_layers,  # N个encoder layer
                 d_model,
                 num_heads,
                 dff,  # 点式前馈网络内层fn的维度
                 input_vocab_size,  # 输入词表大小（源语言（法语））
                 maximun_position_encoding,
                 rate=0.1):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model

        self.embedding = torch.nn.Embedding(num_embeddings=input_vocab_size, embedding_dim=d_model)
        self.pos_encoding = positional_encoding(maximun_position_encoding,
                                                d_model)  # =>[1, max_pos_encoding, d_model=512]

        # self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate).cuda() for _ in range(num_layers)] # 不行
        self.enc_layers = torch.nn.ModuleList([EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)])

        self.dropout = torch.nn.Dropout(rate)

    def load_emb(self, emb):
        emb = torch.from_numpy(np.array(emb, dtype=np.float32))
        self.embedding.weight.data.copy_(emb)

    # x [b, inp_seq_len]
    # mask [b, 1, 1, inp_sel_len]
    def forward(self, x, mask):
        inp_seq_len = x.shape[-1]

        # adding embedding and position encoding
        x = self.embedding(x)  # [b, inp_seq_len]=>[b, inp_seq_len, d_model]
        # 缩放 embedding 原始论文的3.4节有提到： In the embedding layers, we multiply those weights by \sqrt{d_model}.
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        pos_encoding = self.pos_encoding[:, :inp_seq_len, :]
        pos_encoding = pos_encoding.cuda()
        x += pos_encoding  # [b, inp_seq_len, d_model]

        x = self.dropout(x)
        """在进入多头注意力钱先层归一化,"""
        # enc_output = self.layer_norm(enc_output)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)  # [b, inp_seq_len, d_model]=>[b, inp_seq_len, d_model]
        return x  # [b, inp_seq_len, d_model]


class Decoder(torch.nn.Module):
    def __init__(self,
                 num_layers,  # N个encoder layer
                 d_model,
                 num_heads,
                 dff,  # 点式前馈网络内层fn的维度
                 target_vocab_size,  # target词表大小（目标语言（英语））
                 maximun_position_encoding,
                 rate=0.1):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.target_vocab_size = target_vocab_size
        self.embedding = torch.nn.Embedding(num_embeddings=target_vocab_size, embedding_dim=d_model)
        self.pos_encoding = positional_encoding(maximun_position_encoding,
                                                d_model)  # =>[1, max_pos_encoding, d_model=512]

        # self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate).cuda() for _ in range(num_layers)] # 不行
        self.dec_layers = torch.nn.ModuleList([DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)])

        self.dropout = torch.nn.Dropout(rate)

    # x [b, targ_seq_len]
    # look_ahead_mask [b, 1, targ_seq_len, targ_seq_len] 这里传入的look_ahead_mask应该是已经结合了look_ahead_mask和padding mask的mask
    # enc_output [b, inp_seq_len, d_model]
    # padding_mask [b, 1, 1, inp_seq_len]
    def load_emb(self, emb):
        emb = torch.from_numpy(np.array(emb, dtype=np.float32))
        self.embedding.weight.data.copy_(emb)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        targ_seq_len = x.shape[-1]

        attention_weights = {}

        # adding embedding and position encoding
        x = self.embedding(x)  # [b, targ_seq_len]=>[b, targ_seq_len, d_model]
        # 缩放 embedding 原始论文的3.4节有提到： In the embedding layers, we multiply those weights by \sqrt{d_model}.
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        # x += self.pos_encoding[:, :targ_seq_len, :]  # [b, targ_seq_len, d_model]
        pos_encoding = self.pos_encoding[:, :targ_seq_len, :]  # [b, targ_seq_len, d_model]
        pos_encoding = pos_encoding.cuda()
        # x += pos_encoding  # [b, inp_seq_len, d_model]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x, attn_block1, attn_block2 = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)
            # => [b, targ_seq_len, d_model], [b, num_heads, targ_seq_len, targ_seq_len], [b, num_heads, targ_seq_len, inp_seq_len]

            attention_weights[f'decoder_layer{i + 1}_block1'] = attn_block1
            attention_weights[f'decoder_layer{i + 1}_block2'] = attn_block2

        return x, attention_weights
        # => [b, targ_seq_len, d_model],
        # {'..block1': [b, num_heads, targ_seq_len, targ_seq_len],
        #  '..block2': [b, num_heads, targ_seq_len, inp_seq_len], ...}


class Transformer(torch.nn.Module):
    def __init__(self,
                 num_layers,  # N个encoder layer
                 d_model,
                 num_heads,
                 dff,  # 点式前馈网络内层fn的维度
                 input_vocab_size,  # input此表大小（源语言（法语））
                 target_vocab_size,  # target词表大小（目标语言（英语））
                 pe_input,  # input max_pos_encoding
                 pe_target,  # input max_pos_encoding
                 rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers,
                               d_model,
                               num_heads,
                               dff,
                               input_vocab_size,
                               pe_input,
                               rate)
        self.decoder = Decoder(num_layers,
                               d_model,
                               num_heads,
                               dff,
                               target_vocab_size,
                               pe_target,
                               rate)
        self.final_layer = torch.nn.Linear(d_model, target_vocab_size)
        self.initparam()

        self.p_gen = nn.Sequential(
            nn.Linear(self.decoder.d_model * 3, 1),
            nn.Sigmoid())

    # inp [b, inp_seq_len]
    # targ [b, targ_seq_len]
    # enc_padding_mask [b, 1, 1, inp_seq_len]
    # look_ahead_mask [b, 1, targ_seq_len, targ_seq_len]
    # dec_padding_mask [b, 1, 1, inp_seq_len] # 注意这里的维度是inp_seq_len

    def initparam(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, inp, targ, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, enc_padding_mask)  # =>[b, inp_seq_len, d_model]

        dec_output, attention_weights = self.decoder(targ, enc_output, look_ahead_mask, dec_padding_mask)
        # => [b, targ_seq_len, d_model],
        # {'..block1': [b, num_heads, targ_seq_len, targ_seq_len],
        #  '..block2': [b, num_heads, targ_seq_len, inp_seq_len], ...}
        x = self.decoder.embedding(targ)  # [b, targ_seq_len]=>[b, targ_seq_len, d_model]
        x *= torch.sqrt(torch.tensor(self.decoder.d_model, dtype=torch.float32))
        tgt_embed = self.decoder.pos_encoding[:, :x.size(1), :]  # [1, targ_seq_len, d_model]
        tgt_embed = tgt_embed.to(device)
        x += tgt_embed
        attention = attention_weights[f'decoder_layer{num_layers}_block2']

        c = []
        for i in range(attention.size(0)):
            b = torch.zeros(attention.size(2), attention.size(3)).to(device)
            for j in range(attention.size(1)):
                b += attention[i][j]
            b = b / attention.size(1)
            b = b.unsqueeze(0)
            c.append(b)
        final_attention = c[0]
        for i in range(1, len(c)):
            final_attention = torch.cat([final_attention, c[i]], dim=0)
        attention = final_attention  # (b, tra_seq_len, inp_seq_len)
        p_vocab = self.final_layer(dec_output)  # =>[b, targ_seq_len, target_vocab_size]
        p_vocab = F.softmax(p_vocab, dim=2)
        hidden_state = enc_output  # (b, inp_seq_len, d_model)
        attn = torch.bmm(dec_output, hidden_state.transpose(1, 2))
        attn = F.softmax(attn, dim=2)
        context_vectors = torch.bmm(attn, hidden_state)
        context_vector1 = torch.matmul(attention, hidden_state)  # (b ,tra_seq_len, d_model)
        total_states = torch.cat((context_vectors, context_vector1, x), dim=-1)  # (b, tra_seq_len, 2 * d_model)
        p_gen = self.p_gen(total_states)  # (b, tra_seq_len,1)
        p_copy = 1 - p_gen

        one_hot = torch.zeros(targ.size(0), targ.size(1), self.decoder.target_vocab_size).to(device)
        one_hot = one_hot.scatter(-1, targ.unsqueeze(-1), 1)

        final_output = torch.add(p_gen * p_vocab, p_copy * one_hot)  # (b,t,v)
        return final_output, attention_weights
        # [b, targ_seq_len, target_vocab_size]
        # {'..block1': [b, num_heads, targ_seq_len, targ_seq_len],
        #  '..block2': [b, num_heads, targ_seq_len, inp_seq_len], ...}



input_vocab_size = len(cure_lang.word2id)
target_vocab_size = len(med_lang.word2id)
dropout_rate = 0.1
EPOCHS = 20  # 50 # 30  # 20

print_trainstep_every = 60  


class CustomSchedule(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warm_steps=4):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warm_steps

        super(CustomSchedule, self).__init__(optimizer)

    def get_lr(self):
        """
        # rsqrt 函数用于计算 x 元素的平方根的倒数.  即= 1 / sqrt{x}
        arg1 = torch.rsqrt(torch.tensor(self._step_count, dtype=torch.float32))
        arg2 = torch.tensor(self._step_count * (self.warmup_steps ** -1.5), dtype=torch.float32)
        dynamic_lr = torch.rsqrt(self.d_model) * torch.minimum(arg1, arg2)
        """
        # print('*'*27, self._step_count)
        arg1 = self._step_count ** (-0.5)
        arg2 = self._step_count * (self.warmup_steps ** -1.5)
        dynamic_lr = (self.d_model ** (-0.5)) * min(arg1, arg2)
        # print('dynamic_lr:', dynamic_lr)
        return [dynamic_lr for group in self.optimizer.param_groups]


# 'none'表示直接返回b个样本的loss，默认求平均
loss_object = torch.nn.CrossEntropyLoss(reduction='none')


def cal_loss(pred, gold, smoothing=False, soft=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    b = pred.size(0)
    pred = pred.reshape(-1, pred.size(2))
    gold = gold.reshape(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)  # 按照索引给one_hot里面加1
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.view(b, -1)
    if soft:
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)  # 普通的q_t
        all_med = F.softmax(one_hot)
        log_prb = F.log_softmax(pred, dim=1)
        q_t_2 = (one_hot + all_med) / 2
        loss = -(q_t_2 * log_prb)
        out_weight = Variable(torch.ones(len(med_lang.word2id)))
        out_weight[0] = 0.
        out_weight = torch.unsqueeze(out_weight, 0)
        out_weight = out_weight.to(device)
        final_loss = (loss * out_weight).sum(dim=1)
        loss = final_loss.view(b, -1)
    else:
        loss = loss_object(pred, gold)
        loss = loss.view(b, -1)
    return loss


def mask_loss_func(real, pred):
    """# real [b, targ_seq_len]
# pred [b, targ_seq_len, target_vocab_size]"""
    # print(real.shape, pred.shape)
    _loss = cal_loss(pred, real, smoothing=False, soft=True)  # [b, targ_seq_len]
    # _loss = loss_object(pred.transpose(-1, -2), real)  # [b, targ_seq_len],且是否用标签平滑

    # logical_not  取非
    # mask 每个元素为bool值，如果real中有pad，则mask相应位置就为False
    # mask = torch.logical_not(real.eq(0)).type(_loss.dtype) # [b, targ_seq_len] pad=0的情况
    mask = torch.logical_not(real.eq(pad)).type(_loss.dtype)  # [b, targ_seq_len] pad!=0的情况

    
    _loss *= mask
    return _loss.sum() / mask.sum().item()


def mask_accuracy_func(real, pred):
    """# real [b, targ_seq_len]
# pred [b, targ_seq_len, target_vocab_size]"""

    _pred = pred.argmax(dim=-1)  # [b, targ_seq_len, target_vocab_size]=>[b, targ_seq_len]

    candidates = torch.logical_not(_pred.eq(pad) + _pred.eq(med_lang.word2id['EOS']))
    correct_med = 0
    for b in range(_pred.size(0)):
        for i in range(_pred.size(1)):
            if i == 0:
                if _pred[b][i] != med_lang.word2id['EOS'] and _pred[b][i] != pad and _pred[b][i] in real[b]:
                    correct_med += 1
            else:
                if _pred[b][i] in _pred[b][:i]:
                    correct_med = correct_med
                else:
                    if _pred[b][i] != med_lang.word2id['EOS'] and _pred[b][i] != pad and _pred[b][i] in real[b]:
                        correct_med += 1
    total_ref = torch.logical_not(real.eq(pad) + real.eq(med_lang.word2id['EOS']))
    total_can = candidates.sum().item()
    total_can = total_can if total_can != 0 else 1
    
    precision = float(correct_med) / total_can
    recall = float(correct_med) / float(total_ref.sum().item())
    if precision == 0 or recall == 0:
        F = 0.
    else:
        F = precision * recall * 2. / (precision + recall)
    return precision, recall, F


def create_mask(inp, targ):
    """# inp [b, inp_seq_len] 序列已经加入pad填充
# targ [b, targ_seq_len] 序列已经加入pad填充"""
    # encoder padding mask
    enc_padding_mask = create_padding_mask(inp)  # =>[b,1,1,inp_seq_len] mask=1的位置为pad

    # decoder's first attention block(self-attention)
    # 使用的padding create_mask & look-ahead create_mask
    look_ahead_mask = create_look_ahead_mask(targ.shape[-1])  # =>[targ_seq_len,targ_seq_len] ##################
    dec_targ_padding_mask = create_padding_mask(targ)  # =>[b,1,1,targ_seq_len]
    combined_mask = torch.max(look_ahead_mask, dec_targ_padding_mask)  # 结合了2种mask =>[b,1,targ_seq_len,targ_seq_len]

    # decoder's second attention block(encoder-decoder attention) 使用的padding create_mask
    # 【注意】：这里的mask是用于遮挡encoder output的填充pad，而encoder的输出与其输入shape都是[b,inp_seq_len,d_model]
    # 所以这里mask的长度是inp_seq_len而不是targ_mask_len
    dec_padding_mask = create_padding_mask(inp)  # =>[b,1,1,inp_seq_len] mask=1的位置为pad

    return enc_padding_mask, combined_mask, dec_padding_mask
    # [b,1,1,inp_seq_len], [b,1,targ_seq_len,targ_seq_len], [b,1,1,inp_seq_len]


save_dir = 'save/'

transformer = Transformer(num_layers,
                          d_model,
                          num_heads,
                          dff,
                          input_vocab_size,
                          target_vocab_size,
                          pe_input=input_vocab_size,
                          pe_target=target_vocab_size,
                          rate=dropout_rate)
transformer = transformer.to(device)
'''
if ngpu > 1:  # 并行化
    transformer = torch.nn.DataParallel(transformer, device_ids=list(range(ngpu)))  # 设置并行执行  device_ids=[0,1]
'''
# optimizer = torch.optim.RMSprop(transformer.parameters(), lr=0, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0,
# centered=False)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-8)
lr_scheduler = CustomSchedule(optimizer, d_model, warm_steps=4000)
# inp [b,inp_seq_len]
# targ [b,targ_seq_len]
"""
拆分targ, 例如：sentence = "SOS A lion in the jungle is sleeping EOS"
tar_inp = "<start>> A lion in the jungle is sleeping"
tar_real = "A lion in the jungle is sleeping <end>"
"""
train_batches = make_batches(train_data, batch_size)
dev_batches = make_batches(dev_data, batch_size)


def train_step(model, inp, targ):
    # 目标（target）被分成了 tar_inp 和 tar_real
    # tar_inp 作为输入传递到解码器。
    # tar_real 是位移了 1 的同一个输入：在 tar_inp 中的每个位置，tar_real 包含了应该被预测到的下一个标记（token）。
    targ_inp = targ[:, :-1]
    targ_real = targ[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_mask(inp, targ_inp)
    inp = inp.to(device)
    targ_inp = targ_inp.to(device)
    targ_real = targ_real.to(device)
    enc_padding_mask = enc_padding_mask.to(device)
    combined_mask = combined_mask.to(device)
    dec_padding_mask = dec_padding_mask.to(device)

    model.train()  # 设置train mode

    optimizer.zero_grad()  # 梯度清零

    # forward
    prediction, _ = transformer(inp, targ_inp, enc_padding_mask, combined_mask, dec_padding_mask)
    # [b, targ_seq_len, target_vocab_size]
    # {'..block1': [b, num_heads, targ_seq_len, targ_seq_len],
    #  '..block2': [b, num_heads, targ_seq_len, inp_seq_len], ...}

    loss = mask_loss_func(targ_real, prediction)
    precision, recall, F = mask_accuracy_func(targ_real, prediction)

    # backward
    loss.backward()  # 反向传播计算梯度
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)  # 防止梯度爆炸
    optimizer.step()  # 更新参数

    return loss.item(), prediction
    # , precision, recall, F


def validate_step(model, inp, targ):
    targ_inp = targ[:, :-1]
    targ_real = targ[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_mask(inp, targ_inp)
    inp = inp.to(device)
    targ_inp = targ_inp.to(device)
    targ_real = targ_real.to(device)
    enc_padding_mask = enc_padding_mask.to(device)
    combined_mask = combined_mask.to(device)
    dec_padding_mask = dec_padding_mask.to(device)

    model.eval()  # 设置eval mode

    with torch.no_grad():
        # forward
        prediction, _ = model(inp, targ_inp, enc_padding_mask, combined_mask, dec_padding_mask)
        # [b, targ_seq_len, target_vocab_size]
        # {'..block1': [b, num_heads, targ_seq_len, targ_seq_len],
        #  '..block2': [b, num_heads, targ_seq_len, inp_seq_len], ...}

        val_loss = mask_loss_func(targ_real, prediction)
        val_precision, val_recall, val_F = mask_accuracy_func(targ_real, prediction)

    return val_loss.item(), prediction,
    # val_precision, val_recall, val_F


metric_name = 'precision'
# df_history = pd.DataFrame(columns=['epoch', 'loss', metric_name]) # 记录训练历史信息
df_history = pd.DataFrame(
    columns=['epoch', 'loss', 'val_loss', 'val_' + metric_name, 'recall', 'F'])


# 打印时间
def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m_%d %H:%M:%S')
    print('\n' + "==========" * 8 + '%s' % nowtime)


def train_model(model, epochs, train_batches, dev_batches, print_every):
    starttime = time.time()
    print('*' * 27, 'start training...')
    printbar()

    best_acc = -1
    for epoch in range(1, epochs + 1):

        loss_sum = 0.
        metric_sum = 0.
        final_pred = None
        final_ref = None

        for step, (inp, targ) in enumerate(train_batches, start=1):
            # inp [64, 10] , targ [64, 10]
            src = np.array([np.array(s, dtype=np.int64) for s in inp], dtype=np.int64)
            tgt = np.array([np.array(s, dtype=np.int64) for s in targ], dtype='int64')
            inp = Variable(torch.from_numpy(src))
            targ = Variable(torch.from_numpy(tgt))
            loss, prediction = train_step(model, inp, targ)
            pred = prediction.argmax(dim=-1)
            loss_sum += loss
            '''
            先只算损失，不然时间太长
            if step == 1:
                final_pred = pred
                final_ref = targ[:, 1:]
            else:
                final_pred = torch.cat((final_pred, pred), dim=0)
                final_ref = torch.cat((final_ref, targ[:, 1:]))
            # 打印batch级别日志
            if step % print_every == 0:
                print('*' * 8, f'[step = {step}] loss: {loss_sum :.3f}')
            '''
            lr_scheduler.step()  # 更新学习率
        "没写训练集的准确率和recall，F"
        # 一个epoch的train结束，做一次验证
        # test(model, train_dataloader)
        val_loss_sum = 0.
        val_prcision_sum = 0
        val_recall_sum = 0
        val_F_sum = 0
        totalnum = 0
        val_final_pred = None
        val_final_ref = None
        for val_step, (inp, targ) in enumerate(dev_batches, start=1):
            # inp [64, 10] , targ [64, 10]

            src = np.array([np.array(s, dtype=np.int64) for s in inp], dtype=np.int64)
            tgt = np.array([np.array(s, dtype=np.int64) for s in targ], dtype='int64')
            inp = Variable(torch.from_numpy(src))
            targ = Variable(torch.from_numpy(tgt))
            totalnum += inp.size(0)
            loss, val_prediction = validate_step(model, inp, targ)
            val_loss_sum += loss
            pred1 = val_prediction.argmax(dim=-1)
            pred1 = pred1.to(device)
            referance = targ[:, 1:].to(device)
            correct_med = 0
            can_num = 0
            ref_num = 0
            for b in range(pred1.size(0)):
                for i in range(pred1.size(1)):
                    if pred1[b][i] != med_lang.word2id['EOS'] and pred1[b][i] != pad:
                        can_num += 1
                    if i == 0:
                        if pred1[b][i] != med_lang.word2id['EOS'] and pred1[b][i] != pad and \
                                pred1[b][i] in referance[b]:
                            correct_med += 1
                    else:
                        if pred1[b][i] in pred1[b][:i]:
                            correct_med = correct_med
                        else:
                            if pred1[b][i] != med_lang.word2id['EOS'] and pred1[b][i] != pad and \
                                    pred1[b][i] in referance[b]:
                                correct_med += 1
                for j in range(referance.size(1)):
                    if referance[b][j] != med_lang.word2id['EOS'] and referance[b][j] != pad:
                        ref_num += 1
                can_num = can_num if can_num != 0 else 1
                ref_num = ref_num if ref_num != 0 else 1
                precision = correct_med / can_num
                recall = correct_med / ref_num
                if precision == 0 or recall == 0:
                    F = 0.
                else:
                    F = precision * recall * 2. / (precision + recall)
                val_prcision_sum += precision
                val_recall_sum += recall
                val_F_sum += F

        # 记录和收集1个epoch的训练（和验证）信息
        # record = (epoch, loss_sum/step, metric_sum/step)
        record = (epoch, loss_sum, val_loss_sum,
                  val_prcision_sum / totalnum, val_recall_sum / totalnum, val_F_sum / totalnum)
        df_history.loc[epoch - 1] = record

        # 打印epoch级别的日志
        print('EPOCH = {} loss: {:.3f} val_loss: {:.3f}, val_{}: {:.3f} '
              'recall: {:.3f} F: {:.3f}'.format(
            record[0], record[1], record[2], metric_name, record[3] * 100, record[4] * 100,
                                                          record[5] * 100))

        printbar()

        # 保存模型
        current_acc_avg = val_F_sum / totalnum  # 看验证集指标
        if current_acc_avg > best_acc:  # 保存更好的模型
            best_acc = current_acc_avg
            checkpoint = save_dir + '{:03d}_{:.2f}_ckpt.tar'.format(epoch, current_acc_avg)

            model_sd = copy.deepcopy(model.state_dict())
            torch.save({
                'loss': loss_sum,
                'epoch': epoch,
                'net': model_sd,
                'opt': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }, checkpoint)

    print('finishing training...')
    endtime = time.time()
    time_elapsed = endtime - starttime
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return df_history


'''
transformer.encoder.load_emb(cure_lang.emb)
transformer.decoder.load_emb(med_lang.emb)

df_history = train_model(transformer, EPOCHS, train_batches, dev_batches, print_trainstep_every)
'''

checkpoint = save_dir + '017_0.24_ckpt.tar'

# ckpt = torch.load(checkpoint, map_location=device)  # dict  save 在 CPU 加载到GPU
ckpt = torch.load(checkpoint)  # dict  save 在 GPU 加载到 GPU
# print('ckpt', ckpt)
transformer_sd = ckpt['net']
# optimizer_sd = ckpt['opt'] # 不重新训练的话不需要
# lr_scheduler_sd = ckpt['lr_scheduler']
# optimizer.load_state_dict(optimizer_sd)
# lr_scheduler.load_state_dict(lr_scheduler_sd)
reload_model = Transformer(num_layers,
                           d_model,
                           num_heads,
                           dff,
                           input_vocab_size,
                           target_vocab_size,
                           pe_input=input_vocab_size,
                           pe_target=target_vocab_size,
                           rate=dropout_rate)
reload_model = reload_model.to(device)
reload_model.load_state_dict(transformer_sd)


def tokenizer_encode(sentence):
    # print(type(vocab)) # torchtext.vocab.Vocab
    # print(len(vocab))
    sentence = [c for c in sentence]
    # print(type(sentence)) # str

    sentence = ['SOS'] + sentence + ['EOS']
    sentence_ids = []
    for token in sentence:
        if token in cure_lang.id2word:
            sentence_ids.append(cure_lang.word2id[token])
        else:
            sentence_ids.append(cure_lang.word2id['OOV'])
    # sentence_ids = [cure_lang.word2id[token] for token in sentence]
    # print(sentence_ids, type(sentence_ids[0])) # int
    return sentence_ids


def tokenzier_decode(sentence_ids):
    sentence = [med_lang.id2word[id] for id in sentence_ids if id < len(med_lang.id2word)]
    # print(sentence)
    return " ".join(sentence)


# inp_sentence 一个法语句子，例如"je pars en vacances pour quelques jours ."
def evaluate(model, inp_sentence):
    model.eval()  # 设置eval mode

    inp_sentence_ids = tokenizer_encode(inp_sentence)  # 转化为索引
    # print(tokenzier_decode(inp_sentence_ids, SRC_TEXT.vocab))
    encoder_input = torch.tensor(inp_sentence_ids).unsqueeze(dim=0)  # =>[b=1, inp_seq_len=10]
    # print(encoder_input.shape)

    decoder_input = [med_lang.word2id['SOS']]
    decoder_input = torch.tensor(decoder_input).unsqueeze(0)  # =>[b=1,seq_len=1]

    # print(decoder_input.shape)

    with torch.no_grad():
        for i in range(20):
            enc_padding_mask, combined_mask, dec_padding_mask = create_mask(encoder_input.cpu(), decoder_input.cpu())
            # [b,1,1,inp_seq_len], [b,1,targ_seq_len,inp_seq_len], [b,1,1,inp_seq_len]
            # forward
            encoder_input = encoder_input.to(device)
            decoder_input = decoder_input.to(device)
            enc_padding_mask = enc_padding_mask.to(device)
            combined_mask = combined_mask.to(device)
            dec_padding_mask = dec_padding_mask.to(device)

            predictions, attention_weights = model(encoder_input,
                                                   decoder_input,
                                                   enc_padding_mask,
                                                   combined_mask,
                                                   dec_padding_mask)
            # [b=1, targ_seq_len, target_vocab_size]
            # {'..block1': [b, num_heads, targ_seq_len, targ_seq_len],
            #  '..block2': [b, num_heads, targ_seq_len, inp_seq_len], ...}

            # 看最后一个词并计算它的 argmax
            prediction = predictions[:, -1:, :]  # =>[b=1, 1, target_vocab_size]
            prediction_id = torch.argmax(prediction, dim=-1).to(device)  # => [b=1, 1]
            # print('prediction_id:', prediction_id, prediction_id.dtype) # torch.int64
            if prediction_id.squeeze().item() == med_lang.word2id['EOS']:
                return decoder_input.squeeze(dim=0), attention_weights

            decoder_input = torch.cat([decoder_input, prediction_id],
                                      dim=-1)  # [b=1,targ_seq_len=1]=>[b=1,targ_seq_len=2]
            # decoder_input在逐渐变长

    return decoder_input.squeeze(dim=0), attention_weights



