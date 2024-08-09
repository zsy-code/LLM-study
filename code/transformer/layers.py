import torch.nn as nn
import torch.nn.functional as F
import torch
from transformer.attns import MultiHeadAttention
import numpy as np
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        
        # 创建一个位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x



class PositionwiseFeedForward(nn.Module):
    """ The Position Wise Feed Forward Layer in Transformer
    """
    def __init__(self, d_in, d_hidden, drop_out=0.1):
        super().__init__()
        self.l_1 = nn.Linear(d_in, d_hidden)
        self.l_2 = nn.Linear(d_hidden, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(drop_out)
    
    def forward(self, x):
        residual = x

        x = self.l_2(F.relu(self.l_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class EncodeLayer(nn.Module):
    """ The Encoder Layer in Transformer
    - 1. Multi-Head Self Attention
    - 2. Position Wise Feed Forward
    """
    def __init__(self, n_heads, d_model, d_k, d_v, d_ff, drop_out=0.1):
        # Multi-Head Self Attention
        self.mh_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, drop_out)
        # Posiiton Wise Feed Forward
        self.pos_ffn = PositionwiseFeedForward(d_model, d_ff, drop_out)


    def forward(self, encoder_input, attn_mask=None):
        output, enc_attn = self.mh_attn(encoder_input, encoder_input, encoder_input, attn_mask)
        output = self.pos_ffn(output)
        return output, enc_attn
    

class DecoderLayer(nn.Module):
    """ The Decoder Layer in Transformer
    - 1. Multi-Head Self Attention
    - 2. Multi-Head Encoder Attention
    - 3. Position Wise Feed Forward
    """
    def __init__(self, n_heads, d_model, d_k, d_v, d_ff, dropout=0.1):
        super().__init__()
        self.mh_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout)
        self.enc_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, decoder_input, encoder_output, dec_attn_mask=None, enc_attn_mask=None):
        # Multi-Head Self Attention
        output, dec_msk_attn = self.mh_attn(decoder_input, decoder_input, decoder_input, dec_attn_mask)
        # Multi-Head Encoder Attention
        output, dec_enc_attn = self.enc_attn(output, encoder_output, encoder_output, enc_attn_mask)
        # Position Wise Feed Forward
        output = self.pos_ffn(output)
        return output, dec_msk_attn, dec_enc_attn