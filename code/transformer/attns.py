import torch.nn as nn
import torch
from torch.nn import functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention mechanism, a modification of the standard dot product attention mechanism.
    The input is divided by sqrt(d_k) and scaled by a softmax function to obtain the attention weights.
    This attention layer is used in the encoder-decoder attention layers.
    """
    def __init__(self, temperature, dropout=0.1, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.temperature = temperature

    def forward(self, q, k, v, mask=None):
        """
        :param q: query tensor. shape = (batch_size, n_heads, seq_len_q, d_k)
        :param k: key tensor. shape = (batch_size, n_heads, seq_len_kv, d_k)
        :param v: value tensor. shape = (batch_size, n_heads, seq_len_v, d_v)
        :calculate: attn = softmax(qk^T / sqrt(d_k))v
        :return: output tensor. shape = (batch_size, n_heads, seq_len_q, d_v)
        """

        # first calculate the dot product of the query with all keys
        # divide by sqrt(d_k) and apply a softmax function to obtain the attention weights
        attention = torch.matmul(q / self.temperature, k.transpose(-1, -2))
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # calculate the weighted sum of values according to attention weights
        output = torch.matmul(attention_weights, v)
        return output, attention_weights
    

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism, a modification of the standard dot product attention mechanism.
    The input is divided by sqrt(d_k) and scaled by a softmax function to obtain the attention weights.
    This attention layer is used in the encoder-decoder attention layers.
    """
    def __init__(self, n_heads, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_heads * d_k)
        self.w_ks = nn.Linear(d_model, n_heads * d_k)
        self.w_vs = nn.Linear(d_model, n_heads * d_v)
        self.fc = nn.Linear(n_heads * d_v, d_model)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        """
        :param q: query tensor. shape = (batch_size, seq_len_q, d_model)
        :param k: key tensor. shape = (batch_size, seq_len_kv, d_model)
        :param v: value tensor. shape = (batch_size, seq_len_v, d_model)
        :return: output tensor. shape = (batch_size, seq_len_q, d_model)
        """
        d_k, d_v, n_heads = self.d_k, self.d_v, self.n_heads
        bsz, len_q, len_k, len_v = q.size(0), k.size(1), v.size(1), v.size(1)

        # used for summation in the residual connection
        residual = q
        
        # linear projection of the input vectors to the multi-head attention heads
        # from (bsz, seq_len, d_model) to (bsz, n_heads, seq_len, d_qkv)
        # 1. transform the original shape to (bsz, seq_len, n_heads, d_qkv) 
        #    - because The raw data is arranged in the order of "batch-seq_len-features."
        #    - directly transposing from (bsz, seq_len, n_heads * d_qkv) to (bsz, n_heads, seq_len, d_qkv) is not feasible because n_heads * d_qkv is a single integrated dimension.
        # 2. transpose to (bsz, n_heads, seq_len, d_qkv)
        q = self.w_qs(q).view(bsz, len_q, n_heads, d_k).transpose(1, 2)
        k = self.w_ks(k).view(bsz, len_k, n_heads, d_k).transpose(1, 2)
        v = self.w_vs(v).view(bsz, len_v, n_heads, d_v).transpose(1, 2)

        if mask is not None:
            # head axis broadcast
            mask = mask.unsqueeze(1)

        # calculate the attention weights and apply dropout
        output, attn = self.attention(q, k, v, mask=mask)
        # transpose back to (bsz, seq_len_q, n_heads * d)
        output = output.transpose(1, 2).contiguous().view(bsz, len_q, -1)
        output = self.dropout(self.fc(output))

        output = self.layernorm(output + residual)
        
        return output, attn
