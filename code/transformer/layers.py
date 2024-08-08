import torch.nn as nn
import torch.nn.functional as F
import torch
from transformer.attns import MultiHeadAttention
import numpy as np


class PositionEncoder(nn.Module):
    """ The Position Encoder in Transformer
    - 1. Embedding Layer
    - 2. Positional Encoding
    """
    def __init__(self, d_model=512, max_seq_len=5000) -> None:
        super().__init__()
        # Embedding Layer
        self.emb = nn.Embedding(max_seq_len, d_model)
        


class PositionwiseFeedForward(nn.Module):
    """ The Position Wise Feed Forward Layer in Transformer
    - 1. Linear Layer
    - 2. Dropout
    - 3. ReLU
    """
    def __init__(self, d_model, d_ff, drop_out=0.1):
        super().__init__()
        pass


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