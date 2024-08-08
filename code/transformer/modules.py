import torch.nn as nn
import torch.nn.functional as F
import torch

from transformer.layers import EncodeLayer, DecoderLayer

class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(Encoder, self).__init__()
        self.encoder_layer = encoder_layer
        self.num_layers = num_layers
    def forward(self, src, mask, return_attns=False):
        enc_output = src
        if return_attns:
            attn_list = []
        for i in range(self.num_layers):
            enc_output, attn = self.encoder_layer(src=enc_output, attn_mask=mask)
