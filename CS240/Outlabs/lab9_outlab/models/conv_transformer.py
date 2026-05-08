import torch
import torch.nn as nn
import math


class ConvTransformerModel(nn.Module):
    def __init__(self, d_model=64, num_heads=2, num_layers=3, max_len=256, pe_type='sin'):
        super().__init__()
        # TODO: Implement Convolution using Transformers
        raise NotImplementedError("Function is not implemented")

    def forward(self, x, return_attn=False):
        raise NotImplementedError("Function is not implemented")