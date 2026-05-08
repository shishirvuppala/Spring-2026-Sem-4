import torch
import torch.nn as nn
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    TODO:
    1. Compute attention scores
    2. Scale by sqrt(d_k)
    3. Apply mask (if provided)
    4. Apply softmax
    5. Multiply with V
    """
    raise NotImplementedError('Function is not implemented')

class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # TODO: Create positional embeddings that you can then use in `forward()`
        raise NotImplementedError('Function is not implemented')


    def forward(self, x):
        # TODO: Return positional encoding matching sequence length
        raise NotImplementedError('Function is not implemented')


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # TODO: Define linear layers for Q, K, V

        # TODO: Output projection

        raise NotImplementedError('Function is not implemented')

    def forward(self, x):
        """
        TODO:
        1. Project x → Q, K, V
        2. Split into heads
        3. Apply attention
        4. Concatenate heads
        5. Final linear layer
        """
        raise NotImplementedError('Function is not implemented')


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        # TODO: Define Multi-head attention module

        # TODO: Define LayerNorms

        # TODO: Define Feedforward network

        raise NotImplementedError('Function is not implemented')

    def forward(self, x):
        """
        TODO:
        1. Attention + residual + norm
        2. FFN + residual + norm
        """
        raise NotImplementedError('Function is not implemented')


class TransformerModel(nn.Module):
    def __init__(self, d_model=64, num_heads=2, num_layers=3, max_len=256, pe_type='sin'):
        super().__init__()

        # TODO: Create Token embedding
        # assuming max tokens will be 10 (i.e. digits 0 to 9)

        # TODO: Positional encoding selection

        # TODO: Stack encoder blocks

        # TODO: Output projection (Map to Prefix-Sum vocab)
        # Assuming output vocab size is 451 (max possible sum + padding)


    def forward(self, x, return_attn=False):
        """
        TODO:
        1. Embed tokens
        2. Add positional encoding
        3. Pass through encoder layers
        4. Project to vocab

        NOTE: If `return_attn` is true, you must return the `output` and the `attention scores`, if not return only the `output`.
        """
        raise NotImplementedError('Function is not implemented')
        
