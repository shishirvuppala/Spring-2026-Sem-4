import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# NOTE: You can assume the input is already updated with positional embeddings.
class OneHotMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        DO NOT MODIFY THE INIT FUNCTION
        """
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        Args:
        x (torch.Tensor): The input sequence embeddings. 
            Shape: (Batch Size, Seq_Len, d_model) -> (B, T, D)

        Returns:
            tuple:
                - output (torch.Tensor): The contextualized output embeddings after 
                linear projection.
                Shape: (Batch Size, Seq_Len, d_model) -> (B, T, D)
                - attn_weights (torch.Tensor): The normalized attention scores 
                representing token-to-token relationships across all heads.
                Shape: (Batch Size, num_heads, Seq_Len, Seq_Len) -> (B, H, T, T)
        """
        raise NotImplementedError('Function is not implemented')