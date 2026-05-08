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
        B, T, D = x.shape
        
        Q = self.q_linear(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # Compute raw attention energy
        logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Get the discrete selection (Forward Pass)
        max_indices = logits.argmax(dim=-1)
        attn_hard = torch.zeros_like(logits).scatter_(-1, max_indices.unsqueeze(-1), 1.0)

        # Get the soft distribution (for the Backward Pass gradient)
        attn_soft = torch.softmax(logits, dim=-1)

        # directly using argmax breaks the cuda graph, so you do something called a Straight Through Estimator (STE). Read more about it here: https://arxiv.org/abs/1611.01144
        # NOTE from TA: Even if you have used normal argmax/scatter or topk, you will get full marks. 
        attn_weights = attn_hard - attn_soft.detach() + attn_soft

        context = torch.matmul(attn_weights, V)

        context = context.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(context), attn_weights
    
    # argmax/scatter implementation
    # def forward(self, x):
    #     B, T, D = x.shape
        
    #     Q = self.q_linear(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
    #     K = self.k_linear(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
    #     V = self.v_linear(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    #     logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

    #     # One-Hot Selection (Argmax + Scatter)
    #     # Find the index of the highest energy score for each query
    #     max_indices = logits.argmax(dim=-1) # Shape: (B, H, T)

    #     # Create one-hot tensor
    #     attn_weights = torch.zeros_like(logits)
    #     attn_weights.scatter_(-1, max_indices.unsqueeze(-1), 1.0)

    #     # This matmul now acts as a pure index-based lookup
    #     context = torch.matmul(attn_weights, V)

    #     context = context.transpose(1, 2).contiguous().view(B, T, D)
    #     return self.out_proj(context), attn_weights