import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RNNCell(nn.Module):
    def __init__(self, dim):
        """
        DO NOT MODIFY THE INIT FUNCTION
        """
        super().__init__()
        
        # h_t = tanh(W_hh * h_{t-1} + W_ih * x_t + bias)
        self.W = nn.Linear(dim, dim, bias=False)
        self.U_b = nn.Linear(dim, dim, bias=True)
        self.activation = nn.Tanh()
        self.V_c = nn.Linear(dim, dim, bias=True)

    def forward(self, x, hidden_prev):
        """
        Processes a single time step.
        x: (B, input_dim)
        hidden_prev: (B, hidden_dim)
        out: (B, hidden_dim)
        """
        raise NotImplementedError('Fill this function')
    
def scaled_dot_product_attention(Q, K, V, mask=None):
    # TODO: Use the code from Task 1 (DO NOT WRITE AGAIN)
    raise NotImplementedError('Fill this function')

class SinusoidalPE(nn.Module):
    # TODO: Use the code from Task 1 (DO NOT WRITE AGAIN)
    raise NotImplementedError('Fill this class')


class MultiHeadAttention(nn.Module):
    # TODO: Use the code from Task 1 (DO NOT WRITE AGAIN)
    raise NotImplementedError('Fill this class')


class TransformerEncoderBlock(nn.Module):
    # TODO: Use the code from Task 1 (DO NOT WRITE AGAIN)
    raise NotImplementedError('Fill this class')


class BlockRecurrentTransformerModel(nn.Module):
    def __init__(self, d_model = 64, num_heads = 2, max_len=256, K = 3):
        """
        DO NOT MODIFY THE INIT FUNCTION
        """
        super().__init__()
        self.d_model = d_model
        self.K = K  # Block size

        self.embedding = nn.Embedding(10, d_model)

        self.pe = SinusoidalPE(d_model, max_len)

        self.rnn_cell = RNNCell(d_model)
        
        self.transformer = TransformerEncoderBlock(d_model, num_heads, d_model)

        self.fc = nn.Linear(d_model, 217) 

    def forward(self, x):
        """
        Args:
        x (torch.Tensor): The input sequence embeddings. 
            Shape: (Batch Size, Seq_Len, d_model) -> (B, T, D)

        Returns:
            - output (torch.Tensor): The contextualized output embeddings after 
            linear projection.
            Shape: (Batch Size, Seq_Len, d_model) -> (B, T, D)
        """
        raise NotImplementedError('Function is not implemented')