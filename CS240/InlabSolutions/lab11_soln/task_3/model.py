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
        hidden = self.activation(self.W(hidden_prev) + self.U_b(x))
        out = self.V_c(hidden)
        return hidden, out
    
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    TODO:
    1. Compute attention scores
    2. Scale by sqrt(d_k)
    3. Apply mask (if provided)
    4. Apply softmax
    5. Multiply with V
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)
    return output, attn

class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        # TODO: Return positional encoding matching sequence length
        return self.pe[:, :x.size(1)].to(x.device)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # TODO: Define linear layers for Q, K, V
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        # TODO: Output projection
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        TODO:
        1. Project x → Q, K, V
        2. Split into heads
        3. Apply attention
        4. Concatenate heads
        5. Final linear layer
        """
        B, T, D = x.shape

        Q = self.q_linear(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        mask = torch.tril(torch.ones(T, T)).to(x.device)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,T)

        out, attn = scaled_dot_product_attention(Q, K, V, mask)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out), attn


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        # TODO: Multi-head attention
        self.attn = MultiHeadAttention(d_model, num_heads)

        # TODO: LayerNorms
        self.norm_att = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_ff)

        # TODO: Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        """
        TODO:
        1. Attention + residual + norm
        2. FFN + residual + norm
        """
        attn_out, attn = self.attn(self.norm_att(x))
        x = x + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x, attn

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
            Shape: (Batch Size, Seq_Len) -> (B, T)

        Returns:
            - output (torch.Tensor): The contextualized output embeddings after 
            linear projection.
            Shape: (Batch Size, Seq_Len, output_vocab_size)
        """
        x = self.embedding(x)
        B, N, D = x.shape
        M = N // self.K  # Number of blocks
        device = x.device

        # Reshape into blocks: (B, M, K, D)
        blocks = x.view(B, M, self.K, D)
        
        all_oi = []
        summary_tokens = []

        # RNN computations
        for i in range(M):
            current_block = blocks[:, i, :, :] 
            h_t = torch.zeros(B, D, device=device) # Tensor
            block_outputs = []

            for t in range(self.K):
                h_t, oi_t = self.rnn_cell(current_block[:, t, :], h_t)
                block_outputs.append(oi_t)
            
            summary_tokens.append(h_t)
            all_oi.append(torch.stack(block_outputs, dim=1))

        S = torch.stack(summary_tokens, dim=1)
        O_local = torch.cat(all_oi, dim=1)

        # Transformer computations
        Z = S + self.pe(S)
        Z, _ = self.transformer(Z)

        # Feature Expansion and Context Injection
        Z_expanded = torch.repeat_interleave(Z, repeats=self.K, dim=1)

        # Merge local RNN features with global Transformer context
        final_output = O_local + Z_expanded

        return self.fc(final_output)