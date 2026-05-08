import torch
import torch.nn as nn
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    1. Compute attention scores
    2. Scale by sqrt(d_k)
    3. Apply mask (if provided)
    4. Apply softmax
    5. Multiply with V
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, 1e9)

    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)
    return output, attn

class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) % 2
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return self.pe[:, :x.size(1)].to(x.device)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_k = d_model

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
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

        self.attn = MultiHeadAttention(d_model, num_heads)

        self.norm_att = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_ff)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        """
        1. Attention + residual + norm
        2. FFN + residual + norm
        """
        attn_out, attn = self.attn(self.norm_att(x))
        x = x + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x, attn


class TransformerModel(nn.Module):
    def __init__(self, d_model=128, num_heads=2, num_layers=2, max_len=256, pe_type='sin'):
        super().__init__()

        self.embedding = nn.Embedding(10, d_model)

        self.pe = SinusoidalPE(d_model, max_len) if pe_type == 'sin' else lambda x: 0

        # here, you can assume d_ff = 2*d_model
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_model*2)
            for _ in range(num_layers)
        ])

        # Assuming output vocab size is 226 (max possible sums)
        self.fc = nn.Linear(d_model, 226) 

    def forward(self, x, return_attn=False):
        """
        1. Embed tokens
        2. Add positional encoding
        3. Pass through encoder layers
        4. Project to vocab
        """
        x = self.embedding(x)

        if return_attn:
            attn_maps = []

        for layer in self.layers:
            x, attn = layer(x)
            if return_attn:
                attn_maps.append(attn)

        if return_attn:
            return self.fc(x), attn_maps
        else:
            return self.fc(x)
        