import torch
import torch.nn as nn
import math

from models.transformer import SinusoidalPE


# this is the main class you needed to implement for the ConvTransformer model
class ConvAttention(nn.Module):
    def __init__(self, d_model, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        self.radius = kernel_size // 2

        self.weight = nn.Parameter(torch.randn(d_model, kernel_size))

    def forward(self, x):
        B, T, D = x.shape
        device = x.device

        pos = torch.arange(T, device=device)
        rel = pos[None, :] - pos[:, None]

        mask = (rel.abs() <= self.radius)
        rel_shifted = rel + self.radius

        scores = torch.full((B, D, T, T), -1e9, device=device)

        for k in range(self.kernel_size):
            scores = torch.where(
                (rel_shifted == k).unsqueeze(0).unsqueeze(0),
                self.weight[:, k].view(1, D, 1, 1),
                scores
            )

        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), -1e9)

        attn = torch.softmax(scores, dim=-1)

        x_perm = x.permute(0, 2, 1)  # (B, D, T)
        out = torch.matmul(attn, x_perm.unsqueeze(-1)).squeeze(-1)

        return out.permute(0, 2, 1), attn


# almost same as TransformerEncoderBlock, but with ConvAttention instead of MHA
class ConvTransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, d_ff=128):
        super().__init__()

        # this is the only difference from the standard TransformerEncoderBlock
        self.attn = ConvAttention(d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        attn_out, attn = self.attn(self.norm1(x))
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, attn


# almost same as TransformerModel, but with ConvTransformerEncoderBlock instead of TransformerEncoderBlock
class ConvTransformerModel(nn.Module):
    def __init__(self, d_model=64, num_heads=2, num_layers=3, max_len=256, pe_type='sin'):
        super().__init__()

        vocab_size = 10      # input digits
        out_vocab = 451      # prefix sum outputs

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.pe = SinusoidalPE(d_model, max_len) if pe_type == 'sin' else None

        # Only difference: Conv blocks instead of MHA blocks
        self.layers = nn.ModuleList([
            ConvTransformerEncoderBlock(d_model)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(d_model, out_vocab)

    def forward(self, x, return_attn=False):
        x = self.embedding(x)

        if self.pe is not None:
            x = x + self.pe(x)

        attn_maps = []

        for layer in self.layers:
            x, attn = layer(x)
            attn_maps.append(attn)

        logits = self.fc(x)

        if return_attn:
            return logits, attn_maps
        return logits