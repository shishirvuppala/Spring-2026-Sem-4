import torch
import torch.nn as nn
import math


class FastRNNModel(nn.Module):
    def __init__(self, d_model=128, num_layers=1, vocab_size=10):
        super().__init__()

        # TODO: Implement the Faster RNN 
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # RNN parameters (same as your numpy version)
        self.U = nn.Parameter(torch.randn(d_model, d_model))
        self.W = nn.Parameter(torch.randn(d_model, d_model))
        self.V = nn.Parameter(torch.randn(vocab_size, d_model))

        self.b = nn.Parameter(torch.zeros(d_model))
        self.c = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x):
        # Embedding
        x_seq = self.embedding(x)  # (T, D)
        
        T, D = x_seq.shape
        H = self.d_model
        
        h = torch.zeros(T, H, device=x.device)
        
        # Compute K_k = A^k B
        K = []
        A_power = torch.eye(H, device=x.device)
        
        for k in range(T):
            K.append(A_power @ self.U)
            A_power = A_power @ self.W
        
        # Stack inputs
        X = x_seq  # (T, D)
        
        # Vectorized hidden computation
        for k in range(T):
            h[k:] += (K[k] @ X[:T-k].T).T
        
        # Bias accumulation
        A_power = torch.eye(H, device=x.device)
        for k in range(T):
            h[k:] += (A_power @ self.b)
            A_power = A_power @ self.W
        
        # Output
        y = torch.stack([self.V @ h[t] + self.c for t in range(T)])
        
        return y
