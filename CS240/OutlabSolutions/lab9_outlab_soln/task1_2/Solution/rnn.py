import torch
import torch.nn as nn
import math

class RNNCell(nn.Module):
    def __init__(self, dim):
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


class RNNModel(nn.Module):
    def __init__(self, d_model=128, num_layers=1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        # TODO: Token embedding
        self.embedding = nn.Embedding(10, d_model)

        # TODO: Stack Manual RNN Cells
        # For simplicity, this implementation shows a single-layer manual loop.
        # For multiple layers, you would stack these cells.
        self.layers = nn.ModuleList([
            RNNCell(d_model)
            for _ in range(num_layers)
        ])

        # TODO: Output projection (Map to Prefix-Sum vocab)
        # Assuming output vocab size is 451 (max possible sum + padding)
        self.fc = nn.Linear(d_model, 451)

    def forward(self, x):
        """
        x: (B, T) -> Integer tokens
        """
        B, T = x.shape
        
        # 1. Embed tokens: (B, T) -> (B, T, d_model)
        x = self.embedding(x)

        # 3. Manual Recurrence Loop (Sequential processing)
        for layer in self.layers:
            out = []

            # 2. Initialize hidden state with zeros
            h_t = torch.zeros(B, self.d_model).to(x.device)
            
            for t in range(T):
                x_t = x[:, t, :] # Current input token: (B, d_model)
                h_t, o_t = layer(x_t, h_t)
                out.append(o_t)

            out = torch.stack(out, dim=1)
            x = x + out

        # 5. Project to output vocab: (B, T, 451)
        out = self.fc(x)
        
        return out