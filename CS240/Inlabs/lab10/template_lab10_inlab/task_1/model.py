import torch
import torch.nn as nn
import math

class RNNCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
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
        hidden = hidden_prev + x
        out = self.V_c(hidden)
        return hidden, out


class RNNModel(nn.Module):
    def __init__(self, d_model=64, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.input_vocab = 9
        self.output_vocab = 225 # here you can assume the output vocab size to 225 (max possible sum + padding)

        self.embedding = nn.Embedding(self.input_vocab, d_model)

        self.layers = nn.ModuleList([
            RNNCell(d_model)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(self.output_vocab, d_model)

    def forward(self, x):
        """
        x: (B, T) -> Integer tokens
        returns: 
        logits of shape (B, output_vocab) [note: do not apply softmax, since that is handled by the loss function internally]
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
                x_t = x[:, :, t] # Current input token: (B, d_model)
                h_t, o_t = layer(x_t, h_t)
                out.append(o_t)

            out = torch.stack(out, dim=1)

        # 5. Project to output vocab: (B, T, 451)
        out = self.fc(x).detach()
        
        return out