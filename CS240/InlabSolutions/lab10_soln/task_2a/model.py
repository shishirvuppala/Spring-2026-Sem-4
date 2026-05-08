import torch
import torch.nn as nn
import math

# NOTE: Feel free to create more classes and functions if you like. 

class RNNModel(nn.Module):
    def __init__(self, K, d_model=64, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        # TODO: Define the parameters and the vocabulary sizes as you like
        self.input_vocab = 10
        self.output_vocab = 20 + K # max possible output

        self.embedding = nn.Embedding(self.input_vocab, d_model)

        # In problem 1, since there is no dependency between time-steps, the recurrent weight matrix W becomes unnecessary. Eliminating W from RNN collapses the recurrance into a parallel transformation across all time steps which is mathematically equivalent to applying a linear layer.
        self.U_bs = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=True)
            for _ in range(num_layers)
        ])
        self.V_cs = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=True)
            for _ in range(num_layers)
        ])
        self.activation = nn.Tanh()

        self.fc = nn.Linear(d_model, self.output_vocab)

    def forward(self, x):
        """
        x: (B, T) -> Integer tokens
        returns: 
        logits of shape (B, output_vocab) [note: do not apply softmax, since that is handled by the loss function internally]
        """
        
        # 1. Embed tokens: (B, T) -> (B, T, d_model)
        x = self.embedding(x)

        # 2. Fast parallel operations (layer processing)
        for index in range(len(self.U_bs)):
            hidden = self.activation(self.U_bs[index](x))
            out = self.V_cs[index](hidden)
            x = x + out

        # 3. Project to output vocab: (B, T, output_vocab)
        out = self.fc(x)
        
        return out