import torch
import torch.nn as nn
import math

class RNNCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # TODO: Create learnable weights W, U, V, b, c
        raise NotImplementedError('Function is not implemented')

    def forward(self, x, hidden_prev):
        """
        Processes a single time step.
        x: (B, input_dim)
        hidden_prev: (B, hidden_dim)
        out: (B, hidden_dim)
        """
        raise NotImplementedError('Function is not implemented')



class RNNModel(nn.Module):
    def __init__(self, d_model=128, num_layers=1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        # TODO: Create Token embedding 
        # assuming max tokens will be 10 (i.e. digits 0 to 9)

        # TODO: Stack Manual RNN Cells

        # TODO: Output projection (Map to Prefix-Sum vocab)
        # Assuming output vocab size is 451 (max possible sum + padding)

    def forward(self, x):
        """
        x: (B, T) -> Integer tokens

        Return:
        
        out: (B, T, 451)
        """
        raise NotImplementedError('Function is not implemented')