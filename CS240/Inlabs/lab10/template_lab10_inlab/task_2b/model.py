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
        # self.input_vocab = 
        # self.output_vocab = 

        raise NotImplementedError("This function is incomplete")

    def forward(self, x):
        """
        x: (B, T) -> Integer tokens
        returns: 
        logits of shape (B, output_vocab) [note: do not apply softmax, since that is handled by the loss function internally]
        """        
        raise NotImplementedError("This function is incomplete")
