import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE: DO NOT CREATE NEW CLASSES OR FUNCTIONS

class RNNCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)
        self.U_b = nn.Linear(dim, dim, bias=True)
        self.activation = nn.Tanh()

    def forward(self, x, hidden_prev):
        """
        Processes a single time step.
        x: (B, d_model)
        hidden_prev: (B, d_model)
        Returns: h_t (B, d_model)
        """
        # TODO: implement this function
        raise NotImplementedError("This function is incomplete")


class Encoder(nn.Module):
    def __init__(self, input_vocab, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(input_vocab, d_model)
        self.fwd_cell = RNNCell(d_model)

    def forward(self, x):
        """
        Transforms input sequence to hidden representations.
        x: (B, T)
        Returns: H (B, T, d_model)
        """
        # TODO: implement this function
        raise NotImplementedError("This function is incomplete")
        

class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.va = nn.Linear(d_model, 1, bias=False)

    def forward(self, query_ht, H):
        """
        Computes the context vector using Bahdanau-style positional attention.
        query_ht: (B, d_model) - The 'anchor' state from encoder at current index t
        H: (B, T, d_model) - All encoder states
        Returns: ct (B, d_model), alpha (B, T)
        """
        # TODO: implement this function
        raise NotImplementedError("This function is incomplete")

class Decoder(nn.Module):
    def __init__(self, output_vocab, d_model):
        super().__init__()
        self.d_model = d_model
        self.attention = Attention(d_model)
        
        self.Ws = nn.Linear(d_model, d_model, bias=False)
        self.Wc_b = nn.Linear(d_model, d_model, bias=True) # Bias b goes here
        
        self.V = nn.Linear(d_model, output_vocab, bias=False)
        self.M_c = nn.Linear(d_model, output_vocab, bias=True) # Bias c goes here

    def forward(self, H, return_attention=False):
        """
        Iteratively generates output logits for the entire sequence.
        H: (B, T, d_model) - Encoder memory bank
        Returns: logits (B, T, output_vocab), [Optional] attention (B, T, T) [note: do not apply softmax, since that is handled by the loss function internally]
        """
        # TODO: implement this function
        raise NotImplementedError("This function is incomplete")

class RNNAttentionModel(nn.Module):
    def __init__(self, K, d_model=128):
        super().__init__()
        self.d_model = d_model
        self.output_vocab = 10 * (2 * K + 1)
        
        self.encoder = Encoder(10, d_model)
        self.decoder = Decoder(self.output_vocab, d_model)

    def forward(self, x, return_attention=False):
        """
        Full Forward Pass: Encoder -> Decoder
        x: (B, T)
        """
        # TODO: implement this function
        raise NotImplementedError("This function is incomplete")