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
        return self.activation(self.W(hidden_prev) + self.U_b(x))

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
        B, T = x.shape
        x_emb = self.embedding(x)

        # Forward Pass
        h_fwd = torch.zeros(B, self.d_model).to(x.device)
        fwd_states = []
        for t in range(T):
            h_fwd = self.fwd_cell(x_emb[:, t, :], h_fwd)
            fwd_states.append(h_fwd)
        
        return torch.stack(fwd_states, dim=1)

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
        B, T, _ = H.shape
        
        # Project Query and Keys
        q = self.Wq(query_ht).unsqueeze(1)
        k = self.Wk(H)                   
        
        # Energy score: e = v * tanh(q + k)
        energy = torch.tanh(q + k)        
        e = self.va(energy).squeeze(2)     
        
        alpha = F.softmax(e, dim=1)
        
        # Context vector: (B, 1, T) @ (B, T, 2*d) -> (B, 2*d)
        ct = torch.bmm(alpha.unsqueeze(1), H).squeeze(1)
        return ct, alpha

class Decoder(nn.Module):
    def __init__(self, output_vocab, d_model):
        super().__init__()
        self.d_model = d_model
        self.attention = Attention(d_model)
        
        # used for s_t = tanh(Ws * s_{t-1} + Wc * c_t + b)
        self.Ws = nn.Linear(d_model, d_model, bias=False)
        self.Wc_b = nn.Linear(d_model, d_model, bias=True)
        
        # used for P(y_t) = Softmax(V * s_t + M * c_t + c)
        self.V = nn.Linear(d_model, output_vocab, bias=False)
        self.M_c = nn.Linear(d_model, output_vocab, bias=True)

    def forward(self, H, return_attention=False):
        """
        Iteratively generates output logits for the entire sequence.
        H: (B, T, d_model) - Encoder memory bank
        Returns: logits (B, T, output_vocab), [Optional] attention (B, T, T) [note: do not apply softmax, since that is handled by the loss function internally]
        """
        B, T, _ = H.shape
        device = H.device
        
        # Initial decoder state s_0
        s_t = torch.zeros(B, self.d_model).to(device)
        
        all_logits = []
        all_attention = []

        for t in range(T):
            # Positional Anchor from Encoder
            h_t_enc = H[:, t, :]
            
            # Attention Mechanism
            ct, alpha = self.attention(h_t_enc, H)
            
            # state update s_t = tanh(Ws(s_{t-1}) + Wc(ct) + b)
            s_t = torch.tanh(self.Ws(s_t) + self.Wc_b(ct))
            
            # Output Projection V*s_t + M*c_t + c
            logits = self.V(s_t) + self.M_c(ct)
            
            all_logits.append(logits)
            if return_attention:
                all_attention.append(alpha)

        logits_stack = torch.stack(all_logits, dim=1)
        
        if return_attention:
            return logits_stack, torch.stack(all_attention, dim=1)
        return logits_stack

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

        # Encode the input sequence into memory H
        H = self.encoder(x)
        
        # Delegate the entire decoding process to the Decoder class
        return self.decoder(H, return_attention=return_attention)