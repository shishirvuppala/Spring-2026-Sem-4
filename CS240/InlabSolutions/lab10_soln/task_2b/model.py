import torch
import torch.nn as nn

# NOTE: Feel free to create more classes and functions if you like. 

class RNNCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)
        self.U_b = nn.Linear(dim, dim, bias=True)
        self.activation = nn.Tanh()

    def forward(self, x, hidden_prev):
        hidden = self.activation(self.W(hidden_prev) + self.U_b(x))
        return hidden

class RNNModel(nn.Module):
    def __init__(self, K, d_model=128):
        super().__init__()
        self.d_model = d_model
        self.input_vocab = 10
        self.output_vocab = 10*(2*K+1)
        
        self.embedding = nn.Embedding(self.input_vocab, d_model)

        # In problem 2, since there is dependency on both past and future timesteps, this cannot be achieved with a unidirectional RNN. You need to introduce bi-directionality.
        self.fwd_cell = RNNCell(d_model)
        self.bwd_cell = RNNCell(d_model)

        self.fc = nn.Linear(2 * d_model, self.output_vocab)

    def forward(self, x):
        """
        x: (B, T) -> Integer tokens
        returns: 
        logits of shape (B, output_vocab) [note: do not apply softmax, since that is handled by the loss function internally]
        """       
        B, T = x.shape
        x_emb = self.embedding(x) # (B, T, d_model)

        # --- Forward Pass ---
        h_fwd = torch.zeros(B, self.d_model).to(x.device)
        fwd_list = []
        for t in range(T):
            h_fwd = self.fwd_cell(x_emb[:, t, :], h_fwd)
            fwd_list.append(h_fwd)
        fwd_out = torch.stack(fwd_list, dim=1) # (B, T, d_model)

        # --- Backward Pass ---
        h_bwd = torch.zeros(B, self.d_model).to(x.device)
        bwd_list = [None] * T
        for t in reversed(range(T)):
            h_bwd = self.bwd_cell(x_emb[:, t, :], h_bwd)
            bwd_list[t] = h_bwd
        bwd_out = torch.stack(bwd_list, dim=1) # (B, T, d_model)

        # --- Concatenation ---
        # At each timestep i, combined has [fwd_i ; bwd_i]
        combined = torch.cat([fwd_out, bwd_out], dim=-1) # (B, T, 2*d_model)

        # --- Final Projection ---
        out = self.fc(combined)
        
        return out