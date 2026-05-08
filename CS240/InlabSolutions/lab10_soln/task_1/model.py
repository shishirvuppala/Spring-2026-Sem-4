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
        hidden = self.activation(self.W(hidden_prev) + self.U_b(x)) # bug 1: weights were unused in rnn calculation
        out = self.V_c(hidden)
        return hidden, out


class RNNModel(nn.Module):
    def __init__(self, d_model=64, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.input_vocab = 10 # bug 2: input vocab had 1 less count. This will throw indexing error.
        self.output_vocab = 225

        self.embedding = nn.Embedding(self.input_vocab, d_model)

        self.layers = nn.ModuleList([
            RNNCell(d_model)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(d_model, self.output_vocab) # bug 3: the dimensions were reversed. This will throw error.

    def forward(self, x):
        """
        x: (B, T) -> Integer tokens
        returns: 
        logits of shape (B, output_vocab) [note: do not apply softmax, since that is handled by the loss function internally]
        """ 
        B, T = x.shape
        
        # 1. Embed tokens: (B, T) -> (B, T, d_model)
        x = self.embedding(x)

        # 2. Manual Recurrence Loop (Sequential processing)
        for layer in self.layers:
            out = []

            # 3. Initialize hidden state with zeros
            h_t = torch.zeros(B, self.d_model).to(x.device)
            
            for t in range(T):
                x_t = x[:, t, :] # bug 4: t was indexing the wrong axis
                h_t, o_t = layer(x_t, h_t)
                out.append(o_t)

            out = torch.stack(out, dim=1)
            x = x + out # bug 5: this line was deleted which means rnn was never being used (so no gradients).

        # 4. Project to output vocab: (B, T, 225)
        out = self.fc(x) # bug 6: using .detach() broke the computation graph. So loss was never getting backpropogated through the model.
        
        return out