import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, vocab_size=10, seq_len=25, size=10000, K=3):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size
        self.K = K

        self.X = torch.randint(0, vocab_size, (size, seq_len))

        x_floated = self.X.unsqueeze(1).float()
        kernel_size = 2 * K + 1
        kernel = torch.ones((1, 1, kernel_size))
        window_sum = F.conv1d(x_floated, kernel, padding='same')
        self.Y = window_sum.squeeze(1).long()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]