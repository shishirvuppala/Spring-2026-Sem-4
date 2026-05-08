import torch
from torch.utils.data import Dataset

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, vocab_size=10, seq_len=25, size=40000, K=5):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size

        self.X = torch.randint(0, vocab_size, (size, seq_len))
        self.Y = 2*self.X + K

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]