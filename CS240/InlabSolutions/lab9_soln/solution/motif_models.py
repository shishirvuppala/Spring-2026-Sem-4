import torch
import torch.nn as nn
import torch.nn.functional as F

from signal_ops import prepare_input_batch


class ShallowCNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=7, stride=1, padding=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 32, 32)
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        x = prepare_input_batch(x)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BetterCNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 4)

    def forward(self, x):
        x = prepare_input_batch(x)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x


class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = F.relu(out)
        return out


class ResidualCNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem_conv = nn.Conv1d(1, 16, 5, padding=2)
        self.stem_bn = nn.BatchNorm1d(16)
        self.pool = nn.MaxPool1d(2)
        self.block1 = ResidualBlock1D(16)
        self.block2 = ResidualBlock1D(16)
        self.conv = nn.Conv1d(16, 32, 3, padding=1)
        self.bn = nn.BatchNorm1d(32)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 4)

    def forward(self, x):
        x = prepare_input_batch(x)
        x = F.relu(self.stem_bn(self.stem_conv(x)))
        x = self.pool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = F.relu(self.bn(self.conv(x)))
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x
