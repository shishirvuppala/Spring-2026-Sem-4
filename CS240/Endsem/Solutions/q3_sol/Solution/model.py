"""
model.py  —  Do not modify it.
"""

import torch
import torch.nn as nn

class SmallCNN(nn.Module):
    """
    A small CNN for 10-class image classification.

    Input : (B, 1, 28, 28)   — greyscale images
    Output: (B, 10)           — raw logits (pre-softmax)

    Architecture
    ------------
    Block 1:  Conv(1  -> 16, 3x3, pad=1) -> BN -> ReLU -> MaxPool(2)
    Block 2:  Conv(16 -> 32, 3x3, pad=1) -> BN -> ReLU -> MaxPool(2)
    Block 3:  Conv(32 -> 64, 3x3, pad=1) -> BN -> ReLU            <- target layer for Grad-CAM
    Classifier: AdaptiveAvgPool -> Flatten -> Linear(64, 10)

    Note: Block 3 has no MaxPool, so the spatial size after Block 3 is 7x7.
    """

    def __init__(self, block3_channels: int = 64):
        """
        Parameters
        ----------
        block3_channels : int
            Number of output channels in Block 3 (the prunable layer).
            Default is 64. After pruning, pass the reduced count here.
        """
        super().__init__()

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 28x28 -> 14x14
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 14x14 -> 7x7
        )

        # Block 3  — this is the layer students will prune
        self.block3 = nn.Sequential(
            nn.Conv2d(32, block3_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(block3_channels),
            nn.ReLU(inplace=True),
            # No MaxPool here: spatial size stays at 7x7
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, C, 7, 7) -> (B, C, 1, 1)
            nn.Flatten(),             # (B, C)
            nn.Linear(block3_channels, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x
