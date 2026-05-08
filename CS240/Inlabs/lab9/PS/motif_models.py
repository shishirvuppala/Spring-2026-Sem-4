import torch
import torch.nn as nn
import torch.nn.functional as F

from signal_ops import prepare_input_batch


# ============================================================
# Parts B and C of the story:
# Hamza now builds a baseline motif detector and then improves
# it using ideas from the CNN outlab: BatchNorm and global
# average pooling.
# ============================================================


class ShallowCNN1D(nn.Module):
    """
    Baseline motif detector.

    Architecture (must be exact)
    ----------------------------
    Conv1d(1, 8, kernel_size=7, stride=1, padding=3)
    ReLU
    MaxPool1d(kernel_size=2, stride=2)
    Conv1d(8, 16, kernel_size=5, stride=1, padding=2)
    ReLU
    MaxPool1d(kernel_size=2, stride=2)
    Flatten
    Linear(16 * 32, 32)
    ReLU
    Linear(32, 4)

    Notes
    -----
    - Input shape is (B, 1, 128)
    - Output is logits of shape (B, 4)
    - Do NOT apply softmax inside forward()
    """

    def __init__(self):
        super().__init__()
        # TODO: define the required layers
        raise NotImplementedError

    def forward(self, x):
        """
        Accept x in any format supported by prepare_input_batch().
        Return logits of shape (B, 4).
        """
        raise NotImplementedError


class BetterCNN1D(nn.Module):
    """
    Improved motif detector.

    Architecture (must be exact)
    ----------------------------
    Conv1d(1, 16, kernel_size=5, stride=1, padding=2)
    BatchNorm1d(16)
    ReLU
    MaxPool1d(kernel_size=2, stride=2)
    Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
    BatchNorm1d(32)
    ReLU
    AdaptiveAvgPool1d(1)
    squeeze last dimension only
    Linear(32, 4)

    Notes
    -----
    - Input shape is (B, 1, 128)
    - Output is logits of shape (B, 4)
    - Do NOT apply softmax inside forward()
    """

    def __init__(self):
        super().__init__()
        # TODO: define the required layers
        raise NotImplementedError

    def forward(self, x):
        """
        Accept x in any format supported by prepare_input_batch().
        Return logits of shape (B, 4).
        """
        raise NotImplementedError


# ------------------------------------------------------------
# Optional bonus
# ------------------------------------------------------------

class ResidualBlock1D(nn.Module):
    """
    Optional bonus.

    Required block
    --------------
    Conv1d(channels, channels, 3, padding=1)
    BatchNorm1d(channels)
    ReLU
    Conv1d(channels, channels, 3, padding=1)
    BatchNorm1d(channels)
    add skip connection
    ReLU

    Input and output shapes must be identical.
    """

    def __init__(self, channels):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class ResidualCNN1D(nn.Module):
    """
    Optional bonus research model.

    Architecture
    ------------
    Conv1d(1, 16, 5, padding=2)
    BatchNorm1d(16)
    ReLU
    MaxPool1d(2)
    ResidualBlock1D(16)
    ResidualBlock1D(16)
    Conv1d(16, 32, 3, padding=1)
    BatchNorm1d(32)
    ReLU
    AdaptiveAvgPool1d(1)
    Linear(32, 4)
    """

    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
