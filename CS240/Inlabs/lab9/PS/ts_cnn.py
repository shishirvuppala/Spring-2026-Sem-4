"""
Autograder compatibility layer.

Students are expected to work mostly in:
    - signal_ops.py
    - motif_models.py
    - motif_training.py

The autograder imports from ts_cnn.py, so this file simply re-exports
the required functions and classes.
"""

from signal_ops import (
    conv1d_output_length,
    maxpool1d_output_length,
    manual_convolution_1d,
    prepare_input_batch,
    predict,
)

from motif_models import (
    ShallowCNN1D,
    BetterCNN1D,
    ResidualBlock1D,
    ResidualCNN1D,
)

from motif_training import (
    count_parameters,
    _accuracy_from_logits,
    train_better_model,
)

__all__ = [
    "conv1d_output_length",
    "maxpool1d_output_length",
    "manual_convolution_1d",
    "prepare_input_batch",
    "predict",
    "ShallowCNN1D",
    "BetterCNN1D",
    "ResidualBlock1D",
    "ResidualCNN1D",
    "count_parameters",
    "_accuracy_from_logits",
    "train_better_model",
]
