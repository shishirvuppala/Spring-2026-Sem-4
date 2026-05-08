import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from signal_ops import prepare_input_batch
from motif_models import BetterCNN1D


# ============================================================
# Part D of the story:
# Hamza trains the improved detector on archived labeled traces
# before deployment.
#
# To keep this a good 2-hour closed lab, the small bookkeeping
# helpers are already provided. You only need to fill the key
# training lines inside train_better_model().
# ============================================================


def count_parameters(model):
    """
    Return the number of trainable parameters of the model
    as a Python int.
    """
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _accuracy_from_logits(logits, y):
    """
    Private helper.

    Parameters
    ----------
    logits : torch.Tensor of shape (B, 4)
    y : torch.Tensor of shape (B,)

    Returns
    -------
    float
        Scalar accuracy as a Python float.
    """
    preds = logits.argmax(dim=1)
    return float((preds == y).float().mean().item())


def train_better_model(
    train_x,
    train_y,
    val_x,
    val_y,
    epochs=8,
    batch_size=64,
    lr=1e-3,
    device="cpu",
):
    """
    Train BetterCNN1D and return:
        (model, history)

    where history is a dict with keys:
        "train_loss", "train_acc", "val_acc"

    Story context
    -------------
    Hamza wants to know whether the improved detector is accurate enough
    on validation traces before it can be used.

    Required behavior
    -----------------
    - Construct BetterCNN1D inside this function
    - Use nn.CrossEntropyLoss()
    - Use Adam optimizer with learning rate lr
    - Shuffle the training data
    - Use logits directly with CrossEntropyLoss
    - Run validation in eval mode with torch.no_grad()
    - Store average train loss, train acc, and val acc per epoch

    Input data may be NumPy arrays or torch tensors.

    What not to do
    --------------
    - Do not apply softmax before CrossEntropyLoss
    - Do not use validation data for gradient updates
    - Do not print inside this function
    """
    device = torch.device(device)

    train_x = prepare_input_batch(train_x)
    val_x = prepare_input_batch(val_x)

    if isinstance(train_y, np.ndarray):
        train_y = torch.from_numpy(train_y)
    elif not isinstance(train_y, torch.Tensor):
        train_y = torch.tensor(train_y)

    if isinstance(val_y, np.ndarray):
        val_y = torch.from_numpy(val_y)
    elif not isinstance(val_y, torch.Tensor):
        val_y = torch.tensor(val_y)

    train_y = train_y.to(torch.long)
    val_y = val_y.to(torch.long)

    train_ds = TensorDataset(train_x, train_y)
    val_ds = TensorDataset(val_x, val_y)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # TODO:
    # 1) Create the model
    # 2) Create the loss function
    # 3) Create the optimizer
    #
    # Expected choices:
    # - BetterCNN1D
    # - nn.CrossEntropyLoss
    # - torch.optim.Adam
    raise NotImplementedError

    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    for _ in range(epochs):
        # TODO:
        # Put the model in training mode.
        # Then iterate over train_loader and perform one full training epoch.
        #
        # During training, compute:
        # - average training loss over all examples
        # - training accuracy over all examples
        #
        # Store them in variables:
        #   train_loss
        #   train_acc
        raise NotImplementedError

        # TODO:
        # Put the model in evaluation mode.
        # Run over val_loader without gradients.
        # Compute validation accuracy over all examples.
        #
        # Store it in:
        #   val_acc
        raise NotImplementedError

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))

    return model, history
