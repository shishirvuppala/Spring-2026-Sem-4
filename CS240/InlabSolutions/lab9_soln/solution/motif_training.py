import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from signal_ops import prepare_input_batch
from motif_models import BetterCNN1D


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _accuracy_from_logits(logits, y):
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

    model = BetterCNN1D().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            batch_n = yb.size(0)
            total_loss += float(loss.item()) * batch_n
            total_correct += int((logits.argmax(dim=1) == yb).sum().item())
            total_seen += batch_n

        train_loss = total_loss / total_seen
        train_acc = total_correct / total_seen

        model.eval()
        val_correct = 0
        val_seen = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                val_correct += int((logits.argmax(dim=1) == yb).sum().item())
                val_seen += yb.size(0)

        val_acc = val_correct / val_seen

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))

    return model, history
