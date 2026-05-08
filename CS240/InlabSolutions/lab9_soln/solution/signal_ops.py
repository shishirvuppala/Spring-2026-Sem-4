import numpy as np
import torch


def conv1d_output_length(L_in, kernel_size, stride, padding):
    return int((L_in + 2 * padding - kernel_size) // stride + 1)


def maxpool1d_output_length(L_in, kernel_size, stride):
    return int((L_in - kernel_size) // stride + 1)


def manual_convolution_1d(x, kernel, stride=1, padding=0):
    x = np.asarray(x)
    kernel = np.asarray(kernel)

    if x.ndim != 1 or kernel.ndim != 1:
        raise ValueError("x and kernel must both be 1D arrays")
    if stride <= 0 or padding < 0:
        raise ValueError("stride must be positive and padding must be non-negative")

    x_pad = np.pad(x, (padding, padding), mode="constant")
    out_len = conv1d_output_length(len(x), len(kernel), stride, padding)
    out = np.empty(out_len, dtype=np.result_type(x, kernel, np.float32))

    for t in range(out_len):
        start = t * stride
        out[t] = np.sum(x_pad[start:start + len(kernel)] * kernel)

    return out


def prepare_input_batch(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    x = x.to(dtype=torch.float32)

    if x.ndim == 1:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 2:
        x = x.unsqueeze(1)
    elif x.ndim == 3:
        if x.shape[1] != 1:
            raise ValueError("Expected shape (B, 1, L) for 3D inputs")
    else:
        raise ValueError("Input must have shape (L,), (B, L), or (B, 1, L)")

    return x


def predict(model, x, batch_size=128, device="cpu"):
    device = torch.device(device)
    x = prepare_input_batch(x)

    preds = []
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start:start + batch_size].to(device)
            logits = model(xb)
            preds.append(logits.argmax(dim=1).cpu())

    return torch.cat(preds, dim=0).to(torch.long)
