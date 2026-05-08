import numpy as np
import torch


# ============================================================
# Part A of the story:
# Hamza first studies small toy traces and makes sure the
# low-level signal operations are correct before building the
# neural motif detector.
#
# To keep this 2-hour lab focused, a few tiny formula helpers
# are already provided. You mainly need to implement:
#   1) manual_convolution_1d
#   2) prepare_input_batch
#
# The prediction helper is also provided so that you can test
# your models more easily once they are built.
# ============================================================


def conv1d_output_length(L_in, kernel_size, stride, padding):
    """
    Compute the output length of a 1D convolution layer with dilation = 1.
    """
    return int((L_in + 2 * padding - kernel_size) // stride + 1)


def maxpool1d_output_length(L_in, kernel_size, stride):
    """
    Compute the output length of a 1D max-pooling layer with:
        padding = 0
        ceil_mode = False
    """
    return int((L_in - kernel_size) // stride + 1)


def manual_convolution_1d(x, kernel, stride=1, padding=0):
    """
    Implement the 1D convolution used by deep learning libraries,
    i.e. cross-correlation without reversing the kernel.

    Story context
    -------------
    Hamza wants to test whether a short filter correctly responds to
    local patterns hidden inside a noisy 1D trace.

    Parameters
    ----------
    x : np.ndarray of shape (L,)
        One 1D signal.
    kernel : np.ndarray of shape (K,)
        One 1D kernel.
    stride : int
    padding : int

    Returns
    -------
    np.ndarray of shape (L_out,)

    Requirements
    ------------
    - Use NumPy only.
    - Use symmetric zero padding on both sides.
    - Do NOT reverse the kernel.
    - Do NOT mutate the input arrays.

    What not to do
    --------------
    - Do not use torch.nn.Conv1d here.
    - Do not use scipy.
    """
    raise NotImplementedError


def prepare_input_batch(x):
    """
    Convert incoming traces into a torch.float32 tensor of shape (B, 1, L).

    Accepted input shapes
    ---------------------
    - (L,)
    - (B, L)
    - (B, 1, L)

    Accepted input types
    --------------------
    - np.ndarray
    - torch.Tensor

    Required behavior
    -----------------
    - (L,)      -> (1, 1, L)
    - (B, L)    -> (B, 1, L)
    - (B, 1, L) -> unchanged shape
    - dtype must be torch.float32

    What not to do
    --------------
    - Do not move the tensor to GPU here.
    - Do not normalize here.
    - Do not accept arbitrary higher-dimensional inputs.
    """
    raise NotImplementedError


def predict(model, x, batch_size=128, device="cpu"):
    """
    Run batched prediction for the motif detector.

    This helper is provided so that you can more easily test the models
    from Parts B and C once they are implemented.

    Required behavior
    -----------------
    - Use model.eval()
    - Use torch.no_grad()
    - Accept x in any format supported by prepare_input_batch()
    - Run in mini-batches of size batch_size
    - Use argmax(dim=1) directly on logits
    - Return a 1D torch.long tensor of shape (B,)
    """
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
