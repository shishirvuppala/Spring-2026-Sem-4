import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from model import SmallCNN

DEVICE = torch.device("cpu")

P_MAX = 20000
A_MIN = 0.8

CLASS_NAMES = [
        "T-shirt/top", "Trouser",  "Pullover", "Dress",  "Coat",
        "Sandal",      "Shirt",    "Sneaker",  "Bag",     "Ankle boot",
    ]

# ─────────────────────────────────────────────────────────────────────────────
# Part 0
# Use a forward hook to capture and visualise the feature maps of block3[0]
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_maps(model, input_tensor):
    """
    Capture the output feature maps of block3[0] for a single input.
    No backward pass is needed — wrap the forward pass in torch.no_grad().

    Parameters
    ----------
    model        : SmallCNN in eval mode
    input_tensor : torch.Tensor  shape (1, 1, 28, 28)

    Returns
    -------
    feature_maps : torch.Tensor  shape (K, H, W)
    """
    store = {}

    def hook(module, inp, out):
        # YOUR CODE HERE
        pass

    # Register the hook, run the forward pass, then remove it
    # YOUR CODE HERE
    h = ...

    with torch.no_grad():
        # YOUR CODE HERE
        pass

    # YOUR CODE HERE
    h.remove()

    return ...   # (K, H, W)


def visualise_feature_maps(model, sample_images, class_idx=0):
    inp   = sample_images[class_idx].unsqueeze(0)
    fmaps = get_feature_maps(model, inp)
    K     = fmaps.shape[0]

    ncols = int(np.ceil(np.sqrt(K)))
    nrows = int(np.ceil(K / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.8, nrows * 1.8))
    fig.suptitle(f"Feature maps — block3[0]  |  input class: {CLASS_NAMES[class_idx]}",
                 fontsize=10)

    for idx, ax in enumerate(axes.flat):
        if idx < K:
            ax.imshow(fmaps[idx].numpy(), cmap="viridis")
            ax.set_title(f"k={idx}", fontsize=7)
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig("feature_maps.png", dpi=120, bbox_inches="tight")
    print(f"Saved feature_maps.png  ({K} feature maps, {nrows}x{ncols} grid)")
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PART A — GRAD-CAM
# ══════════════════════════════════════════════════════════════════════════════

def get_gradcam_components(model, input_tensor, target_class):
    """
    Compute the activations and gradients for Grad-CAM.
    Parameters
    ----------
    model         : SmallCNN in eval mode
    input_tensor  : torch.Tensor of shape (1, C, H, W)
    target_class  : int

    Returns
    -------
    activations : Tensor (K, H', W')
    gradients   : Tensor (K, H', W')
    """
    # Define any helper functions and variables you need here. 

    # --- TODO 1: define and register the forward hook ---
    # Hint: A forward hook receives (module, input, output)
    def forward_hook(module, inp, out):
        # YOUR CODE HERE
        pass

    # --- TODO 2: define and register the backward hook ---
    # Hint: A full backward hook receives (module, grad_input, grad_output).
    def backward_hook(module, grad_in, grad_out):
        # YOUR CODE HERE
        pass

    # ---- TODO 3: Register both hooks on CNN layer of last block of model
    # Hint: use  .register_forward_hook(...)  and  .register_full_backward_hook(...)
    # YOUR CODE HERE

    # --- TODO 4: forward pass + backward pass ---
    # Hint: do NOT use torch.no_grad() here — gradients must flow.
    # Hint: Call model.zero_grad() first to clear any accumulated gradients.
    # YOUR CODE HERE
 
    # --- TODO 5: remove both hooks ---
    # YOUR CODE HERE

    # --- TODO 6: extract activations and gradients from the stores, and return ---
    # YOUR CODE HERE
    activations = ...
    gradients   = ...

    return activations, gradients


def compute_alpha(model, input_tensor, target_class):
    """
    Compute alpha_k^c for every filter k
    Parameters
    ----------
    model         : SmallCNN in eval mode
    input_tensor  : torch.Tensor of shape (1, C, H, W)
    target_class  : int

    Returns
    -------
    alpha : Tensor (K,)   — importance weight per filter
    """
    activations, gradients = get_gradcam_components(model, input_tensor, target_class)

    alpha = ...
    return alpha


def compute_gradcam_heatmap(model, input_tensor, target_class):
    """
    Compute full Grad-CAM heatmap  L^c, upsampled to input resolution.
 
      - Compute the weighted sum of activations using alpha_k^c as weights, per Eq.(2).
      - Upsample bilinearly to (H_in, W_in) using F.interpolate
      - Normalise to [0, 1]
 
    Parameters
    ----------
    model         : SmallCNN in eval mode
    input_tensor  : torch.Tensor  shape (1, 1, 28, 28)
    target_class  : int
 
    Returns
    -------
    heatmap : torch.Tensor  shape (H_in, W_in),  values in [0, 1]
    """
    activations, gradients = get_gradcam_components(model, input_tensor, target_class)
    alpha = compute_alpha(model, input_tensor, target_class)  # (K,)
 
    # --- Use Eq.(2) to compute the weighted sum ---
    # YOUR CODE HERE
    cam = ...
 
    # --- Upsample to input image resolution ---
    # Use F.interpolate with mode="bilinear", align_corners=False
    # Input to interpolate must be 4-D: (1, 1, H', W').
    H_in = input_tensor.shape[-2]
    W_in = input_tensor.shape[-1]
    # YOUR CODE HERE
    cam_up = ...
 
    # --- Normalise to [0, 1] ---
    # Hint: subtract the min, divide by (max - min).
    # Guard against the edge case where max == min.
    # YOUR CODE HERE
    cam_up = ...

    return cam_up
 

def compute_alpha_bar(model, sample_images):
    """
    alpha_bar_k = (1 / N_images * N_classes) * sum_{n,c} alpha_k^{c,(n)}

    Parameters
    ----------
    model         : SmallCNN in eval mode
    sample_images : torch.Tensor  shape (N_images, 1, 28, 28)
    
    Returns Tensor (K,)
    """

    # Loop over all 10 images and all 10 classes.
    # For each (image, class) pair, call compute_alpha and accumulate.
    # YOUR CODE HERE

    return ...

# ══════════════════════════════════════════════════════════════════════════════
# PART B — PRUNING
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        preds    = model(x).argmax(1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
    return correct / total


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def select_filters(model):
    """
    Return the indices of the filters to KEEP such that the pruned model 
    has at most p_max parameters.

    Parameters
    ----------
    model         : SmallCNN (original unpruned model) in eval mode
 
    Returns
    -------
    keep_indices : list[int]  — filter indices to retain (sorted ascending)
    n_removed    : int        — number of filters removed
    """
    keep_indices = ...
    n_removed    = ...

    # WRITE YOUR CODE HERE

    return keep_indices, n_removed

def build_pruned_model(model, keep_indices):
    """
    Construct a new SmallCNN with block3 pruned to keep_indices.
    Correctly update: Conv weight, BN (weight/bias/running stats),
    and the subsequent Linear layer's input dimension.

    Parameters
    ----------
    model        : original SmallCNN in eval mode
    keep_indices : list of int — indices of filters to KEEP
 
    Returns
    -------
    pruned : SmallCNN with block3_channels=len(keep_indices)
    """
    keep = sorted(keep_indices)
    n_keep = len(keep)

    # Instantiate a new model with the reduced channel count
    pruned = SmallCNN(block3_channels=n_keep)

    # block1, block2 — copy unchanged
    pruned.block1.load_state_dict(model.block1.state_dict())
    pruned.block2.load_state_dict(model.block2.state_dict())

    # --- Prune block3[0] ---
    # YOUR CODE HERE
 
    # --- Prune block3[1]  (BatchNorm2d) ---
    # Four tensors to prune: weight (gamma), bias (beta), running_mean, running_var
    # YOUR CODE HERE
 
    # --- Prune classifier[2]  (Linear) ---
    # YOUR CODE HERE
 

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

# We will be running the same main function to evaluate your final submission, so do NOT change its structure or the files it reads/writes.
def main():    
    model = SmallCNN()
    model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
    model.eval()

    sample_images = torch.load("sample_images.pt", map_location=DEVICE)   # (10, 1, 28, 28)

    with open("test_loader.pkl", "rb") as f:
        test_loader = pickle.load(f)

    # Feel free to add code to print sample_images here

    # ── Part 0 ────────────────────────────────────────────────────────────────
    visualise_feature_maps(model, sample_images, class_idx=0)

    # ── Part 1 ────────────────────────────────────────────────────────────────
    alpha_bar = compute_alpha_bar(model, sample_images)
    torch.save(alpha_bar, "alpha_bar.pt")
    print(f"  Saved alpha_bar.pt  — shape: {alpha_bar.shape},  ", f"min={alpha_bar.min():.4f},  max={alpha_bar.max():.4f}")

    # ── Part 2 ────────────────────────────────────────────────────────────────
    keep_idx, n_removed = select_filters(model)
    print(f"  Selected {len(keep_idx)} filters to keep, removed {n_removed} filters.")

    pruned_model = build_pruned_model(model, keep_idx)
    pruned_model.eval()
 
    try:
        pruned_model(torch.randn(1, 1, 28, 28))
        print("    Forward pass OK")
    except Exception as e:
        print(f"    Forward pass FAILED — {e}")
 
    torch.save(pruned_model.state_dict(), "pruned_model.pth")
    print("    Saved pruned_model.pth")
 
    # ── Evaluation and constraint check ─────────────────────────────────
    original_accuracy = evaluate(model,        test_loader)
    pruned_accuracy   = evaluate(pruned_model, test_loader)
    original_params   = count_params(model)
    pruned_params     = count_params(pruned_model)
 
    print("Pruning Results")
    print(f"    Original — accuracy: {original_accuracy:.4f}   params: {original_params:,}")
    print(f"    Pruned   — accuracy: {pruned_accuracy:.4f}   params: {pruned_params:,}")
    print(f"    Param reduction: {(original_params - pruned_params) / original_params * 100:.1f}%")
    print()
 
    # ── Constraint check ──────────────────────────────────────────────────────
 
    p_ok = pruned_params   <= P_MAX
    a_ok = pruned_accuracy >= A_MIN
 
    print("── Constraint check ─────────────────────────────────────────────────")
    print(f"  Params    {pruned_params:,}  <=  {P_MAX:,}  :  {'PASS' if p_ok else 'FAIL'}")
    print(f"  Accuracy  {pruned_accuracy:.4f}  >=  {A_MIN:.2f}       :  {'PASS' if a_ok else 'FAIL'}")
    if p_ok and a_ok:
        print("  Both constraints satisfied.")
    elif p_ok:
        print("  Parameter constraint met, but accuracy too low.")
        print("  Try removing fewer filters, or switch to a different criterion.")
    elif a_ok:
        print("  Accuracy constraint met, but too many parameters remain.")
        print("  Try removing more filters.")
    else:
        print("  Neither constraint met.")
        print("  Revisit your importance criterion and the number of filters removed.")
    print()
 
if __name__ == "__main__":
    main()