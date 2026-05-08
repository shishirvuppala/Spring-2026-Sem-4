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
    store = {}

    def hook(module, inp, out):
        store["feat"] = out.detach()

    h = model.block3[0].register_forward_hook(hook)

    with torch.no_grad():
        model(input_tensor)

    h.remove()

    return store["feat"][0]   # (K, H, W)


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
    Parameters
    ----------
    model         : nn.Module in eval mode
    input_tensor  : torch.Tensor of shape (1, C, H, W)
    target_class  : int

    Returns
    -------
    activations : Tensor (K, H', W')
    gradients   : Tensor (K, H', W')
    """
    activations_store = {}
    gradients_store   = {}

    def fwd_hook(module, inp, out):
        activations_store["feat"] = out.detach()

    def bwd_hook(module, grad_in, grad_out):
        gradients_store["feat"] = grad_out[0].detach()

    # Register on the conv layer inside block3 (index 0 of the Sequential)
    target_layer = model.block3[0]
    h_fwd = target_layer.register_forward_hook(fwd_hook)
    h_bwd = target_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad()

    # Forward pass — no torch.no_grad() so gradients can flow
    logits = model(input_tensor)               # (1, 10)
    score  = logits[0, target_class]           # scalar pre-softmax logit
    score.backward()

    h_fwd.remove()
    h_bwd.remove()

    activations = activations_store["feat"][0]   # (K, H', W')
    gradients   = gradients_store["feat"][0]     # (K, H', W')

    return activations, gradients


def compute_alpha(model, input_tensor, target_class):
    """
    Compute alpha_k^c for every filter k
    Returns
    -------
    alpha : Tensor (K,)   — importance weight per filter
    """
    activations, gradients = get_gradcam_components(model, input_tensor, target_class)

    # Eq.(1): alpha_k = (1 / H*W) * sum_{i,j} d(y^c)/d(A^k_{ij})
    alpha = gradients.mean(dim=(-2, -1))     # (K,)
    return alpha


def compute_gradcam_heatmap(model, input_tensor, target_class):
    """
    Compute full Grad-CAM heatmap  L^c, upsampled to input resolution.
 
      - Compute the weighted sum of activations using alpha_k^c as weights, per Eq.(2).
      - Upsample bilinearly to (H_in, W_in) using F.interpolate
      - Normalise to [0, 1]
 
    Parameters
    ----------
    model         : SmallCNN
    input_tensor  : torch.Tensor  shape (1, 1, 28, 28)
    target_class  : int
 
    Returns
    -------
    heatmap : torch.Tensor  shape (H_in, W_in),  values in [0, 1]
    """
    activations, _ = get_gradcam_components(model, input_tensor, target_class)
    alpha = compute_alpha(model, input_tensor, target_class)  # (K,)

    # Eq.(2): L = ReLU( sum_k alpha_k * A^k )
    weighted = (alpha[:, None, None] * activations).sum(dim=0)   # (H', W')
    heatmap  = F.relu(weighted)

    # Upsample to input resolution
    H_in = input_tensor.shape[-2]
    W_in = input_tensor.shape[-1]
    heatmap = F.interpolate(
        heatmap.unsqueeze(0).unsqueeze(0),   # (1, 1, H', W')
        size=(H_in, W_in),
        mode="bilinear",
        align_corners=False,
    ).squeeze()                               # (H_in, W_in)

    # Normalise to [0, 1]
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max > h_min:
        heatmap = (heatmap - h_min) / (h_max - h_min)
    return heatmap
 

def compute_alpha_bar(model, sample_images):
    """
    alpha_bar_k = (1 / N_images * N_classes) * sum_{n,c} alpha_k^{c,(n)}

    Returns Tensor (K,)
    """

    model.eval()
    K         = model.block3[0].out_channels
    alpha_sum = torch.zeros(K)
    n_calls   = 0

    for n in range(sample_images.shape[0]):
        inp = sample_images[n].unsqueeze(0)      # (1, 1, 28, 28)
        for c in range(10):
            alpha_sum += compute_alpha(model, inp, c)
            n_calls   += 1

    return alpha_sum / n_calls

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
    model         : SmallCNN (original unpruned model)
 
    Returns
    -------
    keep_indices : list[int]  — filter indices to retain (sorted ascending)
    n_removed    : int        — number of filters removed
    """

    alpha_bar = torch.load("alpha_bar.pt", map_location=DEVICE)   # (K,)

    K = alpha_bar.shape[0]   # 64
    sorted_idx = alpha_bar.argsort().tolist()   # ascending importance

    pruned_model = None
    n_pruned     = 0

    for n_remove in range(1, K):
        keep_idx     = sorted_idx[n_remove:]
        candidate    = build_pruned_model(model, keep_idx)
        if count_params(candidate) <= P_MAX:
            pruned_model = candidate
            n_pruned     = n_remove
            break

    keep_idx = sorted_idx[n_pruned:]

    return keep_idx, n_pruned

def build_pruned_model(model, keep_indices):
    """
    Construct a new SmallCNN with block3 pruned to keep_indices.
    Correctly update: Conv weight, BN (weight/bias/running stats),
    and the subsequent Linear layer's input dimension.

    Parameters
    ----------
    model        : original SmallCNN
    keep_indices : list of int — indices of filters to KEEP
 
    Returns
    -------
    pruned : SmallCNN with block3_channels=len(keep_indices)
    """
    keep = sorted(keep_indices)
    n_keep = len(keep)

    pruned = SmallCNN(block3_channels=n_keep)

    # block1, block2 — copy unchanged
    pruned.block1.load_state_dict(model.block1.state_dict())
    pruned.block2.load_state_dict(model.block2.state_dict())

    # block3[0]  Conv2d weight: (64, 32, 3, 3) -> (n_keep, 32, 3, 3)
    pruned.block3[0].weight.data = model.block3[0].weight.data[keep].clone()

    # block3[1]  BatchNorm2d: gamma, beta, running_mean, running_var
    old_bn = model.block3[1]
    new_bn = pruned.block3[1]
    new_bn.weight.data       = old_bn.weight.data[keep].clone()
    new_bn.bias.data         = old_bn.bias.data[keep].clone()
    new_bn.running_mean.data = old_bn.running_mean.data[keep].clone()
    new_bn.running_var.data  = old_bn.running_var.data[keep].clone()

    # classifier[2]  Linear: weight (10, 64) -> (10, n_keep)
    pruned.classifier[2].weight.data = model.classifier[2].weight.data[:, keep].clone()
    pruned.classifier[2].bias.data   = model.classifier[2].bias.data.clone()

    return pruned


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