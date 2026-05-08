"""
Section 7 – Everything In One Frame
=====================================
classifiers.py

  7.4  train_classifier(...)              [TODO — graded]

Run:
    python classifiers.py

Submit:
    submission_<roll_number>/q1/classifiers.py
    Then run: check-cs240
    Then call a TA:  submit-cs240

─────────────────────────────────────────────────────────────────────────────
CONTEXT
─────────────────────────────────────────────────────────────────────────────
NASA's Kepler space telescope monitored ~150 000 stars and recorded periodic
dips in brightness caused by planets passing in front of them.  The hard
problem was separating real exoplanet transits from false positives: binary
star eclipses, instrumental artefacts, background blends.

You will train a small neural network to do this classification.

Each sample has six features measured from the light curve:

  transit_depth      – fractional drop in stellar brightness       (0–1)
  transit_duration   – duration of the transit in hours            (1–20 h)
  orbital_period     – estimated orbital period in days            (1–500 d)
  stellar_radius     – radius of the host star in solar radii      (0.1–3)
  equilibrium_temp   – estimated planet equilibrium temperature (K)(100–2500)
  signal_to_noise    – detection signal-to-noise ratio             (5–100)

Label:  0 = false positive,   1 = confirmed exoplanet candidate

─────────────────────────────────────────────────────────────────────────────
IMPORTANT DIFFERENCES FROM SECTIONS 4–5
─────────────────────────────────────────────────────────────────────────────
1.  Input has SIX features, not two.  The features have very different scales
    (e.g. equilibrium_temp can be ~2000 while transit_depth is < 0.01).
    You MUST standardise before training.

2.  Your model must return RAW LOGITS — not probabilities.
    Do NOT apply sigmoid inside forward().
    Use BCEWithLogitsLoss, which applies sigmoid internally for numerical
    stability.

3.  Because the model outputs logits, computing accuracy requires you to
    apply sigmoid yourself before thresholding at 0.5.

4.  Mini-batching is handled by make_batches() below — already written for
    you.  You do NOT need to implement a Dataset or DataLoader.
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import torch
import torch.nn as nn
from torch import optim

# ── reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)
rng = np.random.default_rng(42)

# ════════════════════════════════════════════════════════════════════════════
# DATA GENERATION — do not modify
# ════════════════════════════════════════════════════════════════════════════

def _generate_exoplanet_data(n_samples: int = 1000):
    """
    Synthesise a labelled exoplanet-transit dataset.

    False positives (label 0) and confirmed candidates (label 1) are drawn
    from different multivariate distributions that partially overlap, making
    the problem non-trivially learnable but not trivially separable.
    """
    n_pos = n_samples // 2
    n_neg = n_samples - n_pos

    # ── confirmed exoplanet transits ─────────────────────────────────────────
    depth_pos    = rng.beta(2, 5,   n_pos) * 0.08 + 0.002   # small, consistent
    duration_pos = rng.normal(4.0, 1.5,  n_pos).clip(1, 20)
    period_pos   = rng.lognormal(3.0, 1.2, n_pos).clip(1, 500)
    radius_pos   = rng.normal(1.0, 0.3,  n_pos).clip(0.1, 3)
    temp_pos     = rng.normal(1200, 400, n_pos).clip(100, 2500)
    snr_pos      = rng.normal(45, 15,   n_pos).clip(5, 100)

    # ── false positives ──────────────────────────────────────────────────────
    depth_neg    = rng.beta(5, 2,   n_neg) * 0.3 + 0.01     # deeper, noisier
    duration_neg = rng.normal(9.0, 3.5,  n_neg).clip(1, 20)
    period_neg   = rng.lognormal(2.0, 1.8, n_neg).clip(1, 500)
    radius_neg   = rng.normal(1.6, 0.6,  n_neg).clip(0.1, 3)
    temp_neg     = rng.normal(700,  350, n_neg).clip(100, 2500)
    snr_neg      = rng.normal(22, 12,   n_neg).clip(5, 100)

    X_pos = np.stack([depth_pos, duration_pos, period_pos,
                      radius_pos, temp_pos, snr_pos], axis=1)
    X_neg = np.stack([depth_neg, duration_neg, period_neg,
                      radius_neg, temp_neg, snr_neg], axis=1)

    X = np.vstack([X_pos, X_neg]).astype(np.float32)
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.float32)

    shuffle = rng.permutation(n_samples)
    return X[shuffle], y[shuffle]


X_all, y_all = _generate_exoplanet_data(n_samples=1000)

# ── train / validation split  (80 / 20, no shuffling — DataLoader shuffles) ─
split      = int(0.8 * len(X_all))
X_train_np = X_all[:split]
y_train_np = y_all[:split]
X_val_np   = X_all[split:]
y_val_np   = y_all[split:]

# feature names — for reference
FEATURE_NAMES = [
    "transit_depth", "transit_duration", "orbital_period",
    "stellar_radius", "equilibrium_temp", "signal_to_noise",
]

# ════════════════════════════════════════════════════════════════════════════
# MINI-BATCH HELPER — do not modify
# ════════════════════════════════════════════════════════════════════════════

def make_batches(X: torch.Tensor, y: torch.Tensor, batch_size: int):
    """
    Yield (X_batch, y_batch) tensors for one epoch, in random order.

    This replaces DataLoader for this exercise.
    Usage inside your training loop:

        for X_batch, y_batch in make_batches(X_train, y_train, batch_size):
            ...  # your five-line canonical loop goes here
    """
    n   = X.shape[0]
    idx = torch.randperm(n)
    for start in range(0, n, batch_size):
        batch_idx = idx[start : start + batch_size]
        yield X[batch_idx], y[batch_idx]


# ════════════════════════════════════════════════════════════════════════════
# 7.1 — ExoplanetClassifier                               [GIVEN]
# ════════════════════════════════════════════════════════════════════════════

class ExoplanetClassifier(nn.Module):
    """
    Two-hidden-layer binary classifier for exoplanet transit data.

    Architecture
    ────────────
        Linear(6, 32) → ReLU → Linear(32, 16) → ReLU → Linear(16, 1) → squeeze

    The forward pass must return RAW LOGITS of shape (B,).
    Do NOT apply sigmoid — BCEWithLogitsLoss does this internally.

    Parameters
    ----------
    (none — input_dim is always 6 for this dataset)

    Returns
    -------
    logits : torch.Tensor, shape (B,)
    """

    def __init__(self):
        super().__init__()
        # ── TODO 7.1a ────────────────────────────────────────────────────────
        # Define three linear layers named fc1, fc2, fc3.
        #   fc1 : 6  → 32
        #   fc2 : 32 → 16
        #   fc3 : 16 → 1
        # Hint: nn.Linear(in_features, out_features)
        # ─────────────────────────────────────────────────────────────────────
        self.fc1 = nn.Linear(6, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── TODO 7.1b ────────────────────────────────────────────────────────
        # Pass x through: fc1 → ReLU → fc2 → ReLU → fc3
        # Then squeeze the output from (B, 1) to (B,).
        # Return the result — these are logits, NOT probabilities.
        # Hint: torch.relu(...)  or  nn.functional.relu(...)
        # ─────────────────────────────────────────────────────────────────────
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x).squeeze(1)
        return logits


# ════════════════════════════════════════════════════════════════════════════
# 7.2 — standardize                                       [GIVEN]
# ════════════════════════════════════════════════════════════════════════════

def standardize(
    X_train: np.ndarray,
    X_val:   np.ndarray,
) -> tuple:
    """
    Standardise features to zero mean and unit variance.

    Compute the mean and standard deviation on X_train ONLY.
    Apply the same transformation to both X_train and X_val.
    Never fit statistics on the validation set — that would be data leakage.

    Parameters
    ----------
    X_train : np.ndarray, shape (N_train, 6)
    X_val   : np.ndarray, shape (N_val,   6)

    Returns
    -------
    X_train_std : np.ndarray, shape (N_train, 6),  mean≈0  std≈1 per column
    X_val_std   : np.ndarray, shape (N_val,   6),  shifted by train statistics
    """
    # ── TODO 7.2 ─────────────────────────────────────────────────────────────
    # 1. Compute mean and std across rows (axis=0) of X_train.
    # 2. Subtract mean, divide by std for both arrays.
    # 3. Guard against zero std: use (std + 1e-8) in the denominator.
    # Hint: numpy operations only — no PyTorch here.
    # ─────────────────────────────────────────────────────────────────────────
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train_std = (X_train - mean) / (std + 1e-8)
    X_val_std = (X_val - mean) / (std + 1e-8)
    return X_train_std, X_val_std


# ════════════════════════════════════════════════════════════════════════════
# 7.3 — compute_accuracy                                  [GIVEN]
# ════════════════════════════════════════════════════════════════════════════

def compute_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Compute binary classification accuracy from raw logits.

    Parameters
    ----------
    logits : torch.Tensor, shape (N,)  — raw model output (not probabilities)
    labels : torch.Tensor, shape (N,)  — ground truth, values in {0, 1}

    Returns
    -------
    accuracy : float in [0, 1]

    Note
    ────
    The model outputs logits, not probabilities.  You must apply sigmoid
    before thresholding.  A prediction is positive when sigmoid(logit) >= 0.5,
    which is equivalent to logit >= 0.
    """
    # ── TODO 7.3 ─────────────────────────────────────────────────────────────
    # 1. Apply torch.sigmoid to logits.
    # 2. Threshold at 0.5 to get predicted labels (a BoolTensor or 0/1 tensor).
    # 3. Compare to labels and compute the fraction that are correct.
    # Hint: .float().mean().item() converts a bool tensor to a Python float.
    # ─────────────────────────────────────────────────────────────────────────
    probs = torch.sigmoid(logits)
    predictions = probs >= 0.5
    accuracy = (predictions == labels.bool()).float().mean().item()
    return accuracy



# ════════════════════════════════════════════════════════════════════════════
# 7.4 — train_classifier                                  [TODO — graded]
# ════════════════════════════════════════════════════════════════════════════

def train_classifier(
    model,
    X_train:    np.ndarray,
    y_train:    np.ndarray,
    X_val:      np.ndarray,
    y_val:      np.ndarray,
    lr:         float = 1e-3,
    batch_size: int   = 32,
    num_epochs: int   = 150,
) -> tuple:
    """
    Train a binary classifier with mini-batch gradient descent.

    Steps to implement
    ──────────────────
    1. Call standardize(X_train, X_val) to get normalised arrays.
    2. Convert all four arrays to float32 torch tensors. instantiate new tensors using torch.tensor(..., dtype=torch.float32), do not modify the existing ones.
    3. Create the loss function: nn.BCEWithLogitsLoss()
       Create the optimiser: optim.Adam(model.parameters(), lr=lr)
       Ignore the LR scheduler, we spoke about in the previous section — it's not required here.
       learning rate is constant and is passed as an argument to this function.
    4. Epoch loop (num_epochs iterations):
         a. Inner batch loop using make_batches(X_tr, y_tr, batch_size):
              Run the canonical five-line training loop on each batch.
              (forward → loss → zero_grad → backward → step)
              Accumulate batch losses in a list.
         b. After the batch loop, record the MEAN batch loss for this epoch
            in train_losses.
         c. Compute validation accuracy with torch.no_grad() and
            compute_accuracy; append to val_accuracies.

    Parameters
    ----------
    model      : ExoplanetClassifier (or any nn.Module returning logits)
    X_train    : shape (N_train, 6), raw (un-standardised) numpy array
    y_train    : shape (N_train,),   labels in {0, 1}
    X_val      : shape (N_val,   6), raw numpy array
    y_val      : shape (N_val,),     labels in {0, 1}
    lr, batch_size, num_epochs : hyper-parameters

    Returns
    -------
    model          : trained nn.Module (mutated in place)
    train_losses   : list of length num_epochs, mean batch loss per epoch
    val_accuracies : list of length num_epochs, validation accuracy per epoch
    """
    # ── TODO 7.4 ─────────────────────────────────────────────────────────────
    raise NotImplementedError("TODO 7.4: implement train_classifier")
    return model, [], []

# ════════════════════════════════════════════════════════════════════════════
# PROVIDED SCAFFOLDING — do not modify
# ════════════════════════════════════════════════════════════════════════════

class SimpleDataset:
    """Thin wrapper kept for reference — make_batches is used instead."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ════════════════════════════════════════════════════════════════════════════
# AUTO-GRADER — do not modify anything below this line
# ════════════════════════════════════════════════════════════════════════════

def _test_exoplanet_classifier_shape():
    """7.1 — output shape is (B,) and output is logits (not bounded to (0,1))."""
    try:
        m = ExoplanetClassifier()
        x = torch.randn(16, 6)
        out = m(x)
        assert out.shape == (16,), (
            "7.1 FAIL – expected output shape (16,), "
            f"got {tuple(out.shape)}"
        )
        # logits should not be confined to (0, 1)
        assert not (out.min().item() >= 0 and out.max().item() <= 1), (
            "7.1 FAIL – output looks like probabilities (all in [0,1]). "
            "Did you accidentally apply sigmoid in forward()?"
        )
        print("  ✓  test_exoplanet_classifier_shape passed")
    except NotImplementedError as e:
        print(f"  (skipped – not implemented yet: {e})")


def _test_exoplanet_classifier_param_count():
    """7.1 — parameter count matches the stated architecture."""
    try:
        m = ExoplanetClassifier()
        expected = 6*32 + 32 + 32*16 + 16 + 16*1 + 1   # = 769
        got = sum(p.numel() for p in m.parameters())
        assert got == expected, (
            f"7.1 FAIL – parameter count: expected {expected}, got {got}. "
            "Check fc1(6→32), fc2(32→16), fc3(16→1)."
        )
        print("  ✓  test_exoplanet_classifier_param_count passed")
    except NotImplementedError as e:
        print(f"  (skipped – not implemented yet: {e})")


def _test_standardize():
    """7.2 — X_train_std has mean≈0, std≈1 per feature; val uses train stats."""
    try:
        X_tr_s, X_val_s = standardize(X_train_np.copy(), X_val_np.copy())
        assert X_tr_s is not None, "7.2 FAIL – returned None for X_train_std"
        assert X_val_s is not None, "7.2 FAIL – returned None for X_val_std"

        col_means = X_tr_s.mean(axis=0)
        col_stds  = X_tr_s.std(axis=0)
        assert np.allclose(col_means, 0, atol=1e-5), (
            f"7.2 FAIL – X_train_std column means not ≈ 0: {col_means}"
        )
        assert np.allclose(col_stds, 1, atol=1e-5), (
            f"7.2 FAIL – X_train_std column stds not ≈ 1: {col_stds}"
        )
        # val must use TRAIN statistics — not refit on val
        assert X_val_s.shape == X_val_np.shape, (
            f"7.2 FAIL – X_val_std shape {X_val_s.shape} "
            f"!= expected {X_val_np.shape}"
        )
        print("  ✓  test_standardize passed")
    except NotImplementedError as e:
        print(f"  (skipped – not implemented yet: {e})")


def _test_compute_accuracy():
    """7.3 — correct on trivial all-positive and all-negative cases."""
    try:
        # large positive logits → all predicted 1
        logits_pos = torch.full((20,), 10.0)
        labels_pos = torch.ones(20)
        assert compute_accuracy(logits_pos, labels_pos) == 1.0, (
            "7.3 FAIL – accuracy should be 1.0 for all-positive logits "
            "and all-positive labels"
        )
        # large negative logits → all predicted 0
        logits_neg = torch.full((20,), -10.0)
        labels_neg = torch.zeros(20)
        assert compute_accuracy(logits_neg, labels_neg) == 1.0, (
            "7.3 FAIL – accuracy should be 1.0 for all-negative logits "
            "and all-negative labels"
        )
        # half right
        logits_mix = torch.cat([torch.full((10,), 10.0),
                                 torch.full((10,), 10.0)])
        labels_mix = torch.cat([torch.ones(10), torch.zeros(10)])
        acc = compute_accuracy(logits_mix, labels_mix)
        assert abs(acc - 0.5) < 1e-5, (
            f"7.3 FAIL – expected 0.5 for half-correct, got {acc:.4f}"
        )
        print("  ✓  test_compute_accuracy passed")
    except NotImplementedError as e:
        print(f"  (skipped – not implemented yet: {e})")


def _test_train_classifier():
    """7.4 — trains to >82% validation accuracy; lengths and monotonicity."""
    try:
        torch.manual_seed(42)
        model = ExoplanetClassifier()
        model, train_losses, val_accuracies = train_classifier(
            model, X_train_np, y_train_np, X_val_np, y_val_np,
            lr=1e-3, batch_size=32, num_epochs=150,
        )
        assert len(train_losses) == 150, (
            f"7.4 FAIL – len(train_losses) = {len(train_losses)}, expected 150"
        )
        assert len(val_accuracies) == 150, (
            f"7.4 FAIL – len(val_accuracies) = {len(val_accuracies)}, "
            "expected 150"
        )
        assert train_losses[-1] < train_losses[0], (
            f"7.4 FAIL – loss did not decrease: "
            f"first={train_losses[0]:.4f}  last={train_losses[-1]:.4f}"
        )
        final_acc = val_accuracies[-1]
        assert final_acc > 0.82, (
            f"7.4 FAIL – final val accuracy {final_acc:.3f} < 0.82. "
            "Check your loss function, optimiser, and standardisation."
        )
        assert all(0 <= a <= 1 for a in val_accuracies), (
            "7.4 FAIL – some val_accuracies are outside [0, 1]"
        )
        print(f"  ✓  test_train_classifier passed  "
              f"(final val acc = {final_acc:.3f})")
    except NotImplementedError as e:
        print(f"  (skipped – not implemented yet: {e})")


if __name__ == "__main__":

    print("=" * 60)
    print("Section 7 — Everything In One Frame  [GRADED]")
    print("=" * 60)
    print()

    tests = [
        _test_exoplanet_classifier_shape,
        _test_exoplanet_classifier_param_count,
        _test_standardize,
        _test_compute_accuracy,
        _test_train_classifier,
    ]

    for fn in tests:
        try:
            fn()
        except NotImplementedError as e:
            print(f"  (skipped – not implemented yet: {e})")
        except AssertionError as e:
            print(f"  FAIL: {e}")

    print()
    print("=" * 60)
    print("Run check-cs240 when all tests pass, then call a TA.")
    print("=" * 60)