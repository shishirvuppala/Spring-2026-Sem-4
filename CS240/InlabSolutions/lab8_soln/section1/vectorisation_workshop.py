"""
vectorisation_workshop.py
=========================
NumPy → PyTorch Vectorisation Workshop

Reference NumPy implementations are provided for every step.
Your job is to complete the THREE PyTorch functions marked  ← TODO.
All other PyTorch functions are given as worked examples – read them
carefully; they demonstrate the key translation patterns.

Full pipeline  (B = batch size, N = feature dim)
────────────────────────────────────────────────
  x1, x2      (B, N)      raw samples + softmax         ← TODO
  z           (B, N, N)   pairwise matrix + noise + T   ← TODO
  x_flat      (B, N²)     flattened                     [given]
  mu          (N²,)       sample mean                   [given]
  Sigma       (N², N²)    sample covariance              ← TODO
  ll          scalar      log-likelihood                 [given]
"""

import numpy as np
import torch
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════
#  NUMPY  –  reference implementations  (do NOT modify)
# ══════════════════════════════════════════════════════════════════

def generate_data_np(B: int, N: int):
    """
    Sample x1 ~ N(0, 10·I) and x2 ~ Uniform(-100, 100).

    Returns
    -------
    x1 : ndarray  (B, N)
    x2 : ndarray  (B, N)
    """
    x1 = np.random.randn(B, N) * np.sqrt(10)
    x2 = np.random.uniform(-100.0, 100.0, size=(B, N))
    return x1, x2


def softmax_np(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax along axis=1."""
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def pairwise_matrix_np(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Batch outer product: z[b, i, j] = x1[b, i] * x2[b, j].  Shape (B, N, N)."""
    return x1[:, :, None] * x2[:, None, :]


def add_noise_np(z: np.ndarray) -> np.ndarray:
    """Add elementwise N(0,1) noise."""
    return z + np.random.randn(*z.shape)


def transpose_last_two_np(z: np.ndarray) -> np.ndarray:
    """Swap last two axes."""
    return z.transpose(0, 2, 1)


def flatten_np(z: np.ndarray) -> np.ndarray:
    """Flatten last two dimensions: (B, N, N) → (B, N²)."""
    return z.reshape(z.shape[0], -1)


def sample_mean_np(x: np.ndarray) -> np.ndarray:
    """ML mean over batch: (B, D) → (D,)."""
    return x.mean(axis=0)


def sample_cov_np(x: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """ML covariance  Σ̂ = (1/B) Σ (x[b]-μ)(x[b]-μ)ᵀ.  Shape (D, D)."""
    c = x - mu
    return (c.T @ c) / x.shape[0]


def log_likelihood_np(x: np.ndarray, mu: np.ndarray,
                      Sigma: np.ndarray, eps: float = 1e-6) -> float:
    """Total log-likelihood of the batch under N(mu, Sigma)."""
    D, B = mu.shape[0], x.shape[0]
    S = Sigma + eps * np.eye(D)
    S_inv = np.linalg.inv(S)
    _, log_det = np.linalg.slogdet(S)
    c = x - mu
    quad = np.sum((c @ S_inv) * c, axis=1)
    return float(
        -0.5 * quad.sum()
        - 0.5 * B * log_det
        - 0.5 * B * D * np.log(2.0 * np.pi)
    )


def pipeline_np(B: int, N: int, seed: int = 0) -> float:
    """Full NumPy pipeline.  Returns scalar log-likelihood."""
    np.random.seed(seed)
    x1, x2 = generate_data_np(B, N)
    x1, x2 = softmax_np(x1), softmax_np(x2)
    z = pairwise_matrix_np(x1, x2)
    z = add_noise_np(z)
    z = transpose_last_two_np(z)
    x_flat = flatten_np(z)
    mu = sample_mean_np(x_flat)
    S  = sample_cov_np(x_flat, mu)
    return log_likelihood_np(x_flat, mu, S)


# ══════════════════════════════════════════════════════════════════
#  PYTORCH  –  complete the three TODO functions
# ══════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────
def generate_data_torch(B: int, N: int):
    """
    Sample x1 and x2 as torch tensors, then apply softmax to each.

      x1 ~ N(0, 10·I)         → scale torch.randn by √10
      x2 ~ Uniform(-100,100)  → transform torch.rand from [0,1) to (-100,100)

    Apply softmax across the feature dimension (dim=1) to both before
    returning.

    Returns
    -------
    x1 : Tensor  (B, N)   rows sum to 1
    x2 : Tensor  (B, N)   rows sum to 1

    [TODO]
    Hints:
      • torch.randn(B, N) * (10 ** 0.5)
      • torch.rand(B, N) * 200.0 - 100.0
      • F.softmax(x, dim=1)
    """
    # ------------------------------------------------------------------ #
    #  YOUR CODE HERE                                                      #
    # ------------------------------------------------------------------ #
    raise NotImplementedError("generate_data_torch: implement me!")


# ──────────────────────────────────────────────────────────────────
def pairwise_matrix_torch(x1: torch.Tensor,
                           x2: torch.Tensor) -> torch.Tensor:
    """
    Compute the batch outer product, add noise, then transpose.

    Step (a) – outer product:
        z[b, i, j] = x1[b, i] * x2[b, j],  giving z ∈ R^{B × N × N}.
    Step (b) – add noise:
        z ← z + w,  where w ~ N(0, 1) elementwise, same shape as z.
    Step (c) – transpose:
        swap the last two axes: z[b] ← z[b]ᵀ.

    Parameters
    ----------
    x1 : Tensor  (B, N)
    x2 : Tensor  (B, N)

    Returns
    -------
    z  : Tensor  (B, N, N)

    [TODO]
    Hints:
      • x1.unsqueeze(2) * x2.unsqueeze(1) broadcasts to (B, N, N).
      • torch.randn_like(z) samples noise matching z's shape and dtype.
      • z.transpose(1, 2) swaps the last two dimensions.
    """
    # ------------------------------------------------------------------ #
    #  YOUR CODE HERE                                                      #
    # ------------------------------------------------------------------ #
    raise NotImplementedError("pairwise_matrix_torch: implement me!")


# ──────────────────────────────────────────────────────────────────
def flatten_torch(z: torch.Tensor) -> torch.Tensor:
    """
    Flatten the last two dimensions: (B, N, N) → (B, N²).

    .contiguous() is required before .view() here because the tensor
    may be non-contiguous in memory after the transpose in Step 2.

    [GIVEN]
    """
    return z.contiguous().view(z.shape[0], -1)


# ──────────────────────────────────────────────────────────────────
def sample_mean_torch(x: torch.Tensor) -> torch.Tensor:
    """
    ML mean over the batch: (B, D) → (D,).

    [GIVEN — note dim= replaces numpy's axis=]
    """
    return x.mean(dim=0)


# ──────────────────────────────────────────────────────────────────
def sample_cov_torch(x: torch.Tensor,
                      mu: torch.Tensor) -> torch.Tensor:
    """
    ML covariance  Σ̂ = (1/B) Σ (x[b]-μ)(x[b]-μ)ᵀ.

    Parameters
    ----------
    x  : Tensor  (B, D)
    mu : Tensor  (D,)

    Returns
    -------
    Sigma : Tensor  (D, D)

    [TODO]
    Hints:
      • mu has shape (D,).  To subtract it from every row of x (shape B×D),
        use mu.unsqueeze(0) to get shape (1, D) — it then broadcasts.
      • The centred matrix c has shape (B, D).
      • c.T @ c gives (D, D).  Divide by B = x.shape[0].
    """
    # ------------------------------------------------------------------ #
    #  YOUR CODE HERE                                                      #
    # ------------------------------------------------------------------ #
    raise NotImplementedError("sample_cov_torch: implement me!")


# ──────────────────────────────────────────────────────────────────
def log_likelihood_torch(x: torch.Tensor, mu: torch.Tensor,
                          Sigma: torch.Tensor,
                          eps: float = 1e-6) -> torch.Tensor:
    """
    Total log-likelihood of the batch under N(mu, Sigma).

    A small ridge ε·I is added to Sigma before inversion for numerical
    stability when B is small relative to D.

    [GIVEN — note torch.linalg.inv, torch.linalg.slogdet, and how
     mu.unsqueeze(0) broadcasts the mean subtraction over the batch]
    """
    D, B = mu.shape[0], x.shape[0]
    S = Sigma + eps * torch.eye(D, dtype=x.dtype)
    S_inv = torch.linalg.inv(S)
    _, log_det = torch.linalg.slogdet(S)
    c = x - mu.unsqueeze(0)
    quad = (c @ S_inv * c).sum(dim=1)
    return (
        -0.5 * quad.sum()
        - 0.5 * B * log_det
        - 0.5 * B * D * torch.log(torch.tensor(2.0 * torch.pi))
    )


# ──────────────────────────────────────────────────────────────────
def pipeline_torch(B: int, N: int, seed: int = 0) -> torch.Tensor:
    """Full PyTorch pipeline.  Returns scalar log-likelihood tensor."""
    torch.manual_seed(seed)
    x1, x2 = generate_data_torch(B, N)
    z       = pairwise_matrix_torch(x1, x2)
    x_flat  = flatten_torch(z)
    mu      = sample_mean_torch(x_flat)
    S       = sample_cov_torch(x_flat, mu)
    return log_likelihood_torch(x_flat, mu, S)


# ══════════════════════════════════════════════════════════════════
#  UNIT TESTS
# ══════════════════════════════════════════════════════════════════

def _np32(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32))


def _check(t: torch.Tensor, ref: np.ndarray,
           atol: float = 1e-4, label: str = "") -> bool:
    ok = torch.allclose(t.float(), _np32(ref), atol=atol)
    print(f"    [{'PASS' if ok else 'FAIL'}] {label}")
    return ok


# ── step tests ────────────────────────────────────────────────────

def test_generate_data():
    """generate_data_torch returns softmax-normalised rows."""
    print("test_generate_data_torch")
    torch.manual_seed(0)
    x1, x2 = generate_data_torch(8, 12)
    assert x1.shape == (8, 12), f"x1 wrong shape: {x1.shape}"
    assert x2.shape == (8, 12), f"x2 wrong shape: {x2.shape}"
    # after softmax every row must sum to 1
    ones = torch.ones(8)
    ok1 = torch.allclose(x1.sum(dim=1), ones, atol=1e-5)
    ok2 = torch.allclose(x2.sum(dim=1), ones, atol=1e-5)
    print(f"    [{'PASS' if ok1 else 'FAIL'}] x1 rows sum to 1")
    print(f"    [{'PASS' if ok2 else 'FAIL'}] x2 rows sum to 1")
    # values must be non-negative (softmax output)
    print(f"    [{'PASS' if x1.min() >= 0 else 'FAIL'}] x1 all non-negative")
    print(f"    [{'PASS' if x2.min() >= 0 else 'FAIL'}] x2 all non-negative")


def test_pairwise_matrix():
    """pairwise_matrix_torch: shape, outer-product structure, transpose."""
    print("test_pairwise_matrix_torch")
    # use deterministic inputs so we can check the outer product before noise
    torch.manual_seed(42)
    B, N = 3, 5
    x1 = torch.rand(B, N)
    x2 = torch.rand(B, N)

    torch.manual_seed(42)          # reset so noise is seeded
    z = pairwise_matrix_torch(x1, x2)
    assert z.shape == (B, N, N), f"wrong shape: {z.shape}"
    print(f"    [PASS] output shape {tuple(z.shape)}")

    # The result includes N(0,1) noise + transpose, so we verify the
    # transpose by checking z[b, i, j] and z[b, j, i] are NOT equal on
    # average across off-diagonal entries (would fail if transpose was skipped
    # AND noise is zero, which it never is).
    # A minimal structural check: output must not be symmetric (noise + T means
    # z[b] = outer(x1,x2)^T + w, which is asymmetric with probability 1).
    sym_err = (z - z.transpose(1, 2)).abs().mean().item()
    print(f"    [INFO] mean |z - zᵀ| = {sym_err:.4f}  (> 0 expected after noise)")


def test_sample_cov():
    """sample_cov_torch matches numpy reference."""
    print("test_sample_cov_torch")
    rng  = np.random.default_rng(11)
    x_np = rng.standard_normal((16, 8)).astype(np.float32)
    mu_np = sample_mean_np(x_np)
    S_np  = sample_cov_np(x_np, mu_np)

    x_t  = torch.from_numpy(x_np)
    mu_t = sample_mean_torch(x_t)
    S_t  = sample_cov_torch(x_t, mu_t)

    assert S_t.shape == (8, 8), f"wrong shape: {S_t.shape}"
    _check(S_t, S_np, label="values match numpy reference")
    sym_err = (S_t - S_t.T).abs().max().item()
    print(f"    [INFO] symmetry error = {sym_err:.2e}  (should be ~0)")


# ── pipeline test ──────────────────────────────────────────────────

def test_pipeline(B: int = 16, N: int = 8):
    """End-to-end smoke test: pipeline runs and returns a scalar."""
    print("test_pipeline_torch")
    ll = pipeline_torch(B, N, seed=0)
    assert ll.shape == torch.Size([]), f"expected scalar, got {ll.shape}"
    print(f"    [PASS] pipeline returned ll = {ll.item():.4f}")


# ── runner ────────────────────────────────────────────────────────

def run_all_tests():
    sep = "=" * 54
    step_tests = [
        test_generate_data,
        test_pairwise_matrix,
        test_sample_cov,
    ]

    print(sep)
    print("  UNIT TESTS  (step-by-step)")
    print(sep)
    for fn in step_tests:
        try:
            fn()
        except NotImplementedError as e:
            print(f"    [SKIP] not implemented yet – {e}")
        except AssertionError as e:
            print(f"    [FAIL] assertion – {e}")
        print()

    print(sep)
    print("  PIPELINE TEST  (end-to-end)")
    print(sep)
    try:
        test_pipeline()
    except NotImplementedError as e:
        print(f"    [SKIP] pipeline blocked – {e}")
    except Exception as e:
        print(f"    [FAIL] {e}")
    print()


if __name__ == "__main__":
    run_all_tests()