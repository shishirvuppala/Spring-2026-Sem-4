"""
easy.py
==============================
NumPy → PyTorch Inspection Exercise

No functions to implement — just read both implementations side by side
and make sure you understand what each line does and why.

Pipeline  (B = batch, N = items per batch, D = feature dim)
────────────────────────────────────────────────────────────
  X            (B, N, D)   raw input
  norms        (B, N, 1)   per-vector L2 norms
  X_tilde      (B, N, D)   L2-normalised rows
  S            (B, N, N)   batched pairwise cosine similarities
  m            (B,)        per-batch mean similarity
"""

import numpy as np
import torch


# ══════════════════════════════════════════════════════════════════
#  NUMPY  –  reference implementation
# ══════════════════════════════════════════════════════════════════

def cosine_similarity_pipeline_np(
    X: np.ndarray,          # (B, N, D)
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    X   : (B, N, D)   raw feature tensor
    eps : float        small constant for numerical stability

    Returns
    -------
    S : (B, N, N)   pairwise cosine-similarity matrices
    m : (B,)        per-batch mean similarity
    """

    # ── L2 norm of each feature vector ────────────────────────────
    # np.linalg.norm with axis=-1 reduces along D, giving (B, N).
    # keepdims=True keeps the trailing axis so shape stays (B, N, 1),
    # which lets us broadcast-divide X of shape (B, N, D).
    norms = np.linalg.norm(X, axis=-1, keepdims=True)   # (B, N, 1)

    # ── L2-normalise each vector ──────────────────────────────────
    # Adding eps before dividing prevents division by zero when a
    # vector is all-zeros.
    X_tilde = X / (norms + eps)                          # (B, N, D)

    # ── Batched pairwise cosine similarities ──────────────────────
    # For each b: S[b] = X_tilde[b] @ X_tilde[b].T
    # np.matmul broadcasts over the leading batch dimension, so
    # (B, N, D) @ (B, D, N) → (B, N, N) in one call.
    # .transpose(0, 2, 1) swaps the last two axes: (B, N, D) → (B, D, N).
    S = np.matmul(X_tilde, X_tilde.transpose(0, 2, 1))  # (B, N, N)

    # ── Per-batch mean similarity ─────────────────────────────────
    # Average over both spatial axes (i and j) for each batch element.
    # axis=(1, 2) collapses both, leaving shape (B,).
    m = S.mean(axis=(1, 2))                              # (B,)

    return S, m


# ══════════════════════════════════════════════════════════════════
#  PYTORCH  –  read carefully and compare to the numpy version
# ══════════════════════════════════════════════════════════════════

def cosine_similarity_pipeline_torch(
    X: torch.Tensor,        # (B, N, D)
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Parameters
    ----------
    X   : Tensor  (B, N, D)   raw feature tensor
    eps : float                small constant for numerical stability

    Returns
    -------
    S : Tensor  (B, N, N)   pairwise cosine-similarity matrices
    m : Tensor  (B,)        per-batch mean similarity

    Translation notes (numpy → torch)
    ──────────────────────────────────
    np.linalg.norm(X, axis=-1, keepdims=True)
        → torch.linalg.norm(X, dim=-1, keepdim=True)
          • 'axis' becomes 'dim'
          • 'keepdims' becomes 'keepdim'  (no trailing 's')

    X_tilde.transpose(0, 2, 1)
        → X_tilde.transpose(1, 2)
          • numpy transpose takes a full axis tuple; torch's two-arg form
            just names the two axes to swap (the others stay put)

    np.matmul(A, B)
        → torch.bmm(A, B)   or equivalently   A @ B
          • torch.bmm is the explicit batched matrix-multiply; both A and B
            must be exactly 3-D
          • @ dispatches to bmm automatically for 3-D tensors

    S.mean(axis=(1, 2))
        → S.mean(dim=(1, 2))
          • 'axis' → 'dim'; tuple syntax works the same way
    """

    # ── L2 norm of each feature vector ────────────────────────────
    # torch.linalg.norm mirrors np.linalg.norm; 'dim' replaces 'axis'
    # and 'keepdim' replaces 'keepdims'.
    norms = torch.linalg.norm(X, dim=-1, keepdim=True)  # (B, N, 1)

    # ── L2-normalise ──────────────────────────────────────────────
    X_tilde = X / (norms + eps)                          # (B, N, D)

    # ── Batched pairwise cosine similarities ──────────────────────
    # X_tilde.transpose(1, 2) swaps dims 1 and 2: (B, N, D) → (B, D, N).
    # torch.bmm then does B independent (N,D)@(D,N) multiplications,
    # producing (B, N, N).  The @ operator is equivalent.
    S = torch.bmm(X_tilde, X_tilde.transpose(1, 2))     # (B, N, N)
    # you can use '@' instead of torch.bmm:
    # S = X_tilde @ X_tilde.transpose(1, 2) # works!!!

    # ── Per-batch mean similarity ─────────────────────────────────
    # 'dim' accepts a tuple just like numpy's 'axis'.
    m = S.mean(dim=(1, 2))                               # (B,)

    return S, m


# ══════════════════════════════════════════════════════════════════
#  UNIT TESTS
# ══════════════════════════════════════════════════════════════════

def _check(t: torch.Tensor, ref: np.ndarray,
           atol: float = 1e-5, label: str = "") -> bool:
    ok = torch.allclose(t.float(),
                        torch.from_numpy(ref.astype(np.float32)),
                        atol=atol)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return ok


def run_tests():
    sep = "=" * 54
    print(sep)
    print("  cosine_similarity_pipeline  –  unit tests")
    print(sep)

    # ── shared random input ────────────────────────────────────────
    rng = np.random.default_rng(42)
    B, N, D = 4, 6, 8
    X_np = rng.standard_normal((B, N, D)).astype(np.float32)
    X_t  = torch.from_numpy(X_np)

    S_np, m_np = cosine_similarity_pipeline_np(X_np)
    S_t,  m_t  = cosine_similarity_pipeline_torch(X_t)

    # ── shape checks ──────────────────────────────────────────────
    print("\nShape checks")
    assert S_t.shape == (B, N, N), f"S shape wrong: {S_t.shape}"
    print(f"  [PASS] S shape is {tuple(S_t.shape)}")
    assert m_t.shape == (B,), f"m shape wrong: {m_t.shape}"
    print(f"  [PASS] m shape is {tuple(m_t.shape)}")

    # ── value checks against numpy reference ─────────────────────
    print("\nValue checks (torch vs numpy reference)")
    _check(S_t, S_np, label="S values match numpy")
    _check(m_t, m_np, label="m values match numpy")

    # ── diagonal entries should be ~1 ─────────────────────────────
    print("\nSanity checks")
    diag_err = (torch.diagonal(S_t, dim1=1, dim2=2) - 1.0).abs().max().item()
    ok_diag = diag_err < 1e-5
    print(f"  [{'PASS' if ok_diag else 'FAIL'}] diagonal entries ≈ 1  "
          f"(max err = {diag_err:.2e})")

    # ── S should be symmetric ─────────────────────────────────────
    sym_err = (S_t - S_t.transpose(1, 2)).abs().max().item()
    ok_sym = sym_err < 1e-5
    print(f"  [{'PASS' if ok_sym else 'FAIL'}] S is symmetric  "
          f"(max err = {sym_err:.2e})")

    # ── values in [-1, 1] ─────────────────────────────────────────
    in_range = (S_t >= -1.0 - 1e-5).all() and (S_t <= 1.0 + 1e-5).all()
    print(f"  [{'PASS' if in_range else 'FAIL'}] all S values in [-1, 1]")

    # ── D=1 edge case: scalars, cos-sim reduces to ±1 ─────────────
    print("\nEdge case: D=1")
    X1_np = rng.standard_normal((2, 5, 1)).astype(np.float32)
    X1_t  = torch.from_numpy(X1_np)
    S1_np, _ = cosine_similarity_pipeline_np(X1_np)
    S1_t,  _ = cosine_similarity_pipeline_torch(X1_t)
    _check(S1_t, S1_np, label="D=1 values match numpy")
    d1_range = (S1_t.abs() <= 1.0 + 1e-5).all()
    print(f"  [{'PASS' if d1_range else 'FAIL'}] D=1 values in [-1, 1]")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    run_tests()