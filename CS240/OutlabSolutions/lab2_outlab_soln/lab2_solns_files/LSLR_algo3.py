import numpy as np
from typing import Callable, Optional, Tuple, List
from functions.func import func
from .optim import LSLROptimiser


class LSLRAlgo3(LSLROptimiser):
    """
    SVRG-CD: Stochastic Variance Reduced Gradient with
    Randomized Coordinate Descent (importance sampling).

    Each call to step() runs one SVRG epoch:
      1. Snapshot w_tilde = w, compute full gradient mu = nabla f(w_tilde).
      2. Inner loop of m iterations of coordinate updates with
         variance-reduced gradient estimator:
           v_t = mu + (1/p_gamma)(g_new - g_old) * e_gamma
         where g_new = [nabla f(w_t)]_gamma, g_old = mu[gamma].
      3. Return updated w.

    Coordinates are sampled with probability proportional to
    column norms squared (Randomized Kaczmarz / importance sampling).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        super().__init__(X, y)

        # --- Persistent pre-computations (O(nd + d^2)) ---
        self.Gram = self.X.T @ self.X                       # (d, d)
        self.Xy = self.X.T @ self.y                         # (d,)
        self.y_norm_sq = np.sum(self.y ** 2)                 # scalar

        # Column norms = diag(Gram) = coordinate Lipschitz constants
        self.col_norms_sq = np.diag(self.Gram).copy()       # (d,)
        self.col_norms_sq[self.col_norms_sq == 0] = 1e-10

        # Importance sampling probabilities: p_j = L_j / sum(L)
        self.frobenius_sq = np.sum(self.col_norms_sq)
        self.probs = self.col_norms_sq / self.frobenius_sq  # (d,)

        # Inner loop length (standard choice: m = 2d)
        self.m = 2 * self.n_features

    def lr(self) -> float:
        # eta = 1 / (2 * L_max) where L_max relates to the Frobenius norm
        # For importance-sampled CD-SVRG a safe and effective rate:
        return 1.0 / self.frobenius_sq

    def step(self, params: np.ndarray) -> np.ndarray:
        """
        One SVRG epoch: snapshot + m inner coordinate updates.
        All heavy products are maintained incrementally.
        """
        d = self.n_features
        n = self.n_samples
        eta = self.lr()
        m = self.m
        Gram = self.Gram
        Xy = self.Xy
        probs = self.probs
        col_norms_sq = self.col_norms_sq

        # --- Epoch-level (O(d^2) amortised) ---
        w = params.copy()

        # Full gradient at snapshot: mu = (2/n)(Gram @ w_tilde - Xy)
        Gw_tilde = Gram @ w                           # O(d^2)
        mu = (2.0 / n) * (Gw_tilde - Xy)              # (d,)

        # Cache Gram @ mu for incremental Gw updates
        G_mu = Gram @ mu                               # O(d^2)

        # Maintain Gw = Gram @ w incrementally
        Gw = Gw_tilde.copy()                           # O(d)

        # --- Inner loop (O(d) per iteration) ---
        for _ in range(m):
            # 1. Sample coordinate  -- O(d) due to categorical sampling
            gamma = np.random.choice(d, p=probs)

            # 2. Partial derivatives -- O(1) via maintained Gw
            g_new = (2.0 / n) * (Gw[gamma] - Xy[gamma])   # current
            g_old = mu[gamma]                                # snapshot

            # 3. Variance-reduced scalar correction
            delta = (g_new - g_old) / probs[gamma]           # scalar

            # 4. Update w:  w <- w - eta * (mu + delta * e_gamma)
            w -= eta * mu                                    # O(d)
            w[gamma] -= eta * delta                          # O(1) correction

            # 5. Update Gw incrementally:
            #    Gw_new = G @ w_new
            #           = Gw - eta * G_mu - eta * delta * G[:, gamma]
            Gw -= eta * G_mu + eta * delta * Gram[:, gamma]  # O(d)

        self.epochs_done += 1
        return w

    def eval_lslr(self, w: np.ndarray) -> float:
        """Evaluate (1/n)||Xw - y||^2 using pre-computed Gram quantities."""
        term1 = w @ (self.Gram @ w)
        term2 = -2.0 * (w @ self.Xy)
        term3 = self.y_norm_sq
        return (term1 + term2 + term3) / self.n_samples

    def full_grad(self, w: np.ndarray) -> np.ndarray:
        """Full gradient: (2/n)(X^T X w - X^T y)."""
        return (2.0 / self.n_samples) * (self.Gram @ w - self.Xy)

    def stoch_grad(self, w: np.ndarray, gamma: int) -> np.ndarray:
        """
        Coordinate stochastic gradient (unscaled by 1/p_gamma).
        G(w, gamma) = e_gamma * [nabla f(w)]_gamma
        """
        grad = np.zeros_like(w)
        partial = (2.0 / self.n_samples) * (np.dot(self.Gram[gamma], w) - self.Xy[gamma])
        grad[gamma] = partial
        return grad

