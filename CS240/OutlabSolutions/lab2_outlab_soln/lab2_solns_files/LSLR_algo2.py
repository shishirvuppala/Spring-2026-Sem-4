import numpy as np
from typing import Callable, Optional, Tuple, List
from functions.func import func
from .optim import LSLROptimiser


class LSLRAlgo2(LSLROptimiser):
    
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 1000000) -> None:
        super().__init__(X, y)
        
        ## TODO Use this for any pre-computations you need
        self.batch_size = min(batch_size, self.n_samples)
        
        # Pre-compute row norms for importance sampling
        self.row_norms_sq = np.sum(self.X**2, axis=1)  # Shape (n,)
        self.row_norms_sq[self.row_norms_sq == 0] = 1e-10
        self.probs = self.row_norms_sq / np.sum(self.row_norms_sq)
        
        # Pre-compute importance sampling weights: 1 / (n * p_i)
        self.sample_weights = 1.0 / (self.n_samples * self.probs)
        ##

    def lr(self) -> float:
        ## TODO learning rate schedule
        return 1
    
    def step(self, params: np.ndarray) -> np.ndarray:
        ## TODO Implement the step method
        # Sample batch indices with importance sampling
        indices = np.random.choice(
            self.n_samples, 
            size=self.batch_size, 
            p=self.probs
        )
        
        # Form mini-batch
        X_batch = self.X[indices]  # (batch_size, n_features)
        y_batch = self.y[indices]  # (batch_size,)
        weights = self.sample_weights[indices]  # (batch_size,)
        
        # Compute residuals
        residuals = X_batch @ params - y_batch
        
        # Compute importance-weighted gradient
        weighted_residuals = weights * residuals
        grad = (2.0 / self.batch_size) * (X_batch.T @ weighted_residuals)
        
        # Adaptive step size via line search on batch
        X_grad = X_batch @ grad
        numerator = np.dot(X_grad, residuals)
        denominator = np.sum(X_grad**2)
        
        if denominator < 1e-10:
            step_size = 0.0
        else:
            step_size = numerator / denominator
        
        # Update parameters
        w_next = params - step_size * grad
        return w_next
    
    def eval_lslr(self, w: np.ndarray) -> float:
        ## TODO Evaluate LSLR objective: (1/n)||Xw - y||^2
        residuals = self.X @ w - self.y
        return float(np.mean(residuals**2))
    def full_grad(self, w: np.ndarray) -> np.ndarray:
        ## TODO 
        residuals = self.X @ w - self.y
        return (2.0 / self.n_samples) * (self.X.T @ residuals)
    def stoch_grad(self, w: np.ndarray, gamma: int) -> np.ndarray:
        """
        Compute stochastic gradient for a single sample index gamma.
        G(w, γ) = ∇f_γ(w) = x_γ (x_γ^T w - y_γ)
        This is the gradient contribution from the gamma-th sample.
        """
        ## TODO Implement stochastic gradient computation
        x_i = self.X[gamma]  # i-th sample (feature vector)
        y_i = self.y[gamma]  # i-th target
        residual = np.dot(x_i, w) - y_i
        return (2.0 / self.n_samples) * residual * x_i

