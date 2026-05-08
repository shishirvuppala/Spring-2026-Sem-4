import numpy as np
from typing import Callable, Optional, Tuple, List
from functions.func import func
from .optim import LSLROptimiser

class LSLRAlgo1(LSLROptimiser):
    
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        super().__init__(X, y)
        ## TODO Use this for any pre-computations you need
        self.Gram = self.X.T @ self.X
        self.Xy = self.X.T @ self.y
        self.y_norm_sq = np.sum(self.y ** 2)
        self.col_norms_sq = np.diag(self.Gram).copy()
        self.col_norms_sq[self.col_norms_sq == 0] = 1e-10
        self.probs = self.col_norms_sq / np.sum(self.col_norms_sq)
        ##

    def lr(self) -> float:
        ## TODO 
        return 1.0

    def step(self, params: np.ndarray) -> np.ndarray:
        ## TODO 
        n_features = self.X.shape[1]
        gamma = np.random.choice(n_features, p=self.probs)
        
        grad_numerator = np.dot(self.Gram[gamma], params) - self.Xy[gamma]
        step_size = grad_numerator / self.col_norms_sq[gamma]
        
        w_next = params.copy()
        w_next[gamma] -= step_size
        return w_next

    def eval_lslr(self, w: np.ndarray) -> float:
        ## TODO 
        term1 = w @ (self.Gram @ w)
        term2 = -2 * (w @ self.Xy)
        term3 = self.y_norm_sq
        return (term1 + term2 + term3) / self.n_samples
    
    def full_grad(self, w: np.ndarray) -> np.ndarray:
        ## TODO 
        grad = (self.Gram @ w) - self.Xy
        return (2.0 / self.n_samples) * grad
    
    def stoch_grad(self, w: np.ndarray, gamma: int) -> np.ndarray:   
        ## TODO 
        grad = np.zeros_like(w)
        partial_deriv = (np.dot(self.Gram[gamma], w) - self.Xy[gamma]) / self.n_samples
        grad[gamma] = partial_deriv
        return grad
