import numpy as np
from typing import Callable, Optional, Tuple, List
from functions.func import func


class Optimiser:
    def __init__(self, f:func) -> None:
        self.f = f

    def step(self, params: np.ndarray) -> np.ndarray: #type: ignore
        """
        gives you x_{t+1} given x_t
        """
        pass


if __name__ == "__main__":
    pass


class LSLROptimiser:
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape
        self.epochs_done = 0
    def lr(self) -> float:
        """
        Implements the learning rate schedule
        """
        t = self.epochs_done + 1
        return 1.0
    
    def step(self, params: np.ndarray) -> np.ndarray: #type: ignore
        """
        - called at every epoch.
        - a single epoch can have many iterations too, for example the inner loop of SVRG)
        - gives you w_{t+1} given w_t
        """
        final_params = params.copy()
        learning_rate = self.lr()

        full_gradient = self.full_grad(final_params)
        final_params -= learning_rate * full_gradient
        self.epochs_done += 1
        return final_params


    def eval_lslr(self, w: np.ndarray) -> float:
        residuals = self.X @ w - self.y
        return (1/2*self.n_samples) * (np.linalg.norm(residuals, ord=2)**2)
    
    def full_grad(self, w: np.ndarray) -> np.ndarray:
        residuals = self.X @ w - self.y
        return (self.X.T @ residuals) / self.n_samples
    
    def stoch_grad(self, w: np.ndarray, gamma: int) -> np.ndarray:
        residuals = self.X @ w - self.y
        return self.X.T[gamma,:] @ residuals /self.n_samples





if __name__ == "__main__":
    pass
