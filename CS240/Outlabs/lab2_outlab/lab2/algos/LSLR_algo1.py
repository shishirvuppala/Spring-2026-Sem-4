import numpy as np
from typing import Callable, Optional, Tuple, List
from functions.func import func
from .optim import LSLROptimiser


class LSLRAlgo1(LSLROptimiser):
    """
    Gradient Descent for LSLR with optimal learning rate.
    
    Uses η = 1/L where L is the Lipschitz constant (largest eigenvalue of Hessian).
    For f(w) = (1/2n)||Xw - y||^2, Hessian = (1/n) X^T X
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        super().__init__(X, y)
        
        ## TODO Use this for any pre-computations you need



        ##



    def lr(self) -> float:
        ## TODO learning rate schedule

        return 1
    def step(self, params: np.ndarray) -> np.ndarray:
        ## TODO Implement the step method

        raise NotImplementedError("Implement step method for LSLRAlgo1")
    def eval_lslr(self, w: np.ndarray) -> float:
        ## TODO Evaluate LSLR objective: (1/n)||Xw - y||^2

        raise NotImplementedError("Implement eval_lslr method for LSLRAlgo1")
    def full_grad(self, w: np.ndarray) -> np.ndarray:
        ## TODO 
        raise NotImplementedError("Implement full_grad method for LSLRAlgo1")
    def stoch_grad(self, w: np.ndarray, gamma: int) -> np.ndarray:
       
        ## TODO Implement stochastic gradient computation

        raise NotImplementedError("Implement stoch_grad method for LSLRAlgo1")

