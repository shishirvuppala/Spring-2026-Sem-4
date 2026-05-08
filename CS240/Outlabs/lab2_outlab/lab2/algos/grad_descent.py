import numpy as np
from typing import List
from .optim import Optimiser
from functions.func import func


class GradientDescent(Optimiser):
    """Standard Gradient Descent optimizer."""
    
    def __init__(self, f: func, lr: float = 0.01) -> None:
        """
        Args:
            f: Function to optimize
            lr: Learning rate (step size)
        """
        super().__init__(f)
        self.lr = lr
    
    def step(self, params: np.ndarray) -> np.ndarray: #type: ignore
        """
        Performs one gradient descent step.
        
        Args:
            params: Current parameters x_t
        
        Returns:
            Updated parameters x_{t+1}
        """
        ## TODO Implement the gradient descent step
        pass

if __name__ == "__main__":
    # Simple test
    from functions.func import rosenbrock
    
    f = rosenbrock(a=1.0, b=100.0)
    optimizer = GradientDescent(f, lr=0.001)
    
    x = np.array([-1.0, 2.0])
    print(f"Initial x: {x}")
    print(f"Initial f(x): {f.eval(x)}")
    
    # Run a few iterations
    for i in range(10):
        x = optimizer.step(x)
        print(f"Iteration {i+1}: x = {x}, f(x) = {f.eval(x):.6f}")
