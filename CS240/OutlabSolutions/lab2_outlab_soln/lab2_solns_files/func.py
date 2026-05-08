import numpy as np
from typing import Callable, Optional, Tuple, List, Any, Union

class func:
    def __init__(self):
        pass
    def __call__(self, x: np.ndarray) -> np.ndarray: # type: ignore
        return self.eval(x)
    def eval(self, x: np.ndarray) -> np.ndarray:# type: ignore
        pass
    def grad(self, x: np.ndarray) -> np.ndarray: # type: ignore
        pass
    def hessian(self, x: np.ndarray) -> np.ndarray: # type: ignore
        pass 


class LSLR(func):
    def __init__(self, X: np.ndarray, y: np.ndarray, debug: bool = False) -> None:
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape
        self.debug = debug
        if self.debug:
            print(f"LSLR func instantiated with shape {self.n_samples}x{self.n_features}")
        super().__init__()

    def eval(self, w: np.ndarray) -> float: # type: ignore
        ## TODO 

        output = (1/2*self.n_samples)*((self.X @ w - self.y)**2)
        if self.debug:
            print(f"LSLR eval output shape is {output.shape}") 
        
        return output

    def grad(self, w: np.ndarray) -> np.ndarray: # type: ignore
        ## TODO 
        grad = (1/self.n_samples)*(self.X.transpose()@ (self.X @ w - self.y))
        if self.debug:
            print(f"LSLR gradient output shape is {grad.shape}")
        return grad
    
    def hessian(self, w: np.ndarray) -> np.ndarray: # type: ignore
        ## TODO
        hess = (1/self.n_samples)*(self.X.transpose()@self.X)
        if self.debug:
            print(f"LSLR Hessian output shape is {hess.shape}")
        return hess

class LSLR_alt(func):
    def __init__(self, X: np.ndarray, y: np.ndarray, debug: bool = False) -> None:
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape
        self.debug = debug
        if self.debug:
            print(f"LSLR func instantiated with shape {self.n_samples}x{self.n_features}")
        super().__init__()

    def eval(self, w: np.ndarray) -> float: # type: ignore
        ## TODO 

        output = (1/self.n_samples)*((self.X @ w - self.y)**2)
        if self.debug:
            print(f"LSLR eval output shape is {output.shape}") 
        
        return output

    def grad(self, w: np.ndarray) -> np.ndarray: # type: ignore
        ## TODO 
        grad = (2/self.n_samples)*(self.X.transpose()@ (self.X @ w - self.y))
        if self.debug:
            print(f"LSLR gradient output shape is {grad.shape}")
        return grad
    
    def hessian(self, w: np.ndarray) -> np.ndarray: # type: ignore
        ## TODO
        hess = (2/self.n_samples)*(self.X.transpose()@self.X)
        if self.debug:
            print(f"LSLR Hessian output shape is {hess.shape}")
        return hess



class rosenbrock(func):
    def __init__(self, a: float = 1.0, b: float = 100.0) -> None:
        self.a = a
        self.b = b
        super().__init__()

    def eval(self, x: np.ndarray) -> np.ndarray: # type: ignore
       ## TODO: Implement the Rosenbrock function evaluation
        pass
    def grad(self, x: np.ndarray) -> np.ndarray: # type: ignore
        ## TODO: Implement the Rosenbrock function gradient
        pass
    def hessian(self, x: np.ndarray) -> np.ndarray: # type: ignore
        ## TODO: Implement the Rosenbrock function Hessian
        pass
class rot_anisotropic(func):
    def __init__(self, U: np.ndarray, V: np.ndarray, S: np.ndarray, b: np.ndarray) -> None:
        self.U = U
        self.V = V
        self.S = S
        self.b = b
        super().__init__()

    def eval(self, x: np.ndarray) -> np.ndarray: # type: ignore
        ## TODO: Implement the rotated anisotropic function evaluation
        pass
    def grad(self, x: np.ndarray) -> np.ndarray: # type: ignore
        ## TODO: 
        pass
    def hessian(self, x: np.ndarray) -> np.ndarray: # type: ignore
        ## TODO:
        pass

if __name__ == "__main__":
    pass