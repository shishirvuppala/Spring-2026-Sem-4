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
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape
        super().__init__()

    def eval(self, w: np.ndarray) -> float: # type: ignore
        ## TODO 

        raise NotImplementedError("Function evaluation not implemented yet.")

    def grad(self, w: np.ndarray) -> np.ndarray: # type: ignore
        ## TODO 
        raise NotImplementedError("Gradient computation not implemented yet.")

    def hessian(self, w: np.ndarray) -> np.ndarray: # type: ignore
        ## TODO

        raise NotImplementedError("Hessian computation not implemented yet.")


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