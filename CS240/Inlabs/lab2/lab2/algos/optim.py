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