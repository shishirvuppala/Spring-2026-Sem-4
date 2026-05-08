import numpy as np


def tent_map(x):
    x = np.asarray(x, dtype=np.float64)
    return np.maximum(0.0, 1.0 - 2.0 * np.abs(x - 0.5))


def target_function(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    y = x.copy()
    for _ in range(4):
        y = tent_map(y)
    return y
