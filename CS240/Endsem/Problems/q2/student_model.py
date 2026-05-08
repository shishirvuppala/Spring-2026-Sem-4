import numpy as np


class FeedForwardNN:
    """
    TODO:
    This starter model is a plain 3-layer linear network with one hidden layer.
    Modify or extend this class in any way you like, but keep input dimension 1
    and output dimension 1.
    """

    def __init__(self, hidden_dim=8, seed=0):
        self.layer_dims = [1, hidden_dim, 1]
        self.params = {}

        rng = np.random.default_rng(seed)
        self.params["W1"] = rng.standard_normal((1, hidden_dim)) * 0.1
        self.params["b1"] = np.zeros((1, hidden_dim), dtype=np.float64)
        self.params["W2"] = rng.standard_normal((hidden_dim, 1)) * 0.1
        self.params["b2"] = np.zeros((1, 1), dtype=np.float64)

    @staticmethod
    def mse_loss(y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        h = x @ self.params["W1"] + self.params["b1"]
        y = h @ self.params["W2"] + self.params["b2"]
        cache = {"x": x, "h": h, "y": y}
        return y, cache

    def backward(self, y_pred, y_true, cache):
        m = y_true.shape[0]
        grads = {}

        dy = (2.0 / m) * (y_pred - y_true)
        grads["W2"] = cache["h"].T @ dy
        grads["b2"] = np.sum(dy, axis=0, keepdims=True)

        dh = dy @ self.params["W2"].T
        grads["W1"] = cache["x"].T @ dh
        grads["b1"] = np.sum(dh, axis=0, keepdims=True)
        return grads

    def predict(self, x):
        y_pred, _ = self.forward(x)
        return y_pred
