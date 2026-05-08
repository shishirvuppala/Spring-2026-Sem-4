import numpy as np


class FeedForwardNN:
    def __init__(self, layer_dims=None, seed=0):
        if layer_dims is None:
            layer_dims = [1, 8, 8, 8, 1]
        self.layer_dims = list(layer_dims)
        self.num_layers = len(self.layer_dims) - 1
        self.params = {}

        rng = np.random.default_rng(seed)
        for l in range(1, self.num_layers + 1):
            fan_in = self.layer_dims[l - 1]
            fan_out = self.layer_dims[l]
            scale = np.sqrt(2.0 / fan_in) if l < self.num_layers else np.sqrt(1.0 / fan_in)
            self.params[f"W{l}"] = rng.standard_normal((fan_in, fan_out)) * scale
            self.params[f"b{l}"] = np.zeros((1, fan_out), dtype=np.float64)

    @staticmethod
    def relu(z):
        return np.maximum(0.0, z)

    @staticmethod
    def relu_grad(z):
        return (z > 0).astype(np.float64)

    @staticmethod
    def mse_loss(y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def forward(self, x):
        a = np.asarray(x, dtype=np.float64)
        activations = [a]
        preactivations = []

        for l in range(1, self.num_layers + 1):
            W = self.params[f"W{l}"]
            b = self.params[f"b{l}"]
            z = a @ W + b
            preactivations.append(z)
            if l < self.num_layers:
                a = self.relu(z)
            else:
                a = z
            activations.append(a)

        cache = {"activations": activations, "preactivations": preactivations}
        return a, cache

    def backward(self, y_pred, y_true, cache):
        grads = {}
        m = y_true.shape[0]
        dz = (2.0 / m) * (y_pred - y_true)

        for l in range(self.num_layers, 0, -1):
            a_prev = cache["activations"][l - 1]
            W = self.params[f"W{l}"]
            grads[f"W{l}"] = a_prev.T @ dz
            grads[f"b{l}"] = np.sum(dz, axis=0, keepdims=True)

            if l > 1:
                da_prev = dz @ W.T
                dz = da_prev * self.relu_grad(cache["preactivations"][l - 2])

        return grads

    def predict(self, x):
        y_pred, _ = self.forward(x)
        return y_pred
