import numpy as np


class SGD:
    """
    Plain gradient descent
    """

    def __init__(self, lr=1e-3):
        self.lr = lr

    def step(self, params, grads):
        for key in params:
            params[key] -= self.lr * grads[key]


class Momentum:
    """
    SGD + momentum
    v = beta * v - lr * grad
    theta = theta + v
    """

    def __init__(self, lr=1e-3, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.velocity = {}

    def step(self, params, grads):
        for key in params:
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])

            self.velocity[key] = self.beta * \
                self.velocity[key] - self.lr * grads[key]
            params[key] += self.velocity[key]


class Nesterov:
    """
    Nesterov momentum
    """

    def __init__(self, lr=1e-3, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.velocity = {}

    def step(self, params, grads):
        for key in params:
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])

            v_prev = self.velocity[key].copy()
            self.velocity[key] = self.beta * \
                self.velocity[key] - self.lr * grads[key]
            params[key] += -self.beta * v_prev + \
                (1 + self.beta) * self.velocity[key]


class AdaGrad:
    """
    AdaGrad
    """

    def __init__(self, lr=1e-2, eps=1e-8):
        self.lr = lr
        self.eps = eps
        self.cache = {}

    def step(self, params, grads):
        for key in params:
            if key not in self.cache:
                self.cache[key] = np.zeros_like(params[key])

            self.cache[key] += grads[key] ** 2
            params[key] -= self.lr * grads[key] / \
                (np.sqrt(self.cache[key]) + self.eps)


class RMSProp:
    """
    RMSProp
    """

    def __init__(self, lr=1e-3, beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.cache = {}

    def step(self, params, grads):
        for key in params:
            if key not in self.cache:
                self.cache[key] = np.zeros_like(params[key])

            self.cache[key] = self.beta * self.cache[key] + \
                (1 - self.beta) * (grads[key] ** 2)
            params[key] -= self.lr * grads[key] / \
                (np.sqrt(self.cache[key]) + self.eps)


class Adam:
    """
    Adam
    """

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, params, grads):
        self.t += 1

        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

            self.m[key] = self.beta1 * self.m[key] + \
                (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + \
                (1 - self.beta2) * (grads[key] ** 2)

            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
