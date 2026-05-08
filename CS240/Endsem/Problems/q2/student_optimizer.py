import numpy as np

class Optimizer:
    """
    TODO:
    Complete or replace this optimizer class in any way you like.
    The training loop will call step(params, grads).
    
    Currently, this is a simple SGD optimizer with a fixed learning rate.
    """

    def __init__(self, lr=1e-3):
        self.lr = lr
        self.state = {}

    def step(self, params, grads):
        for key in params:
            params[key] -= self.lr * grads[key]
