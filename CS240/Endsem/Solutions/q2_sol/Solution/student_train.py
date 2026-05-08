from random import seed

import numpy as np
import matplotlib.pyplot as plt

from targets import target_function
from student_model import FeedForwardNN
from student_optimizer import SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam


class StudentClass:
    def __init__(self):
        self.TRAIN_SAMPLES = 2048
        self.EPOCHS = 3000
        self.BATCH_SIZE = 128
        self.LEARNING_RATE = 1e-3
        pass

    def build_model(self, seed=0):
        return FeedForwardNN(layer_dims=[1, 64, 64, 64, 1], seed=seed)

    def build_optimizer(self, name="adam"):
        if name == "sgd":
            return SGD(lr=self.LEARNING_RATE)
        elif name == "momentum":
            return Momentum(lr=self.LEARNING_RATE, beta=0.9)
        elif name == "nesterov":
            return Nesterov(lr=self.LEARNING_RATE, beta=0.9)
        elif name == "adagrad":
            return AdaGrad(lr=self.LEARNING_RATE, eps=1e-8)
        elif name == "rmsprop":
            return RMSProp(lr=self.LEARNING_RATE, beta=0.9, eps=1e-8)
        elif name == "adam":
            return Adam(lr=self.LEARNING_RATE, beta1=0.9, beta2=0.999, eps=1e-8)
        else:
            raise ValueError("Unknown optimizer name")

    def make_training_data(self, seed=0):
        rng = np.random.default_rng(seed)

        n_grid = self.TRAIN_SAMPLES // 2
        n_rand = self.TRAIN_SAMPLES - n_grid
        x_grid = np.linspace(0.0, 1.0, n_grid, dtype=np.float64).reshape(-1, 1)
        x_rand = rng.uniform(0.0, 1.0, size=(n_rand, 1))
        x = np.vstack([x_grid, x_rand])
        y = target_function(x)
        return x, y

    def iterate_minibatches(self, x, y, batch_size, rng):
        idx = np.arange(x.shape[0])
        rng.shuffle(idx)
        for start in range(0, len(idx), batch_size):
            batch_idx = idx[start:start + batch_size]
            yield x[batch_idx], y[batch_idx]

    def fit_model(self, seed=0):
        rng = np.random.default_rng(seed)
        x_train, y_train = self.make_training_data(seed=seed)

        model = self.build_model(seed=seed)
        optimizer = self.build_optimizer()

        for _ in range(self.EPOCHS):
            for xb, yb in self.iterate_minibatches(x_train, y_train, self.BATCH_SIZE, rng):
                y_pred, cache = model.forward(xb)
                grads = model.backward(y_pred, yb, cache)
                optimizer.step(model.params, grads)

        return model
