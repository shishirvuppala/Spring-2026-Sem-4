import numpy as np

from targets import target_function
from student_model import FeedForwardNN
from student_optimizer import Optimizer


class StudentClass:
    """
    Main class used by the evaluator.

    You may modify the internals of this class as needed.
    However, do not change the method signatures of:
        - build_model(self, seed=0)
        - build_optimizer(self)
        - fit_model(self, seed=0)
    """

    def __init__(self):
        # Starter hyperparameters.
        # You are free to tune these, but keep training time in mind.
        self.TRAIN_SAMPLES = 2048
        self.EPOCHS = 3000
        self.BATCH_SIZE = 128
        self.LEARNING_RATE = 1e-3

    def build_model(self, seed=0):
        """
        Construct and return the model.

        The input and output dimensions must remain 1.
        You may change the architecture and any model hyperparameters.
        """
        return FeedForwardNN(hidden_dim=8, seed=seed)

    def build_optimizer(self):
        """
        Construct and return the optimizer.

        You may replace the starter optimizer with any optimizer
        of your choice, and you may tune its hyperparameters.
        """
        return Optimizer(lr=self.LEARNING_RATE)

    def make_training_data(self, seed=0):
        """
        Generate training data from the target function on [0, 1].

        The starter version uses simple uniform random sampling.
        You may change this if you believe a better sampling strategy helps.
        """
        rng = np.random.default_rng(seed)
        x = rng.uniform(0.0, 1.0, size=(self.TRAIN_SAMPLES, 1))
        y = target_function(x)
        return x, y

    def iterate_minibatches(self, x, y, batch_size, rng):
        """
        Yield shuffled mini-batches from the dataset.
        """
        idx = np.arange(x.shape[0])
        rng.shuffle(idx)

        for start in range(0, len(idx), batch_size):
            batch_idx = idx[start:start + batch_size]
            yield x[batch_idx], y[batch_idx]

    def fit_model(self, seed=0):
        """
        Train the model and return the trained model instance.

        The evaluator will call this method directly.
        You may modify the training loop, but make sure that:
        1. a model is created using build_model(...)
        2. an optimizer is created using build_optimizer(...)
        3. the trained model is returned at the end
        """
        rng = np.random.default_rng(seed)
        x_train, y_train = self.make_training_data(seed=seed)

        model = self.build_model(seed=seed)
        optimizer = self.build_optimizer()

        for _ in range(self.EPOCHS):
            for xb, yb in self.iterate_minibatches(
                x_train, y_train, self.BATCH_SIZE, rng
            ):
                y_pred, cache = model.forward(xb)
                grads = model.backward(y_pred, yb, cache)
                optimizer.step(model.params, grads)

        return model
