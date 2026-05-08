"""
q4.py
=====
Task: Implement multi-class generative classifiers trained using

(1) Exact Normalization (Softmax)
(2) Importance Sampling (IS)
(3) Noise Contrastive Estimation (NCE)

Training is performed using mini-batch gradient descent
with on-the-fly sampling where required.

You are expected to fill in the TODOs only.
Do NOT change function signatures.
"""

import numpy as np


# ======================================================
# METRICS
# ======================================================

def accuracy(y_true, y_pred):
    """
    Compute overall classification accuracy.

    Args:
        y_true (np.ndarray): True labels, shape (N,)
        y_pred (np.ndarray): Predicted labels, shape (N,)

    Returns:
        float
    """
    # TODO
    pass


def precision(y_true, y_pred, cls):
    """
    Precision for class cls (one-vs-rest).

    Args:
        y_true (np.ndarray): shape (N,)
        y_pred (np.ndarray): shape (N,)
        cls (int)

    Returns:
        float
    """
    # TODO
    pass


def recall(y_true, y_pred, cls):
    """
    Recall for class cls (one-vs-rest).

    Args:
        y_true (np.ndarray): shape (N,)
        y_pred (np.ndarray): shape (N,)
        cls (int)

    Returns:
        float
    """
    # TODO
    pass


def f1_score(y_true, y_pred, cls):
    """
    F1 score for class cls.

    Args:
        y_true (np.ndarray): shape (N,)
        y_pred (np.ndarray): shape (N,)
        cls (int)

    Returns:
        float
    """
    # TODO
    pass


# ======================================================
# COMMON UTILITIES
# ======================================================

def sigmoid(z):
    """
    Sigmoid nonlinearity (used in NCE).

    Args:
        z (np.ndarray)

    Returns:
        np.ndarray: same shape as z, values in (0,1)
    """
    # TODO: use np.clip for numerical stability
    pass


def softmax(scores):
    """
    Numerically stable softmax (row-wise).

    Args:
        scores (np.ndarray): shape (B, K)

    Returns:
        np.ndarray: probabilities, shape (B, K)
    """
    # TODO
    pass


# ======================================================
# GENERIC SAMPLER INTERFACE
# ======================================================

class Sampler:
    """
    Generic sampler interface.

    Students are encouraged to subclass this class to implement
    any proposal or noise distribution they want.
    """

    def sample(self, num_samples):
        """
        Draw samples from q.

        Args:
            num_samples (int)

        Returns:
            np.ndarray
        """
        raise NotImplementedError

    def prob(self, samples):
        """
        Evaluate q(samples).

        Args:
            samples (np.ndarray)

        Returns:
            np.ndarray: densities, shape (num_samples,)
        """
        raise NotImplementedError


class GaussianSampler(Sampler):
    """
    Example sampler: multivariate Gaussian q(x).
    """

    def __init__(self, mean, cov):
        """
        Args:
            mean (np.ndarray): shape (D,)
            cov (np.ndarray): shape (D, D)
        """
        self.mean = mean
        self.cov = cov

    def sample(self, num_samples):
        """
        Returns:
            np.ndarray: samples, shape (num_samples, D)
        """
        # TODO
        pass

    def prob(self, samples):
        """
        Args:
            samples (np.ndarray): shape (num_samples, D)

        Returns:
            densities (np.ndarray): shape (num_samples,)
        """
        # TODO
        pass


class CategoricalSampler(Sampler):
    """
    Example sampler: categorical distribution q(y).
    """

    def __init__(self, probs):
        """
        Args:
            probs (np.ndarray): shape (K,)
        """
        self.probs = probs

    def sample(self, num_samples):
        """
        Returns:
            np.ndarray: sampled labels, shape (num_samples,)
        """
        # TODO
        pass

    def prob(self, samples):
        """
        Args:
            samples (np.ndarray): shape (num_samples,)

        Returns:
            densities (np.ndarray) : shape (num_samples,)
        """
        # TODO
        pass


# ======================================================
# SOFTMAX GENERATIVE CLASSIFIER (EXACT NORMALIZATION)
# ======================================================

class SoftmaxGenerativeClassifier:
    """
    Multi-class generative classifier trained using
    exact normalization (softmax).
    """

    def __init__(self, num_classes, lr=1e-2,
                 batch_size=64, max_epochs=50):
        """
        Args:
            num_classes (int): K
            lr (float)
            batch_size (int)
            max_epochs (int)
        """
        self.K = num_classes
        self.lr = lr
        self.batch_size = batch_size # also referred to as B in a lot of places
        self.max_epochs = max_epochs

        # Discriminative parameters
        self.W = None  # shape (K, D)
        self.b = None  # shape (K,)

        # Recovered generative parameters
        self.mu = None     # shape (K, D)
        self.Sigma = None # shape (D, D)
        self.pi = None    # shape (K,)

    def score(self, X):
        """
        Compute unnormalized scores p(x,y).

        Args:
            X (np.ndarray): shape (B, D)

        Returns:
            np.ndarray: shape (B, K)
        """
        # TODO
        pass

    def gradients(self, X, y):
        """
        Gradients of conditional log-likelihood.

        Args:
            X (np.ndarray): shape (B, D)
            y (np.ndarray): shape (B,)

        Returns:
            tuple W,b
                'W': np.ndarray, shape (K, D)
                'b': np.ndarray, shape (K,)
        """
        # TODO
        pass

    def fit(self, X, y):
        """
        Train using mini-batch gradient descent.

        Args:
            X (np.ndarray): shape (N, D)
            y (np.ndarray): shape (N,)
        """
        # TODO
        pass

    def predict_proba(self, X):
        """
        Compute p(y|x) using exact softmax.

        Args:
            X (np.ndarray): shape (N, D)

        Returns:
            np.ndarray: shape (N, K)
        """
        # TODO
        pass

    def predict(self, X):
        """
        Returns the actual prediction on the basis of predicted probabilities

        Args:
            X (np.ndarray): shape (N, D)

        Returns:
            np.ndarray: shape (N,)
        """
        # TODO 
        pass

    def recover_parameters(self):
        """
        Recover Gaussian parameters from trained W and b.


        Stores the parameters pi_k and Sigma into the class variables, note that self.mu
        will be populated by the main code by this point, so assume that self.mu contains the correct value
        """
        # TODO
        pass


# ======================================================
# IMPORTANCE SAMPLING CLASSIFIER
# ======================================================

class ImportanceSamplingClassifier:
    """
    Multi-class generative classifier trained using
    importance sampling to approximate normalization.
    """

    def __init__(self, num_classes, lr=1e-2,
                 batch_size=64, num_samples=10, max_epochs=50,
                 class_sampler=None):
        """
        Args:
            num_classes (int)
            lr (float)
            batch_size (int)
            num_samples (int): M
            max_epochs (int)
            class_sampler (Sampler): q(y)
        """
        self.K = num_classes
        self.lr = lr
        self.batch_size = batch_size
        self.M = num_samples
        self.max_epochs = max_epochs
        self.class_sampler = class_sampler

        self.W = None  # shape (K, D)
        self.b = None  # shape (K,)

    def score(self, X):
        """
        Args:
            X (np.ndarray): shape (B, D)

        Returns:
            np.ndarray: shape (B, K)
        """
        # TODO
        pass

    def estimate_normalizer(self, X):
        """
        Importance-sampled estimate of Z(x). Note that you should reuse the samples of y obtained for 1 batch
        Only draw new samples when you move to the next batch in training

        Args:
            X (np.ndarray): shape (B, D)

        Returns:
            np.ndarray: shape (B,)
        """
        # TODO
        pass

    def gradients(self, X, y):
        """
        Gradients of IS objective.

        Args:
            X (np.ndarray): shape (B, D)
            y (np.ndarray): shape (B,)

        Returns:
            grad_W,grad_b
                'grad_W': np.ndarray, shape (K, D)
                'grad_b': np.ndarray, shape (K,)
        """
        # TODO
        pass

    def fit(self, X, y):
        """
        Train using mini-batch gradient descent.

        Args:
            X (np.ndarray): shape (N, D)
            y (np.ndarray): shape (N,)
        """
        # TODO
        # Note that you can use a custom sampler by simply defining it here instead of using the class sampler
        # Please note that 1 epoch means 1 complete pass over the whole data, hence the number of steps in 1 epoch is roughly 
        # data size divided by the batch size
        # The best way to run mini batch is to shuffle the data and then makes passes of the batch size along it
        pass

    def predict_proba(self, X):
        """
        Args:
            X (np.ndarray): shape (N, D)

        Returns:
            np.ndarray: shape (N, K)
        """
        # TODO
        pass

    def predict(self, X):
        """
        Returns the actual prediction on the basis of predicted probabilities

        Args:
            X (np.ndarray): shape (N, D)

        Returns:
            np.ndarray: shape (N,)
        """
        # TODO 
        pass


# ======================================================
# NOISE CONTRASTIVE ESTIMATION CLASSIFIER
# ======================================================

class NCEClassifier:
    """
    Multi-class generative classifier trained using
    Noise Contrastive Estimation.
    """

    def __init__(self, num_classes, lr=1e-2,
                 batch_size=64, noise_ratio=5, max_epochs=500,
                 x_sampler=None, y_sampler=None):
        """
        Args:
            num_classes (int)
            lr (float)
            batch_size (int)
            noise_ratio (int): k
            max_epochs (int)
            x_sampler (Sampler): q(x)
            y_sampler (Sampler): q(y)
        """
        self.K = num_classes
        self.lr = lr
        self.batch_size = batch_size
        self.k = noise_ratio
        self.max_epochs = max_epochs
        self.x_sampler = x_sampler
        self.y_sampler = y_sampler

        self.W = None  # shape (K, D)
        self.b = None  # shape (K,)
        self.c = None  # scalar

    def score(self, X, y):
        """
        Compute p(x,y).

        Args:
            X (np.ndarray): shape (B, D)
            y (np.ndarray): shape (B,)

        Returns:
            np.ndarray: shape (B,)
        """
        # TODO
        pass

    def sample_noise(self, num_samples):
        """
        Sample noise pairs from q(x,y).

        Args:
            num_samples (int)

        Returns:
            tuple:
                X_noise (np.ndarray): shape (num_samples, D)
                y_noise (np.ndarray): shape (num_samples,)
        """
        # TODO
        pass

    def gradients(self, X, y):
        """
        Gradients of NCE objective.

        Args:
            X (np.ndarray): shape (B, D)
            y (np.ndarray): shape (B,)

        Returns:
            grad_W, grad_b, grad_c
                'grad_W': np.ndarray, shape (K, D)
                'grad_b': np.ndarray, shape (K,)
                'grad_c': float
        """
        # TODO
        pass

    def fit(self, X, y):
        """
        Train using mini-batch gradient descent.

        Args:
            X (np.ndarray): shape (N, D)
            y (np.ndarray): shape (N,)
        """
        # TODO
        # Note that you can use a custom sampler by simply defining it here instead of using the class sampler
        # Please note that 1 epoch means 1 complete pass over the whole data, hence the number of steps in 1 epoch is roughly 
        # The best way to run mini batch is to shuffle the data and then makes passes of the batch size along it
        pass

    def predict_proba(self, X):
        """
        Args:
            X (np.ndarray): shape (N, D)

        Returns:
            np.ndarray: shape (N, K)
        """
        # TODO
        pass

    def predict(self, X):
        """
        Returns the actual prediction on the basis of predicted probabilities

        Args:
            X (np.ndarray): shape (N, D)

        Returns:
            np.ndarray: shape (N,)
        """
        # TODO 
        pass
