import numpy as np
from cvxopt import matrix, solvers

# suppress solver progress output
solvers.options['show_progress'] = False


class QPSVM:
    """
    SVM trained via dual quadratic programming.
    """

    def __init__(self, C=1.0, gamma=0.5):
        self.C = C
        self.gamma = gamma

        self.alpha = None
        self.X = None
        self.y = None
        self.support_mask = None

    # --------------------------------------------------------
    # RBF kernel
    # --------------------------------------------------------
    def kernel(self, X, Z):
        X2 = np.sum(X**2, axis=1)[:, None]
        Z2 = np.sum(Z**2, axis=1)[None, :]
        dist = X2 + Z2 - 2 * X @ Z.T
        return np.exp(-self.gamma * dist)

    # --------------------------------------------------------
    # Train using QP
    # --------------------------------------------------------
    def fit(self, X, y):
        self.X = X
        self.y = y.astype(float)

        n = X.shape[0]
        K = self.kernel(X, X)

        # QP matrices
        P = matrix(np.outer(self.y, self.y) * K)
        q = matrix(-np.ones(n))

        G = matrix(np.vstack((-np.eye(n), np.eye(n))))
        h = matrix(np.hstack((np.zeros(n), np.ones(n) * self.C)))

        A = matrix(self.y.reshape(1, -1))
        b = matrix(0.0)

        solution = solvers.qp(P, q, G, h, A, b)
        self.alpha = np.array(solution['x']).flatten()

        self.support_mask = self.alpha > 1e-5
        return self

    # --------------------------------------------------------
    # Decision function
    # --------------------------------------------------------
    def project(self, X):
        K = self.kernel(self.X, X)
        return (self.alpha * self.y) @ K

    def predict(self, X):
        return np.sign(self.project(X))
