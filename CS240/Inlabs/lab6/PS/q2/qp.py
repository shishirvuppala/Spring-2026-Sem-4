import numpy as np
from cvxopt import matrix, solvers

# optional: silence solver output
solvers.options['show_progress'] = False


class QPSVM:
    """
    Support Vector Machine trained by solving the dual problem
    using a quadratic programming solver.

    Complete the TODO sections.
    """

    def __init__(self, C=1.0, gamma=0.5):
        self.C = C
        self.gamma = gamma

        # learned parameters
        self.alpha = None

        # training data
        self.X = None
        self.y = None

        # support vector mask
        self.support_mask = None

    # =========================================================
    # RBF Kernel
    # =========================================================
    def kernel(self, X, Z):
        """
        TODO:
        Compute the RBF kernel matrix between X and Z.

        K(x, z) = exp(-gamma ||x - z||^2)
        """
        pass

    # =========================================================
    # Train using quadratic programming
    # =========================================================
    def fit(self, X, y):
        """
        TODO:
        Solve the SVM dual problem using a QP solver.

        Steps:
        1. Store training data.
        2. Compute the kernel (Gram) matrix.
        3. Construct the QP matrices.
        4. Call the solver.
        5. Store alpha values.
        6. Identify support vectors.
        """
        self.X = X
        self.y = y.astype(float)

        n = X.shape[0]

        # ---- TODO: compute Gram matrix ----
        K = None

        # ---- TODO: construct QP matrices ----
        # P, q, G, h, A, b

        P = None
        q = None
        G = None
        h = None
        A = None
        b = None

        # ---- TODO: solve quadratic program ----
        solution = None

        # ---- TODO: extract alpha values ----
        self.alpha = None

        # ---- TODO: identify support vectors ----
        self.support_mask = None

        return self

    # =========================================================
    # Decision function
    # =========================================================
    def project(self, X):
        """
        TODO:
        Compute decision values for input points.

        f(x) = sum_i alpha_i y_i K(x_i, x)
        """
        pass

    def predict(self, X):
        """
        Return class predictions in {-1, +1}.
        """
        return np.sign(self.project(X))