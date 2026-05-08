import numpy as np


class SMO_SVM:
    """
    Support Vector Machine trained using Sequential Minimal Optimization.

    Students must complete TODO sections.
    """

    def __init__(self, C=1.0, gamma=0.5, tol=1e-3, max_passes=5):
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.max_passes = max_passes

        # model parameters
        self.alpha = None

        # training data
        self.X = None
        self.y = None

        # kernel matrix
        self.K = None

        # error cache (stored for entire training)
        self.errors = None

    # =========================================================
    # RBF Kernel (vectorized)
    # =========================================================
    def kernel(self, X, Z):
        """
        TODO: Compute RBF kernel matrix.
        """
        X2 = np.sum(X**2, axis=1)[:, None]
        Z2 = np.sum(Z**2, axis=1)[None, :]
        dist = X2 + Z2 - 2 * X @ Z.T
        return np.exp(-self.gamma * dist)

    # =========================================================
    # Decision function
    # =========================================================
    def decision_function(self, i):
        return np.sum(self.alpha * self.y * self.K[:, i])

    # =========================================================
    # Select second multiplier (Platt heuristic)
    # =========================================================
    def select_j(self, i, Ei):
        """
        TODO:
        Choose j maximizing |Ei - Ej|
        Prefer non-bound multipliers.
        """

        non_bound = np.where((self.alpha > 0) & (self.alpha < self.C))[0]

        if len(non_bound) > 1:
            j = non_bound[np.argmax(np.abs(self.errors[non_bound] - Ei))]
            if j != i:
                return j

        # fallback random selection
        j = np.random.randint(len(self.alpha))
        while j == i:
            j = np.random.randint(len(self.alpha))
        return j

    # =========================================================
    # Training using SMO
    # =========================================================
    def fit(self, X, y):
        self.X = X
        self.y = y.astype(float)

        n = X.shape[0]

        self.alpha = np.zeros(n)
        self.K = self.kernel(X, X)

        # initialize error cache: f(x)=0 ⇒ E_i = -y_i
        self.errors = -self.y.copy()

        passes = 0

        while passes < self.max_passes:
            num_changed = 0

            for i in range(n):
                Ei = self.errors[i]

                # KKT violation check
                if (self.y[i]*Ei < -self.tol and self.alpha[i] < self.C) or \
                   (self.y[i]*Ei > self.tol and self.alpha[i] > 0):

                    j = self.select_j(i, Ei)
                    Ej = self.errors[j]

                    ai_old, aj_old = self.alpha[i], self.alpha[j]

                    # compute bounds
                    if self.y[i] != self.y[j]:
                        L = max(0, aj_old - ai_old)
                        H = min(self.C, self.C + aj_old - ai_old)
                    else:
                        L = max(0, ai_old + aj_old - self.C)
                        H = min(self.C, ai_old + aj_old)

                    if L == H:
                        continue

                    # curvature
                    eta = 2*self.K[i,j] - self.K[i,i] - self.K[j,j]
                    if eta >= 0:
                        continue

                    # update alpha_j
                    self.alpha[j] -= self.y[j]*(Ei - Ej)/eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)

                    if abs(self.alpha[j] - aj_old) < 1e-5:
                        continue

                    # update alpha_i
                    self.alpha[i] += self.y[i]*self.y[j]*(aj_old - self.alpha[j])

                    # vectorized error cache update
                    delta_i = self.alpha[i] - ai_old
                    delta_j = self.alpha[j] - aj_old

                    self.errors += (
                        self.y[i] * delta_i * self.K[i] +
                        self.y[j] * delta_j * self.K[j]
                    )

                    num_changed += 1

            if num_changed == 0:
                passes += 1
            else:
                passes = 0

        return self

    # =========================================================
    # Prediction
    # =========================================================
    def project(self, X):
        """
        TODO: compute decision values for new inputs
        """
        K = self.kernel(self.X, X)
        return (self.alpha * self.y) @ K

    def predict(self, X):
        return np.sign(self.project(X))