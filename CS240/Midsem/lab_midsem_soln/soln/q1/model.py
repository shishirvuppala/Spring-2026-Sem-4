import numpy as np


# =========================================================
# Logistic Model with ℓ4 Regularization
# =========================================================

class LogisticModel:
    """
    Linear classifier with logistic loss and ℓ4 regularization.

        f(x) = w^T x + b

    The model standardizes features using statistics
    computed from the training set.
    """

    def __init__(self, d, lam=0.1, lr=0.01):
        self.w = np.zeros(d)
        self.b = 0.0
        self.lam = lam
        self.lr = lr

        # standardization parameters (learned from training data)
        self.mean = None
        self.std = None

    # =====================================================
    # Standardization utilities
    # =====================================================

    def fit_standardization(self, X):
        """Compute mean/std from training data."""
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8

    def transform(self, X):
        """Apply stored standardization."""
        return (X - self.mean) / self.std

    # =====================================================
    # Model functions
    # =====================================================

    def scores(self, X):
        """Compute raw model scores."""
        X = self.transform(X)
        return X @ self.w + self.b

    def predict(self, X):
        """Predict class labels in {+1, -1}."""
        s = self.scores(X)
        return np.where(s >= 0, 1, -1)

    # =====================================================
    # Loss & gradients
    # =====================================================

    def loss_batch(self, X, y):
        """
        Logistic loss + ℓ4 regularization
        (X must already be standardized)
        """
        z = y * (X @ self.w + self.b)
        # numerically stable logistic loss
        loss = np.mean(np.logaddexp(0, -z))
        # loss = np.mean(np.exp(-z)+1)


        reg = (self.lam / 4.0) * np.sum(self.w**4)

        return loss + reg

    def gradients(self, X, y):
        """
        Compute gradients for a mini-batch.
        (X must already be standardized)
        """
        scores = X @ self.w + self.b

        coeff = -y / (1.0 + np.exp(y * scores))

        grad_w = (coeff[:, None] * X).mean(axis=0)
        grad_b = coeff.mean()

        # ℓ4 regularization gradient
        grad_w += self.lam * (self.w ** 3)

        return grad_w, grad_b

    def step(self, X, y):
        grad_w, grad_b = self.gradients(X=X, y=y)
        self.w -= self.lr * grad_w
        self.b -= self.lr * grad_b


# =========================================================
# Training
# =========================================================

def train(model, X, y, batch_size=64, epochs=40, verbose=False):
    """
    Mini-batch gradient descent training.

    Standardization is fit on the training data
    and applied internally.
    """

    # compute and store normalization stats
    model.fit_standardization(X)

    # transform training data once
    X = model.transform(X)

    n = X.shape[0]

    # deterministic permutation for reproducibility
    rng = np.random.RandomState(0)

    for epoch in range(epochs):

        perm = rng.permutation(n)
        Xs = X[perm]
        ys = y[perm]

        for i in range(0, n, batch_size):
            xb = Xs[i:i+batch_size]
            yb = ys[i:i+batch_size]
            model.step(xb, yb)

        if verbose:
            print(f"Epoch {epoch+1} | loss = {model.loss_batch(X, y):.4f}")


# =========================================================
# Metrics
# =========================================================

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


# =========================================================
# Hyperparameter Selection
# =========================================================

def give_optimal_hyperparameter(Xtr, ytr, Xval, yval):
    """
    Select λ using validation accuracy.

    Training normalization is handled inside train().
    """

    lambdas = np.arange(0, 50, 1)

    best_lam = 0.0
    best_acc = -1

    for lam in lambdas:
        np.random.seed(0)  # ensure fair comparison

        model = LogisticModel(Xtr.shape[1], lam=lam)
        train(model, Xtr, ytr)

        y_pred = model.predict(Xval)
        acc = accuracy(yval, y_pred)

        # tolerance prevents flipping due to tiny differences
        if acc > best_acc + 1e-4:
            best_acc = acc
            best_lam = lam

    return best_lam