import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

from svm import SMO_SVM


# =========================================================
# Helper: plot decision boundary
# =========================================================
def plot_boundary(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.project(grid)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=50, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')

    # highlight support vectors
    # highlight support vectors if model has them
    if hasattr(model, "alpha"):
        sv = model.alpha > 1e-5
        plt.scatter(
            X[sv, 0],
            X[sv, 1],
            s=120,
            facecolors='none',
            edgecolors='red',
            linewidths=2,
            label='Support Vectors'
        )

    plt.title(title)
    plt.legend()


# =========================================================
# Generate dataset (nonlinear)
# =========================================================
def load_data():
    X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
    y = np.where(y == 0, -1, 1)
    return X, y


# =========================================================
# Train SVM
# =========================================================
def train_rbf_svm(X, y):
    svm = SMO_SVM(C=1.0, gamma=2.0)
    svm.fit(X, y)
    return svm


# =========================================================
# Linear baseline (for comparison)
# =========================================================
def train_linear_baseline(X, y):
    # simple linear separator via least squares
    X_aug = np.c_[X, np.ones(len(X))]
    w = np.linalg.pinv(X_aug) @ y

    class LinearModel:
        def project(self, Xnew):
            Xnew_aug = np.c_[Xnew, np.ones(len(Xnew))]
            return Xnew_aug @ w

    return LinearModel()

# =========================================================
# Evaluation Utility
# =========================================================
def evaluate_model(model, X, y, name="Model"):
    preds = model.project(X)
    preds = np.sign(preds)

    accuracy = np.mean(preds == y)
    error = 1 - accuracy

    print(f"\n{name} Results")
    print("-" * (len(name) + 8))
    print(f"Accuracy           : {accuracy:.3f}")
    print(f"Misclassification  : {error:.3f}")

    if hasattr(model, "alpha"):
        num_sv = np.sum(model.alpha > 1e-5)
        print(f"Support Vectors    : {num_sv}")


# =========================================================
# Main
# =========================================================
def main():
    X, y = load_data()

    print("Training RBF SVM...")
    rbf_model = train_rbf_svm(X, y)

    print("Training linear baseline...")
    linear_model = train_linear_baseline(X, y)

    # ✅ Print errors
    evaluate_model(linear_model, X, y, "Linear Model")
    evaluate_model(rbf_model, X, y, "RBF SVM")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plot_boundary(linear_model, X, y, "Linear Boundary (Fails)")

    plt.subplot(1, 2, 2)
    plot_boundary(rbf_model, X, y, "RBF SVM Boundary")

    plt.show()


if __name__ == "__main__":
    main()