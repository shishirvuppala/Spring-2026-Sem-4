import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from qp import QPSVM


# --------------------------------------------------------
# Generate nonlinear dataset
# --------------------------------------------------------
def load_data():
    X, y = make_moons(n_samples=200, noise=0.15, random_state=42)
    y = 2*y - 1  # convert to {-1, +1}
    return X, y


# --------------------------------------------------------
# Plot decision boundary
# --------------------------------------------------------
def plot_boundary(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
    y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.project(grid)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z > 0, alpha=0.2)
    plt.contour(xx, yy, Z, levels=[0])

    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')

    # highlight support vectors
    sv = model.support_mask
    plt.scatter(X[sv, 0], X[sv, 1],
                s=120, facecolors='none', edgecolors='k')

    plt.title(title)


# --------------------------------------------------------
# Evaluation utility
# --------------------------------------------------------
def evaluate(model, X, y):
    preds = model.predict(X)
    acc = np.mean(preds == y)
    print(f"Accuracy: {acc:.3f}")
    print(f"Support vectors: {np.sum(model.support_mask)}")


# --------------------------------------------------------
# Main
# --------------------------------------------------------
def main():
    X, y = load_data()

    print("Training QP SVM...")
    model = QPSVM(C=1.0, gamma=2.0)
    model.fit(X, y)

    evaluate(model, X, y)

    plt.figure(figsize=(6,5))
    plot_boundary(model, X, y, "QP SVM (RBF Kernel)")
    plt.show()


if __name__ == "__main__":
    main()