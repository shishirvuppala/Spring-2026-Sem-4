"""
main.py  —  do not edit

Run all:           python main.py
Run specific:      python main.py --linear --rbf
Available flags:   --linear  --poly  --rbf  --normrbf  --learned
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

from model import SVM
import svm

DATA_PATH = "data.csv"

KERNEL_FLAGS = {
    "--linear"  : "Linear",
    "--poly"    : "Polynomial",
    "--rbf"     : "RBF",
    "--normrbf" : "Normalized RBF",
    "--learned" : "Learned",
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    data = np.genfromtxt(DATA_PATH, delimiter=",", skip_header=1)
    X, y = data[:, :-1], data[:, -1].astype(int)
    return X, y


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def train_test_split(X, y, test_size=0.4, random_state=0):
    rng  = np.random.default_rng(random_state)
    idx  = rng.permutation(len(y))
    n_te = int(len(y) * test_size)
    return (X[idx[n_te:]], X[idx[:n_te]],
            y[idx[n_te:]], y[idx[:n_te]])


# ============================================================
# ACCURACY
# ============================================================

def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


# ============================================================
# BUILD MODELS
# ============================================================

def build_models(X_train, X_test, y_train, selected):
    y     = y_train
    specs = {
        "Linear"       : lambda: svm.linear_kernel(X_train, X_test),
        "Polynomial"   : lambda: svm.polynomial_kernel(X_train, X_test),
        "RBF"          : lambda: svm.rbf_kernel(X_train, X_test),
        "Normalized RBF": lambda: svm.normalized_rbf_kernel(X_train, X_test),
        "Learned"      : lambda: svm.learnable_kernel(X_train, X_test, y),
    }
    models = {}
    for name in selected:
        result = specs[name]()
        if len(result) == 3:
            K_tr, K_te, C = result
        else:
            K_tr, K_te = result
            C = 1.0
        models[name] = (SVM(C=C).fit(K_tr, y), K_tr, K_te)
    return models


# ============================================================
# PLOT DECISION BOUNDARIES
# ============================================================

def plot_all(models, X_train, y_train):
    names = list(models.keys())
    ncols = min(3, len(names))
    nrows = (len(names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = np.array(axes).flatten()

    X2 = X_train[:, :2]
    xx, yy = np.meshgrid(
        np.linspace(X2[:,0].min()-1, X2[:,0].max()+1, 150),
        np.linspace(X2[:,1].min()-1, X2[:,1].max()+1, 150))
    grid        = np.zeros((len(xx.ravel()), X_train.shape[1]))
    grid[:, :2] = np.c_[xx.ravel(), yy.ravel()]
    y_plot      = np.where(y_train == -1, 0, 1)

    kernel_fns = {
        "Linear"        : lambda g: svm.linear_kernel(g, X_train)[0],
        "Polynomial"    : lambda g: svm.polynomial_kernel(g, X_train)[0],
        "RBF"           : lambda g: svm.rbf_kernel(g, X_train)[0],
        "Normalized RBF": lambda g: svm.normalized_rbf_kernel(g, X_train)[0],
        "Learned"       : lambda g: svm.learnable_kernel(g, X_train, y_train)[0],
    }

    for ax, name in zip(axes, names):
        clf, _, _ = models[name]
        pred = clf.predict(kernel_fns[name](grid))
        ax.contourf(xx, yy, pred.reshape(xx.shape), alpha=0.3, cmap="bwr")
        ax.scatter(X2[:,0], X2[:,1], c=y_plot, cmap="bwr", s=15)
        ax.set_title(name)
    for ax in axes[len(names):]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    X, y = load_data()

    if "--plot" in sys.argv:
        svm.plot(X, y)
        return

    # which kernels to run
    flags    = [a for a in sys.argv[1:] if a in KERNEL_FLAGS]
    selected = [KERNEL_FLAGS[f] for f in flags] if flags else list(KERNEL_FLAGS.values())

    # handle both {0,1} and {-1,+1} label formats
    unique_y = np.unique(y)
    if set(unique_y).issubset({0, 1}):
        y_svm = np.where(y == 0, -1, 1)
    else:
        y_svm = y.copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_svm, test_size=0.4, random_state=0)

    models = build_models(X_train, X_test, y_train, selected)

    print("\nRESULTS\n")
    print(f"{'Model':<20}  {'Train':>7}  {'Test':>7}")
    print("-" * 38)
    for name, (clf, K_tr, K_te) in models.items():
        tr = accuracy(y_train, clf.predict(K_tr))
        te = accuracy(y_test,  clf.predict(K_te))
        print(f"{name:<20}  {tr:>7.3f}  {te:>7.3f}")



if __name__ == "__main__":
    main()