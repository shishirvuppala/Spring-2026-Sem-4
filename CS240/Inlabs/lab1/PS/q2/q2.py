# do not modify the imports below
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Polynomial regression implementations
# ============================================================

def poly_features(x, degree):
    """
    x : (N,)
    degree: int

    Returns Phi : (N, degree+1)
        Phi[i] = [1, x_i, x_i^2, ..., x_i^degree]
    """
    # TODO
    raise NotImplementedError

def fit_ols(Phi, y):
    """
    Phi : (N, p+1)
    y   : (N,)

    Returns w : (p+1,)
    """
    # TODO
    raise NotImplementedError


def predict(x, degree, w):
    """
    x : (N, )
    degree: int
    w : (degree+1,)

    Returns y_hat : (N,)
    """
    # TODO
    raise NotImplementedError

# ============================================================
# k-fold cross-validation
# ============================================================

def mse(y, y_hat):
    # TODO
    raise NotImplementedError


def k_fold_cv(x, y, degree, k=5):
    """
    x: (N,)
    y: (N,)
    degree: int
    k: int

    Returns avg_val_mse : float
    """
    # TODO:
    # 1. shuffle data
    # 2. split into k folds
    # 3. train on k-1, validate on 1
    # 4. return average validation MSE
    raise NotImplementedError


def evaluate_degrees(x, y, D_max, k=5):
    """
    x : (N,) array
    y : (N,) array
    D_max : int
    k : int

    Returns
        train_mse : list of length D_max
        cv_mse    : list of length D_max
    """
    # TODO:
    # For each d in 1..D_max:
    #   - construct Phi
    #   - fit OLS
    #   - compute training MSE
    #   - compute k-fold CV MSE
    # Store results in lists and return them
    raise NotImplementedError


def select_degree(cv_mse):
    """
    cv_mse : list
    Returns best_degree : int
    """
    # TODO
    raise NotImplementedError


def fit_final_model(x, y, degree):
    """
    x : (N,) array
    y : (N,) array
    degree : int

    Returns w : (degree+1,) array
    """
    # TODO
    raise NotImplementedError

def plot_errors(train_mse, cv_mse):
    D = len(train_mse)
    plt.plot(range(1, D + 1), train_mse, label="Training MSE")
    plt.plot(range(1, D + 1), cv_mse, label="CV MSE")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

# ============================================================
# Data loading
# ============================================================

def load_data():
    """
    Load the train dataset

    Returns
    -------
    X_train, y_train (q2_train.csv)
    """
    # TODO
    raise NotImplementedError

# ============================================================
# Main experiment (DO NOT MODIFY, AUTOGRADER TESTS WILL RUN SOME OTHER MAIN)
# ============================================================
if __name__ == "__main__":
    np.random.seed(1234)

    x_train, y_train = load_data()
    D_max = 10
    k = 5

    train_mse, cv_mse = evaluate_degrees(x_train, y_train, D_max, k)

    best_d = select_degree(cv_mse)
    w = fit_final_model(x_train, y_train, best_d)

    print("Optimal degree:", best_d)
    print("Weights:", w)

    plot_errors(train_mse, cv_mse)