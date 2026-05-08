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
    return np.vander(x, degree + 1, increasing=True)

def fit_ols(Phi, y):
    """
    Phi : (N, p+1)
    y   : (N,)

    Returns w : (p+1,)
    """
    w = np.linalg.pinv(Phi.T @ Phi) @ Phi.T @ y
    return w
    

def predict(x, degree, w):
    """
    x : (N, )
    degree: int
    w : (degree+1,)

    Returns y_hat : (N,)
    """
    Phi = poly_features(x, degree)
    y_hat = Phi @ w
    return y_hat

# ============================================================
# k-fold cross-validation
# ============================================================

def mse(y, y_hat):
    return np.mean((y - y_hat) ** 2)


def k_fold_cv(x, y, degree, k=5):
    """
    x: (N,)
    y: (N,)
    degree: int
    k: int

    Returns avg_val_mse : float
    """
    # 1. shuffle data
    N = len(x)
    indices = np.arange(N)
    np.random.shuffle(indices)
    
    # 2. split into k folds
    folds = np.array_split(indices, k)

    # 3. train on k-1, validate on 1
    val_mse_list = []
    for i in range(k):
        val_indices = folds[i]
        x_val = x[val_indices]
        y_val = y[val_indices]
        
        train_indices = np.concatenate([folds[j] for j in range(k) if j != i])
        x_train = x[train_indices]
        y_train = y[train_indices]
        
        Phi_train = poly_features(x_train, degree)
        w = fit_ols(Phi_train, y_train)
        
        y_val_pred = predict(x_val, degree, w)
        fold_mse = mse(y_val, y_val_pred)
        val_mse_list.append(fold_mse)

    # 4. return average validation MSE
    return np.mean(val_mse_list)

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
    
    train_mse = []
    cv_mse = []

    for d in range(1, D_max + 1):
        # Compute k-fold CV MSE
        cv_mse.append(k_fold_cv(x, y, d, k))

        # Compute training MSE
        Phi = poly_features(x, d)
        w = fit_ols(Phi, y)
        
        y_train_pred = predict(x, d, w)
        train_mse.append(mse(y, y_train_pred))
        
    return train_mse, cv_mse
        

def select_degree(cv_mse):
    """
    cv_mse : list
    Returns best_degree : int
    """
    return np.argmin(cv_mse) + 1


def fit_final_model(x, y, degree):
    """
    x : (N,) array
    y : (N,) array
    degree : int

    Returns w : (degree+1,) array
    """
    Phi = poly_features(x, degree)
    w = fit_ols(Phi, y)
    return w

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
    data = np.loadtxt('q2_train.csv', delimiter=',', skiprows=1)
    x_train = data[:, 0]
    y_train = data[:, 1]
    return x_train, y_train

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