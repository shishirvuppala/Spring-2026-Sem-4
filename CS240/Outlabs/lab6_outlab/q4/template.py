import numpy as np

# =====================================================
# Normalization Utility
# =====================================================

def normalize_dataset(X_train, X_val, X_test=None):
    """
    Task: Standardize features using training statistics only.
    Use (X - mean) / (std + 1e-8).
    """
    # TODO: Calculate mean and std of X_train
    # TODO: Standardize X_train and X_val
    # TODO: If X_test is provided, standardize it using X_train statistics
    pass


# =====================================================
# Weighted RBF Kernel
# =====================================================

def weighted_rbf_kernel(X1, X2, w):
    """
    Task 1: Implement weighted RBF kernel.
    """
    pass


# =====================================================
# Kernel Ridge Regression Solver
# =====================================================

def solve_alpha(K_train, y_train, lam):
    """
    Task 3: Solve (K + λI) alpha = y. 
    """
    pass


# =====================================================
# Kernel Derivative
# =====================================================

def compute_dK_dw(X1, X2, w):
    """
    Task: Compute ∂K/∂w_k for each dimension k. 
    """
    # TODO: Implement the derivative of the kernel matrix wrt each weight w_k
    pass


# =====================================================
# Validation Gradient Computation
# =====================================================

def compute_gradient(X_train, y_train, X_val, y_val, w, lam, gamma):
    """
    Task 4: Implement the gradient of validation loss wrt w. 
    """
    # TODO: Construct K_train, K_val, and A = K_train + lam*I
    # TODO: Compute dalpha/dw and dK_val/dw for each dimension
    # TODO: Combine using the formula above to find grad_w
    # TODO: Add L2 regularization contribution (gamma * w)
    pass


# =====================================================
# Gradient Descent Training
# =====================================================

def train_weighted_krr(X_train, y_train, X_val, y_val, lam=1e-1, gamma=1e-2, lr=0.1, max_iter=500):
    """
    Task 6: Perform gradient descent on w. 
    """    
    # TODO: Initialize weights w (e.g., np.ones(d))
    
    # TODO: Implement the Gradient Descent loop:
    # 1. Compute grad_w using compute_gradient
    # 2. Update weights
    # 3. Enforce positivity for w after the update 
    
    # TODO: Return the learned weights w
    pass


# =====================================================
# R² Utility
# =====================================================

def r2_score(y_true, y_pred):
    """
    Helper function for local testing.
    """
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res / ss_tot

# =====================================================
# Example Usage
# =====================================================

if __name__ == "__main__":

    np.random.seed(0)

    n_train = 50
    n_val   = 30
    d       = 5

    X_train = np.random.randn(n_train, d)
    X_val   = np.random.randn(n_val, d)

    true_w = np.array([2.0, 1.0, 0.5, 0.0, 3.0])

    def true_function(X):
        return np.sin(X @ true_w)

    y_train = true_function(X_train) + 0.1*np.random.randn(n_train)
    y_val   = true_function(X_val) + 0.1*np.random.randn(n_val)

    learned_w = train_weighted_krr(
        X_train, y_train,
        X_val, y_val,
        lam=1e-1,
        gamma=1e-2
    )

    print("Learned w:", learned_w)

    # Evaluate
    # Re-normalize for final evaluation in main block
    X_train_norm, X_val_norm = normalize_dataset(X_train, X_val)

    K_train = weighted_rbf_kernel(X_train_norm, X_train_norm, learned_w)
    K_val   = weighted_rbf_kernel(X_val_norm, X_train_norm, learned_w)

    alpha = solve_alpha(K_train, y_train, 1e-1)
    y_pred = K_val @ alpha

    print("Validation R2:", r2_score(y_val, y_pred))
