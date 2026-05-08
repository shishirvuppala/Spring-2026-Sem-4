import numpy as np

# =====================================================
# Normalization Utility
# =====================================================

def normalize_dataset(X_train, X_val, X_test=None):
    """
    Standardize features using training statistics only.
    """
    mean = np.mean(X_train, axis=0)
    std  = np.std(X_train, axis=0) + 1e-8

    X_train_norm = (X_train - mean) / std
    X_val_norm   = (X_val - mean) / std

    if X_test is not None:
        X_test_norm = (X_test - mean) / std
        return X_train_norm, X_val_norm, X_test_norm

    return X_train_norm, X_val_norm


# =====================================================
# Weighted RBF Kernel
# =====================================================

def weighted_rbf_kernel(X1, X2, w):
    """
    Compute weighted RBF kernel: K(x,z) = exp( - sum_k w_k (x_k - z_k)^2 )
    """
    # TODO 1: Implement weighted RBF kernel.
    diff = X1[:, None, :] - X2[None, :, :]
    dist = np.sum(w * (diff**2), axis=2)
    return np.exp(-dist)


# =====================================================
# Kernel Ridge Regression Solver
# =====================================================

def solve_alpha(K_train, y_train, lam):
    """
    Solve (K + λI) alpha = y WITHOUT using matrix inverse.
    """
    # TODO 2: Use np.linalg.solve (NO inverse allowed)
    n = K_train.shape[0]
    A = K_train + lam * np.eye(n)
    return np.linalg.solve(A, y_train)


# =====================================================
# Kernel Derivative
# =====================================================

def compute_dK_dw(X1, X2, w):
    """
    Compute ∂K/∂w_k for each dimension k.
    """
    # TODO 3: Implement derivative: ∂K_ij / ∂w_k = - (x_ik - x_jk)^2 * K_ij
    K = weighted_rbf_kernel(X1, X2, w)
    diff_sq = (X1[:, None, :] - X2[None, :, :])**2
    
    dK_dw = []
    for k in range(len(w)):
        dK_dw.append(-diff_sq[:, :, k] * K)
    return dK_dw


# =====================================================
# Validation Gradient Computation
# =====================================================

def compute_gradient(X_train, y_train, X_val, y_val, w, lam, gamma):
    """
    Compute gradient of validation loss wrt w.
    """
    K_train = weighted_rbf_kernel(X_train, X_train, w)
    K_val   = weighted_rbf_kernel(X_val, X_train, w)
    alpha   = solve_alpha(K_train, y_train, lam)

    e = K_val @ alpha - y_val
    A = K_train + lam * np.eye(len(X_train))

    dK_train_dw = compute_dK_dw(X_train, X_train, w)
    dK_val_dw   = compute_dK_dw(X_val, X_train, w)

    d = len(w)
    grad = np.zeros(d)

    # TODO 4: Implement gradient formula
    for k in range(d):
        term1 = dK_val_dw[k] @ alpha
        # term2 is the influence of w through alpha: K_val * (d alpha / dw)
        term2 = K_val @ np.linalg.solve(A, dK_train_dw[k] @ alpha)
        grad[k] = np.dot(e, term1 - term2)

    # TODO 4b: Add L2 regularization contribution
    grad += gamma * w
    return grad


# =====================================================
# Gradient Descent Training
# =====================================================

def train_weighted_krr(X_train, y_train, X_val, y_val, lam=1e-1, gamma=1e-3, lr=0.5, max_iter=800, eps=1e-6):
    """
    Train weighted RBF kernel parameters w using gradient descent.
    """
    # TODO 0: Normalize datasets before training.
    X_tr, X_v = normalize_dataset(X_train, X_val)

    n, d = X_tr.shape
    # Initialize weights to 1.0 for better stability
    w = np.ones(d)

    for it in range(max_iter):
        grad = compute_gradient(X_tr, y_train, X_v, y_val, w, lam, gamma)
        
        # Stability fix: simple learning rate decay
        curr_lr = lr / (1 + 0.05 * it)
        
        w = w - curr_lr * grad

        # TODO 5: Enforce positivity constraint
        w = np.maximum(1e-5, w)

        if np.linalg.norm(grad) < 1e-4:
            break

        if it % 50 == 0:
            print(f"Iteration {it}, grad norm = {np.linalg.norm(grad):.6f}")

    return w

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