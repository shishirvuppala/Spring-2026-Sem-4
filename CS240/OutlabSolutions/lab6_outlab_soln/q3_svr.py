import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

silencers = solvers.options.copy()
solvers.options['show_progress'] = False

# ──────────────────────────────────────────────
# 1. Dataset Generation
# ──────────────────────────────────────────────

np.random.seed(42)
N = 100
X_all = np.sort(np.random.uniform(-3, 3, N))
y_all = np.sin(X_all) + np.random.normal(0, 0.1, N)

split = int(0.8 * N)
X_train, y_train = X_all[:split], y_all[:split]
X_test,  y_test  = X_all[split:], y_all[split:]


# ──────────────────────────────────────────────
# 2. Kernel Functions
# ──────────────────────────────────────────────

def linear_kernel(X1, X2):
    """K(x,z) = x^T z"""
    return np.outer(X1, X2)


def rbf_kernel(X1, X2, gamma=1.0):
    """K(x,z) = exp(-gamma * ||x-z||^2)"""
    X1 = X1.reshape(-1, 1)
    X2 = X2.reshape(-1, 1)
    diffs = X1 - X2.T
    return np.exp(-gamma * diffs**2)


# ──────────────────────────────────────────────
# 3. SVR via Quadratic Programming (cvxopt)
# ──────────────────────────────────────────────

def svr_qp(X_train, y_train, kernel_fn, C=1.0, epsilon=0.1):
    """
    Solves the epsilon-SVR dual using cvxopt.

    Dual variables: a = [alpha; alpha*], length 2n.
    Objective (to minimise):
        0.5 * a^T Q a + p^T a
    where Q_ij involves (alpha_i - alpha*_i)(alpha_j - alpha*_j) K(xi,xj).

    We encode the 2n vector as [alpha_1,...,alpha_n, alpha*_1,...,alpha*_n].
    """
    n = len(X_train)
    K = kernel_fn(X_train, X_train)  # (n, n)

    # Build 2n x 2n Q matrix
    # Q = [[K, -K], [-K, K]]
    Q = np.block([[K, -K], [-K, K]])
    Q = matrix(Q.astype(float))

    # Linear part: epsilon * 1 - y stacked with epsilon * 1 + y
    p = np.concatenate([epsilon * np.ones(n) - y_train,
                        epsilon * np.ones(n) + y_train])
    p = matrix(p.astype(float))

    # Box constraints: 0 <= a_i <= C  (2n variables)
    G_top    = -np.eye(2 * n)          # -a <= 0
    G_bottom =  np.eye(2 * n)          #  a <= C
    G = matrix(np.vstack([G_top, G_bottom]).astype(float))
    h = matrix(np.concatenate([np.zeros(2 * n),
                                C * np.ones(2 * n)]).astype(float))

    # Equality constraint: sum(alpha) - sum(alpha*) = 0
    A = matrix(np.concatenate([np.ones(n), -np.ones(n)]).reshape(1, -1).astype(float))
    b = matrix(np.zeros(1).astype(float))

    sol = solvers.qp(Q, p, G, h, A, b)
    a = np.array(sol['x']).flatten()

    alpha     = a[:n]
    alpha_star = a[n:]
    coef      = alpha - alpha_star   # (alpha_i - alpha*_i)

    # Compute bias b using support vectors (strictly inside bounds)
    tol = 1e-4
    sv_mask = (alpha > tol) & (alpha < C - tol)
    sv_star_mask = (alpha_star > tol) & (alpha_star < C - tol)

    biases = []
    for i in np.where(sv_mask)[0]:
        pred = np.sum(coef * K[i])
        biases.append(y_train[i] - epsilon - pred)
    for i in np.where(sv_star_mask)[0]:
        pred = np.sum(coef * K[i])
        biases.append(y_train[i] + epsilon - pred)

    bias = np.mean(biases) if biases else 0.0

    # Support vector indices (either alpha or alpha* is active)
    sv_indices = np.where((alpha > tol) | (alpha_star > tol))[0]

    return coef, bias, sv_indices, K


def svr_predict(X_new, X_train, coef, bias, kernel_fn):
    K_new = kernel_fn(X_new, X_train)   # (m, n)
    return K_new @ coef + bias


# ──────────────────────────────────────────────
# 4. Plotting Helper
# ──────────────────────────────────────────────

def plot_svr(ax, X_train, y_train, X_plot, y_pred, epsilon,
             sv_indices, title, kernel_label):
    ax.scatter(X_train, y_train, s=20, color='steelblue',
               alpha=0.5, label='Training points', zorder=3)
    ax.scatter(X_train[sv_indices], y_train[sv_indices],
               s=80, facecolors='none', edgecolors='red',
               linewidths=1.5, label=f'Support vectors ({len(sv_indices)})', zorder=4)
    ax.plot(X_plot, np.sin(X_plot), 'k--', lw=1.5, label='True: sin(x)')
    ax.plot(X_plot, y_pred, 'orange', lw=2, label='SVR prediction')
    ax.fill_between(X_plot, y_pred - epsilon, y_pred + epsilon,
                    alpha=0.2, color='orange', label=f'ε-tube (ε={epsilon})')
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, loc='upper right')
    ax.set_xlabel('x')
    ax.set_ylabel('y')


# ──────────────────────────────────────────────
# 5. Experiment: Effect of ε and C (RBF kernel, fixed gamma)
# ──────────────────────────────────────────────

X_plot = np.linspace(-3, 3, 300)
gamma_default = 1.0

epsilons = [0.01, 0.1, 0.5]
Cs       = [0.1, 1, 100]

# ── 5a. Vary epsilon (C=1, gamma=1) ──────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('SVR (RBF, gamma=1, C=1): Effect of ε', fontsize=12)

for ax, eps in zip(axes, epsilons):
    kfn = lambda a, b, g=gamma_default: rbf_kernel(a, b, g)
    coef, bias, sv_idx, _ = svr_qp(X_train, y_train, kfn, C=1.0, epsilon=eps)
    y_pred = svr_predict(X_plot, X_train, coef, bias, kfn)
    plot_svr(ax, X_train, y_train, X_plot, y_pred, eps, sv_idx,
             f'ε = {eps}', 'RBF')

plt.tight_layout()
plt.savefig('svr_vary_epsilon.png', dpi=150)
plt.close()
print("Saved: svr_vary_epsilon.png")

# ── 5b. Vary C (epsilon=0.1, gamma=1) ────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('SVR (RBF, gamma=1, ε=0.1): Effect of C', fontsize=12)

for ax, C in zip(axes, Cs):
    kfn = lambda a, b, g=gamma_default: rbf_kernel(a, b, g)
    coef, bias, sv_idx, _ = svr_qp(X_train, y_train, kfn, C=C, epsilon=0.1)
    y_pred = svr_predict(X_plot, X_train, coef, bias, kfn)
    plot_svr(ax, X_train, y_train, X_plot, y_pred, 0.1, sv_idx,
             f'C = {C}', 'RBF')

plt.tight_layout()
plt.savefig('svr_vary_C.png', dpi=150)
plt.close()
print("Saved: svr_vary_C.png")

# ── 5c. Vary gamma (C=1, epsilon=0.1) ────────
gammas = [0.1, 1.0, 10.0]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('SVR (RBF, C=1, ε=0.1): Effect of gamma', fontsize=12)

for ax, g in zip(axes, gammas):
    kfn = lambda a, b, gamma=g: rbf_kernel(a, b, gamma)
    coef, bias, sv_idx, _ = svr_qp(X_train, y_train, kfn, C=1.0, epsilon=0.1)
    y_pred = svr_predict(X_plot, X_train, coef, bias, kfn)
    plot_svr(ax, X_train, y_train, X_plot, y_pred, 0.1, sv_idx,
             f'gamma = {g}', 'RBF')

plt.tight_layout()
plt.savefig('svr_vary_gamma.png', dpi=150)
plt.close()
print("Saved: svr_vary_gamma.png")

# ── 5d. Linear vs RBF comparison ─────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('SVR: Linear vs RBF Kernel (C=1, ε=0.1)', fontsize=12)

# Linear
coef_lin, bias_lin, sv_lin, _ = svr_qp(
    X_train, y_train, linear_kernel, C=1.0, epsilon=0.1)
y_pred_lin = svr_predict(X_plot, X_train, coef_lin, bias_lin, linear_kernel)
plot_svr(axes[0], X_train, y_train, X_plot, y_pred_lin, 0.1, sv_lin,
         'Linear Kernel', 'Linear')

# RBF
kfn_rbf = lambda a, b: rbf_kernel(a, b, gamma_default)
coef_rbf, bias_rbf, sv_rbf, _ = svr_qp(
    X_train, y_train, kfn_rbf, C=1.0, epsilon=0.1)
y_pred_rbf = svr_predict(X_plot, X_train, coef_rbf, bias_rbf, kfn_rbf)
plot_svr(axes[1], X_train, y_train, X_plot, y_pred_rbf, 0.1, sv_rbf,
         'RBF Kernel (gamma=1)', 'RBF')

plt.tight_layout()
plt.savefig('svr_linear_vs_rbf.png', dpi=150)
plt.close()
print("Saved: svr_linear_vs_rbf.png")


# ──────────────────────────────────────────────
# 6. Test MSE Summary
# ──────────────────────────────────────────────

print("\n── Test MSE Summary ──")
kfn_rbf = lambda a, b: rbf_kernel(a, b, 1.0)

for eps in epsilons:
    coef, bias, sv_idx, _ = svr_qp(X_train, y_train, kfn_rbf, C=1.0, epsilon=eps)
    mse = np.mean((svr_predict(X_test, X_train, coef, bias, kfn_rbf) - y_test)**2)
    print(f"  RBF | C=1   | ε={eps:<4} | gamma=1   | #SV={len(sv_idx):3d} | Test MSE={mse:.4f}")

for C in Cs:
    coef, bias, sv_idx, _ = svr_qp(X_train, y_train, kfn_rbf, C=C, epsilon=0.1)
    mse = np.mean((svr_predict(X_test, X_train, coef, bias, kfn_rbf) - y_test)**2)
    print(f"  RBF | C={C:<3} | ε=0.1 | gamma=1   | #SV={len(sv_idx):3d} | Test MSE={mse:.4f}")

for g in gammas:
    kfn = lambda a, b, gamma=g: rbf_kernel(a, b, gamma)
    coef, bias, sv_idx, _ = svr_qp(X_train, y_train, kfn, C=1.0, epsilon=0.1)
    mse = np.mean((svr_predict(X_test, X_train, coef, bias, kfn) - y_test)**2)
    print(f"  RBF | C=1   | ε=0.1 | gamma={g:<3} | #SV={len(sv_idx):3d} | Test MSE={mse:.4f}")

