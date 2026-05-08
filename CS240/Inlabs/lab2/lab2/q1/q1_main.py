"""
q1_main.py

You MUST NOT edit this file
All implementations go into q1_solve.py

Usage:
    python q1_main.py
    python q1_main.py --verify-sklearn
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import q1_solve as solve

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


def load_csv(path):
    data = np.loadtxt(path, delimiter=",")
    return data[:, :-1], data[:, -1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-sklearn", action="store_true")
    args = parser.parse_args()

    print("=== Ridge Regression Lab ===")

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    X_train_raw, y_train = load_csv("q1_train.csv")
    X_test_raw, y_test = load_csv("q1_test.csv")

    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------
    X_train_std, mu, sigma = solve.standardize_train(X_train_raw)
    X_test_std = solve.standardize_apply(X_test_raw, mu, sigma)

    X_train = solve.add_bias(X_train_std)
    X_test = solve.add_bias(X_test_std)

    # --------------------------------------------------
    # OLS baseline
    # --------------------------------------------------
    w_ols = solve.ridge_regression_closed_form(X_train, y_train, lam=0.0)
    train_mse_ols = solve.mse(y_train, X_train @ w_ols)
    test_mse_ols = solve.mse(y_test, X_test @ w_ols)

    print(f"OLS -> Train MSE: {train_mse_ols:.4f}, Test MSE: {test_mse_ols:.4f}")

    # --------------------------------------------------
    # Lambda ranges
    # --------------------------------------------------
    lambdas_grid = np.logspace(-4, 4, 25)

    # --------------------------------------------------
    # Grid Search
    # --------------------------------------------------
    grid_lams, grid_mses = solve.grid_search_lambdas(X_train, y_train, lambdas_grid, k=5)
    best_idx_grid = int(np.argmin(grid_mses))
    best_lambda_grid = float(grid_lams[best_idx_grid])

    print(f"Grid Search best λ = {best_lambda_grid:.6g}, CV MSE = {grid_mses[best_idx_grid]:.4f}")

    # --------------------------------------------------
    # Random Search
    # --------------------------------------------------
    rand_lams, rand_mses = solve.random_search_lambdas(X_train, y_train, n_iter=20, low_exp=-4, high_exp=4, k=5)
    best_idx_rand = int(np.argmin(rand_mses))
    best_lambda_rand = float(rand_lams[best_idx_rand])

    print(f"Random Search best λ = {best_lambda_rand:.6g}, CV MSE = {rand_mses[best_idx_rand]:.4f}")

    # --------------------------------------------------
    # Evaluate best (grid) model
    # --------------------------------------------------
    w_best = solve.ridge_regression_closed_form(X_train, y_train, best_lambda_grid)
    test_mse_best = solve.mse(y_test, X_test @ w_best)

    print(f"Best Ridge -> Test MSE: {test_mse_best:.4f}")

    # --------------------------------------------------
    # Optional sklearn verification
    # --------------------------------------------------
    if args.verify_sklearn:
        print("Running sklearn sanity check...")
        kf = KFold(n_splits=5, shuffle=True)
        sk_mses = []
        for tr, va in kf.split(X_train):
            model = Ridge(alpha=best_lambda_grid, fit_intercept=False)
            model.fit(X_train[tr], y_train[tr])
            sk_mses.append(
                mean_squared_error(y_train[va], model.predict(X_train[va]))
            )
        print(f"[sklearn] CV MSE ≈ {np.mean(sk_mses):.4f}")

    # --------------------------------------------------
    # Plots
    # --------------------------------------------------
    plt.figure()
    plt.semilogx(grid_lams, grid_mses, marker="o", label="Grid Search")
    plt.scatter(rand_lams, rand_mses, c="red", label="Random Search")
    plt.xlabel("λ")
    plt.ylabel("CV MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cv_curve.png")

    # --------------------------------------------------
    # Save metrics
    # --------------------------------------------------
    metrics = {
        "ols_test_mse": test_mse_ols,
        "best_lambda_grid": best_lambda_grid,
        "best_lambda_random": best_lambda_rand,
        "best_ridge_test_mse": test_mse_best
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved cv_curve.png and metrics.json")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
