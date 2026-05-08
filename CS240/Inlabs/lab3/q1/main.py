"""
main.py
=======
YOU DO NOT NEED TO MODIFY THIS FILE.
run python main.py q1_data.csv to evaluate your implementation.
You can also run python main.py q1_data.csv --compare to compare with sklearn implementations.
Do not worry if the sklearn implementations perform slightly better than yours... They should! (If your implementations are within a few percentage points of sklearn, they should be fine.)
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
from q1_solve import load_data, test_train_split, OLSClassifier, LogisticRegressionGD, normalized_test_train_split

# Optional sklearn comparison
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split as sklearn_train_test_split

def evaluate_student_model(X, y):
    # Instantiate student models
    ols_model = OLSClassifier(lr=0.01, max_iter=1000, tol=1e-6)
    log_model = LogisticRegressionGD(lr=0.01, max_iter=1000, tol=1e-6)

    # Test-train split with normalization
    X_train, X_test, y_train, y_test = normalized_test_train_split(X, y, test_size=0.2)

    # Fit models
    ols_model.fit(X_train, y_train)
    log_model.fit(X_train, y_train)

    # Predict
    y_pred_ols = ols_model.predict(X_test)
    y_pred_log = log_model.predict(X_test)

    # Compute accuracy
    acc_ols = np.mean(y_pred_ols == y_test)
    acc_log = np.mean(y_pred_log == y_test)
    print(f"OLS Accuracy, with normalization: {acc_ols:.4f}")
    print(f"Logistic Regression Accuracy, with normalization: {acc_log:.4f}")

    logistic_loss_history = log_model.logistic_loss_history

    plt.figure()
    plt.plot(logistic_loss_history)
    plt.title("Logistic Loss over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Logistic Loss")
    plt.grid()
    plt.show()


    # Also print these out without normalization for reference
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = test_train_split(X, y, test_size=0.2)
    ols_model.fit(X_train_raw, y_train_raw)
    log_model.fit(X_train_raw, y_train_raw)
    y_pred_ols_raw = ols_model.predict(X_test_raw)
    y_pred_log_raw = log_model.predict(X_test_raw)
    acc_ols_raw = np.mean(y_pred_ols_raw == y_test_raw)
    acc_log_raw = np.mean(y_pred_log_raw == y_test_raw)
    print(f"OLS Accuracy, without normalization: {acc_ols_raw:.4f}")
    print(f"Logistic Regression Accuracy, without normalization: {acc_log_raw:.4f}")    

    return acc_ols, acc_log

def evaluate_sklearn_model(X, y):
    # Linear regression as classifier
    X_train, X_test, y_train, y_test = sklearn_train_test_split(X, y, test_size=0.2)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = np.sign(lr.predict(X_test))

    # Logistic regression
    logr = LogisticRegression(random_state=42, solver='lbfgs')
    logr.fit(X_train, y_train)
    y_pred_logr = logr.predict(X_test)

    acc_lr = accuracy_score(y_test, y_pred_lr)
    acc_logr = accuracy_score(y_test, y_pred_logr)

    print(f"Sklearn Linear Regression Accuracy: {acc_lr:.4f}")
    print(f"Sklearn Logistic Regression Accuracy: {acc_logr:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile", help="Path to CSV dataset")
    parser.add_argument("--compare", action="store_true", help="Compare with sklearn implementations")
    args = parser.parse_args()

    # Load dataset
    X, y = load_data(args.datafile)

    # Evaluate student model
    evaluate_student_model(X, y)

    # Optionally compare with sklearn
    if args.compare:
        evaluate_sklearn_model(X, y)


if __name__ == "__main__":
    main()
