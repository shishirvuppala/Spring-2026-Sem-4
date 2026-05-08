"""
SOLUTION FILE — not distributed to students.
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------

def load_data(path: str = "dataset.npz") -> dict:
    data = np.load(path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
    }


# ---------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------

def train_svm(X_train, y_train, kernel="rbf", C=1.0, gamma="scale", degree=3):
    clf = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(model, X_test, y_test):
    acc = model.score(X_test, y_test)
    nsv = int(sum(model.n_support_))
    nsv_per_class = list(model.n_support_.tolist())
    return {
        "accuracy": float(acc),
        "n_support_vectors": nsv,
        "n_support_per_class": nsv_per_class,
    }


# ---------------------------------------------------------------------------
# Part C
# ---------------------------------------------------------------------------

def find_best_config(X_train, y_train, X_test, y_test,
                     accuracy_threshold=0.95):
    from sklearn.model_selection import cross_val_score

    best = None

    for kernel in ["rbf"]:
        for C in [5, 10, 20, 50, 100, 200, 500]:
            for gamma in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]:
                clf = SVC(kernel=kernel, C=C, gamma=gamma)

                # Cross-validation on training set for robustness
                cv_scores = cross_val_score(clf, X_train, y_train, cv=3,
                                           scoring="accuracy")
                if cv_scores.mean() < accuracy_threshold - 0.02:
                    continue

                clf.fit(X_train, y_train)
                acc = clf.score(X_test, y_test)
                nsv = int(sum(clf.n_support_))

                if acc >= accuracy_threshold:
                    if best is None or nsv < best["n_support_vectors"]:
                        best = {
                            "kernel": kernel,
                            "C": float(C),
                            "gamma": gamma,
                            "degree": 3,
                            "accuracy": float(acc),
                            "n_support_vectors": nsv,
                            "model": clf,
                        }

    return best