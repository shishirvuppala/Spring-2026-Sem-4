import numpy as np

from q4 import (
    SoftmaxGenerativeClassifier,
    ImportanceSamplingClassifier,
    NCEClassifier,
    GaussianSampler,
    CategoricalSampler,
    accuracy,
    precision,
    recall,
    f1_score
)


# ======================================================
# DATA LOADING
# ======================================================

def load_data(filepath):
    """
    Load CSV data.
    Assumes last column is the label.
    """
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    return X, y


# ======================================================
# METRIC REPORTING
# ======================================================

def print_metrics_report(y_true, y_pred, title):
    print(f"\n{title}")
    print(f"{'-'*70}")
    print(f"{'Class':<6} | {'Recall':<10} | {'Precision':<10} | {'F1':<10}")
    print(f"{'-'*70}")

    classes = np.unique(y_true)
    for k in classes:
        rec = recall(y_true, y_pred, k)
        prec = precision(y_true, y_pred, k)
        f1 = f1_score(y_true, y_pred, k)
        print(f"{k:<6} | {rec:.4f}     | {prec:.4f}     | {f1:.4f}")

    print(f"{'-'*70}")
    print(f"Overall Accuracy: {accuracy(y_true, y_pred):.4f}")


# ======================================================
# EMPIRICAL CLASS MEANS (VECTORIZED)
# ======================================================

def compute_empirical_class_means(X, y, num_classes):
    """
    Compute empirical class means in a vectorized manner.

    Args:
        X (np.ndarray): Features, shape (N, D)
        y (np.ndarray): Labels, shape (N,)
        num_classes (int)

    Returns:
        np.ndarray: Class means, shape (K, D)
    """
    N, D = X.shape

    # One-hot encode labels: shape (N, K)
    Y = np.eye(num_classes)[y]

    # Sum features per class: (K, D)
    class_sums = Y.T @ X

    # Number of samples per class: (K,)
    class_counts = Y.sum(axis=0)

    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1.0)

    return class_sums / class_counts[:, None]


# ======================================================
# MAIN
# ======================================================
def main():
    print("=== LAB: GENERATIVE CLASSIFIERS ===")

    # --------------------------------------------------
    # SOFTMAX MODEL (TRAIN1 / TEST1)
    # --------------------------------------------------

    print("\n[Loading data for Softmax model: train1.csv / test1.csv]")
    try:
        X_train1, y_train1 = load_data("train1.csv")
        X_test1, y_test1 = load_data("test1.csv")
    except:
        print("Error: train1.csv or test1.csv not found.")
        return

    num_classes = len(np.unique(y_train1))

    empirical_mu = compute_empirical_class_means(
        X_train1, y_train1, num_classes
    )

    print("\nComputed empirical class means (train1)")
    print("Shape:", empirical_mu.shape)

    print("\n[Training Softmax (Exact Normalization)]")
    softmax_model = SoftmaxGenerativeClassifier(
        num_classes=num_classes,
        lr=1e-2,
        batch_size=64,
        max_epochs=50
    )

    # Store empirical means inside the model
    softmax_model.mu = empirical_mu

    softmax_model.fit(X_train1, y_train1)

    preds_softmax = softmax_model.predict(X_test1)
    print_metrics_report(y_test1, preds_softmax, "SOFTMAX RESULTS")

    recovered = softmax_model.recover_parameters()

    print("\nRecovered Parameters from Softmax model")
    print("Class Priors:", softmax_model.pi)
    print("Sigma:", softmax_model.Sigma)


    # --------------------------------------------------
    # IMPORTANCE SAMPLING + NCE (TRAIN2 / TEST2)
    # --------------------------------------------------

    print("\n[Loading data for IS / NCE models: train2.csv / test2.csv]")
    try:
        X_train2, y_train2 = load_data("train2.csv")
        X_test2, y_test2 = load_data("test2.csv")
    except:
        print("Error: train2.csv or test2.csv not found.")
        return

    # --------------------------------------------------
    # IMPORTANCE SAMPLING
    # --------------------------------------------------

    print("\n[Training Importance Sampling Model]")

    num_classes2 = len(np.unique(y_train2))

    class_probs = np.bincount(y_train2) / len(y_train2)

    q_y = CategoricalSampler(class_probs)

    # q_y = CategoricalSampler(
    #     probs=np.ones(num_classes2) / num_classes2
    # ) could use samplers like this as well

    is_model = ImportanceSamplingClassifier(
        num_classes=num_classes2,
        lr=5e-3,
        batch_size=64,
        num_samples=10,
        max_epochs=50,
        class_sampler=q_y
    )
    is_model.fit(X_train2, y_train2)

    preds_is = is_model.predict(X_test2)
    print_metrics_report(y_test2, preds_is, "IMPORTANCE SAMPLING RESULTS")

    # --------------------------------------------------
    # NOISE CONTRASTIVE ESTIMATION
    # --------------------------------------------------

    print("\n[Training Noise Contrastive Estimation Model]")

    q_x = GaussianSampler(
        mean=np.mean(X_train2, axis=0),
        cov=np.cov(X_train2.T)
    )

    q_y = CategoricalSampler(
        probs=np.ones(num_classes2) / num_classes2
    )

    nce_model = NCEClassifier(
        num_classes=num_classes2,
        lr=1e-2,
        batch_size=64,
        noise_ratio=5,
        max_epochs=50,
        x_sampler=q_x,
        y_sampler=q_y
    )
    nce_model.fit(X_train2, y_train2)

    preds_nce = nce_model.predict(X_test2)
    print_metrics_report(y_test2, preds_nce, "NCE RESULTS")


if __name__ == "__main__":
    main()
