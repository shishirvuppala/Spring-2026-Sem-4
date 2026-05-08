import numpy as np
import matplotlib.pyplot as plt

from q1 import OneVsRestClassifier, get_multiclass_accuracy
from q1 import get_precision, get_recall, get_f1_score

def load_data(filepath):
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

def compute_confusion_matrix(y_true, y_pred, classes):
    K = len(classes)
    cm = np.zeros((K, K), dtype=int)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for i in range(len(y_true)):
        cm[class_to_idx[int(y_true[i])], class_to_idx[int(y_pred[i])]] += 1
    return cm

def print_metrics_report(y_true, y_pred, title):
    print(f"\n{title}")
    print(f"{'-'*65}")
    print(f"{'Class':<6} | {'Recall':<10} | {'Precision':<10} | {'F1':<10}")
    print(f"{'-'*65}")
    
    classes = np.unique(y_true)
    for k in classes:
        y_true_k = np.where(y_true == k, 1, 0)
        y_pred_k = np.where(y_pred == k, 1, 0)
        
        rec = get_recall(y_true_k, y_pred_k)
        prec = get_precision(y_true_k, y_pred_k)
        f1 = get_f1_score(y_true_k, y_pred_k)
        
        print(f"{int(k):<6} | {rec:.4f}     | {prec:.4f}     | {f1:.4f}")
    print(f"{'-'*65}")
    print(f"Overall Accuracy: {get_multiclass_accuracy(y_true, y_pred):.4f}")

def plot_confusion_matrices(y_test, preds_naive, preds_weighted):
    classes = np.unique(y_test)
    cm_naive = compute_confusion_matrix(y_test, preds_naive, classes)
    cm_weighted = compute_confusion_matrix(y_test, preds_weighted, classes)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Naive Plot
    axes[0].imshow(cm_naive, cmap='Reds')
    axes[0].set_title(f"Naive Model (Test Data)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    
    for i in range(len(classes)):
        for j in range(len(classes)):
            axes[0].text(j, i, cm_naive[i, j], ha="center", va="center", color="black")

    # Weighted Plot
    axes[1].imshow(cm_weighted, cmap='Greens')
    axes[1].set_title(f"Weighted Model (Test Data)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    
    for i in range(len(classes)):
        for j in range(len(classes)):
            axes[1].text(j, i, cm_weighted[i, j], ha="center", va="center", color="black")
            
    plt.tight_layout()
    plt.show()

def main():
    print("=== LAB 4 Q1: Handling Covariate Shift ===")
    
    try:
        X_train, y_train = load_data('train.csv')
        X_test, y_test = load_data('test.csv')
    except:
        print("Error: CSV files not found. Run datagen.py first.")
        return

    # Add Intercept
    X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]
    X_test_bias = np.c_[np.ones(X_test.shape[0]), X_test]

    # --- Naive Model ---
    print("\n[Training Naive Model]...")
    naive_model = OneVsRestClassifier(mode='naive', lr=0.5, max_iter=4000)
    naive_model.fit(X_train_bias, y_train)
    preds_naive = naive_model.predict(X_test_bias)
    print_metrics_report(y_test, preds_naive, "NAIVE RESULTS")

    # --- Weighted Model ---
    print("\n[Training Weighted Model]...")
    classes = np.unique(y_train)
    K = len(classes)
    equal_ratios = {k: 1.0/K for k in classes}
    print(f"Target Distribution (Equal Ratios): {equal_ratios}")

    weighted_model = OneVsRestClassifier(mode='weighted', lr=0.5, max_iter=4000, 
                                         test_ratios=equal_ratios)
    weighted_model.fit(X_train_bias, y_train)
    preds_weighted = weighted_model.predict(X_test_bias)
    print_metrics_report(y_test, preds_weighted, "WEIGHTED RESULTS")

    # Visualization
    print("\nDisplaying Confusion Matrices...")
    plot_confusion_matrices(y_test, preds_naive, preds_weighted)

if __name__ == "__main__":
    main()