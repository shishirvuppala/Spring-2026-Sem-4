================================================================================
  Lab 5 — Q1: Generative ↔ Discriminative Hybrid Classifier
================================================================================

OVERVIEW
--------
You will implement three tasks that combine a generative model (LDA) with a
discriminative model (Logistic Regression) to create a warm-started hybrid
classifier.  The experiment is evaluated on THREE datasets of increasing
complexity:

  Dataset A — Iris (Sanity Check)
    Binary subset (Setosa vs Versicolor).  N=100, D=4.
    Classes are linearly separable. Both LDA and LR should get ~100%.
    Use this to verify your LDA implementation first.

  Dataset B — Breast Cancer (Convergence Benchmark)
    Wisconsin Diagnostic Breast Cancer.  N=569, D=30.
    Primary dataset for the convergence analysis (Task 3).
    Random init takes much longer to converge than warm start.

  Dataset C — Sonar (High-Dimensional Challenge)
    Mines vs Rocks.  N=208, D=60.  D ≈ N/3 → prone to overfitting.
    Observe whether the LDA prior acts as a regulariser.

FILES
-----
  classifiers.py   — YOUR CODE GOES HERE (LDA + Logistic Regression classes)
  utils.py         — YOUR CODE GOES HERE (data loading + evaluation metrics)
  main.py          — Experiment / plotting script (DO NOT MODIFY)
  debug.py         — Sanity-check script to test your implementations
  setup_data.py    — Data generation script (already run; DO NOT MODIFY)
  data/
    iris/          — train.csv, test.csv
    cancer/        — train.csv, test.csv, subset_train.csv
    sonar/         — train.csv, test.csv

RULES
-----
  • Only NumPy is allowed. Do NOT import any other library.
  • Use vectorised operations. Avoid explicit Python for-loops over samples.
  • Follow the exact function signatures, return types, and shapes in the
    docstrings. The autograder checks shapes and types strictly.
  • Do NOT modify main.py, setup_data.py, or the data/ directory.

================================================================================
  TODO LIST — Functions you must implement
================================================================================

--- utils.py ---

  1. get_true_positives(y_true, y_pred) -> int
       Count samples where y_true=1 AND y_pred=1.

  2. get_false_positives(y_true, y_pred) -> int
       Count samples where y_true=0 AND y_pred=1.

  3. get_false_negatives(y_true, y_pred) -> int
       Count samples where y_true=1 AND y_pred=0.

  4. get_true_negatives(y_true, y_pred) -> int
       Count samples where y_true=0 AND y_pred=0.

  5. get_precision(y_true, y_pred) -> float
       Precision = TP / (TP + FP). Return 0.0 if denominator is 0.

  6. get_recall(y_true, y_pred) -> float
       Recall = TP / (TP + FN). Return 0.0 if denominator is 0.

  7. get_f1_score(y_true, y_pred) -> float
       F1 = 2 * precision * recall / (precision + recall).
       Return 0.0 if denominator is 0.

  8. get_accuracy(y_true, y_pred) -> float
       Fraction of correctly classified samples. Works for multi-class too.

--- classifiers.py ---

  TASK 1 — LDAClassifier

  9. LDAClassifier.fit(X, y) -> None
       Estimate LDA parameters from training data:
         a) Class priors  pi_0, pi_1
         b) Class means   mu_0, mu_1          (shape (D,))
         c) Pooled covariance  sigma           (shape (D, D))
         d) Derive linear weights  w           (shape (D,))  and bias  b  (scalar)
       Formulae:
         w   = Sigma^{-1} (mu_1 - mu_0)
         b   = -0.5 mu_1^T Sigma^{-1} mu_1
             +  0.5 mu_0^T Sigma^{-1} mu_0
             +  ln(pi_1 / pi_0)

 10. LDAClassifier.predict_proba(X) -> np.ndarray
       Return sigmoid(X @ w + b), shape (N,).

 11. LDAClassifier.predict(X) -> np.ndarray
       Threshold at 0.5, return labels in {0, 1}, shape (N,).

  TASK 2 — LogisticRegression

 12. LogisticRegression.sigmoid(z) -> np.ndarray  (static method)
       Numerically stable sigmoid. Clip z to [-500, 500] before exp.

 13. LogisticRegression.compute_loss(y_true, y_pred) -> float
       Binary Cross-Entropy. Add epsilon=1e-9 inside log to avoid log(0).

 14. LogisticRegression.compute_gradient(X, y_true, y_pred)
                                              -> Tuple[np.ndarray, float]
       grad_w = (1/N) X^T (y_pred - y_true),  shape (D,)
       grad_b = (1/N) sum(y_pred - y_true),    scalar

 15. LogisticRegression.fit(X, y, init_params=None) -> None
       Mini-batch gradient descent loop.
       If init_params is None  → random init w ~ N(0,1), b = 0
          (use np.random.seed(42) before sampling for reproducibility).
       If init_params given    → (w_init, b_init) as starting point.
       Each epoch:
         - Shuffle data
         - Iterate over mini-batches, compute predictions + gradient + update
         - After ALL batches, compute loss on FULL train set → append to
           self.loss_history

 16. LogisticRegression.predict_proba(X) -> np.ndarray
       Return sigmoid(X @ w + b), shape (N,).

 17. LogisticRegression.predict(X) -> np.ndarray
       Threshold at 0.5, return labels in {0, 1}, shape (N,).

  TASK 3 — Hybrid Experiment (done automatically by main.py)
       After implementing Tasks 1 & 2, simply run:
           python main.py
       This will train Model A (random init) and Model B (warm start with LDA
       weights), plot loss curves, and print metrics.

================================================================================
  NUMERICAL STABILITY WARNINGS
================================================================================

  • SIGMOID:  exp(-z) overflows for large negative z.
              → Clip z to [-500, 500] before computing 1/(1+exp(-z)).

  • LOG-LOSS: log(0) = -inf corrupts the loss.
              → Use np.clip(y_pred, 1e-9, 1 - 1e-9) before taking log.

  • INVERSE:  The pooled covariance matrix can be nearly singular.
              → np.linalg.pinv is safer than np.linalg.inv.
              → For well-conditioned data, inv is acceptable.

  • LOG-PRIOR: ln(pi_1 / pi_0) can overflow if one class has very few samples.
              → Add a tiny epsilon: np.log((pi_1 + 1e-15) / (pi_0 + 1e-15)).

================================================================================
  HOW TO TEST
================================================================================
  1. Implement functions in utils.py and classifiers.py.
  2. Run  python debug.py   — checks basic correctness and edge cases.
  3. Run  python main.py    — full experiment with plots and metrics.
================================================================================
