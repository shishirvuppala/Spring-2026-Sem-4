================================================================================
  Lab 5 — Q2: Decision Trees: The Geometry of Splitting
================================================================================

OVERVIEW
--------
You will implement a CART (Classification and Regression Tree) classifier
from scratch using Gini Impurity.  The experiment is evaluated on THREE
datasets of increasing complexity:

  Dataset A — Two Moons (Non-Linear 2D)
    Two interleaving crescent-shaped classes.  N=500, D=2.
    Tests whether your tree can learn a non-linear decision boundary.
    Visualise the "staircase" boundary at different depths.

  Dataset B — Iris (Multi-Class 4D)
    Full Iris dataset with 3 classes.  N=150, D=4.
    Tests multi-class support in your Gini and majority-vote logic.

  Dataset C — Concentric Circles (Non-Linear 2D)
    Two concentric circles.  N=500, D=2.
    Another non-linear problem; observe overfitting at high depth.

FILES
-----
  cart.py        — YOUR CODE GOES HERE (Gini, find_best_split, DecisionTree)
  utils.py       — YOUR CODE GOES HERE (evaluation metrics)
  main.py        — Experiment / plotting script (DO NOT MODIFY)
  debug.py       — Sanity-check script to test your implementations
  setup_data.py  — Data generation script (already run; DO NOT MODIFY)
  data/
    moons/       — train.csv, test.csv
    iris/        — train.csv, test.csv
    circles/     — train.csv, test.csv

RULES
-----
  • Only NumPy is allowed. Do NOT import any other library.
  • Use vectorised operations wherever possible. The outer loops over
    features and thresholds in find_best_split are acceptable, but the
    split and Gini computation within each iteration must be vectorised.
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

--- cart.py ---

  TASK 1 — Gini Impurity

  9. gini_impurity(y) -> float
       Compute Gini(y) = 1 - sum_c (p_c)^2 where p_c is the fraction of
       samples belonging to class c. Return 0.0 for empty arrays.
       Hint: Use np.unique(y, return_counts=True).

  TASK 2 — Greedy Splitter

 10. find_best_split(X, y) -> Optional[Dict]
       For every feature j and every unique threshold t in X[:, j]:
         - Split y into y_left (X[:, j] <= t) and y_right (X[:, j] > t).
         - Compute weighted cost: J = (N_L/N)*Gini(y_L) + (N_R/N)*Gini(y_R).
       Return dict {"feature_index": j*, "threshold": t*, "cost": J*}
       for the split with minimum cost, or None if no valid split exists.

  TASK 3 — Decision Tree Classifier

 11. DecisionTree._majority_class(y) -> int   (static method)
       Return the most frequent class label in y.
       Hint: np.bincount + np.argmax.

 12. DecisionTree._grow_tree(X, y, depth) -> Union[DecisionNode, LeafNode]
       Recursively build the tree.
       Stopping criteria (return LeafNode if ANY hold):
         a) max_depth is not None and depth >= max_depth
         b) All labels identical (pure node)
         c) len(y) < min_samples_split
         d) find_best_split returns None
       Otherwise: find best split, partition data, recurse on left/right.

 13. DecisionTree.fit(X, y) -> None
       Build the tree by calling self._grow_tree(X, y, depth=0).
       Store the result in self.root.

 14. DecisionTree._predict_single(x, node) -> int
       Traverse the tree for a single sample x.
       If node is LeafNode → return predicted_class.
       If DecisionNode → go left if x[j] <= t, else right.

 15. DecisionTree.predict(X) -> np.ndarray
       Apply _predict_single to each row of X. Return shape (N,).

  TASK 3 also involves evaluation — once you implement the tree,
  main.py will:
    - Train trees at max_depth = {1, 3, 5, 10, None}.
    - Evaluate each with accuracy, precision, recall, F1 on the test set.
    - Plot decision boundaries for 2D datasets at depth 3 and 10.
    - Plot train vs test accuracy across depths (overfitting analysis).

================================================================================
  ALGORITHMIC NOTES
================================================================================

  • SPLIT SEARCH: The split search iterates over all features and all unique
    values per feature. For each candidate split, use boolean indexing to
    partition labels. This is O(D * N * U) where U = max unique values.
    Since D and U are typically small, this is acceptable.

  • GINI IMPURITY: Gini(y) = 1 - sum(p_c^2). A pure node has Gini = 0.
    The maximum for K classes is 1 - 1/K (balanced classes).

  • MAJORITY CLASS: When creating a leaf, use the most frequent class.
    np.bincount(y.astype(int)) returns an array of counts; np.argmax
    gives the class with the highest count.

  • TREE DEPTH: The root is at depth 0. max_depth=None means unlimited
    growth (the tree grows until all leaves are pure or contain fewer
    than min_samples_split samples).

  • OVERFITTING: With unlimited depth, the tree can perfectly memorise the
    training data (train accuracy = 1.0), but this often hurts test accuracy.
    Observe this in the accuracy-vs-depth plots.

================================================================================
  HOW TO TEST
================================================================================
  1. Implement functions in utils.py and cart.py.
  2. Run  python debug.py   — checks basic correctness and edge cases.
  3. Run  python main.py    — full experiment with plots and metrics.
================================================================================
