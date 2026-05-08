"""
cart.py
=======
Lab 5, Q2: Implement a CART (Classification and Regression Tree) classifier
           using Gini Impurity for binary and multi-class classification.

Only NumPy is allowed. Do NOT import any other libraries.
"""

import numpy as np
from typing import Optional, Dict, Union


# ============================================
# TREE NODE DATA STRUCTURES
# ============================================

class LeafNode:
    """
    A leaf (terminal) node in the decision tree.

    Attributes:
        predicted_class (int): The majority class label for this leaf.
        n_samples       (int): Number of training samples that reached this leaf.
    """

    def __init__(self, predicted_class: int, n_samples: int) -> None:
        self.predicted_class = predicted_class
        self.n_samples = n_samples


class DecisionNode:
    """
    An internal (decision) node in the decision tree.

    Attributes:
        feature_index (int):   The feature index j used for the split.
        threshold     (float): The threshold t used for the split.
        left          (Union[DecisionNode, LeafNode]): Left child  (x_j <= t).
        right         (Union[DecisionNode, LeafNode]): Right child (x_j >  t).
        n_samples     (int):   Number of training samples that reached this node.
    """

    def __init__(
        self,
        feature_index: int,
        threshold: float,
        left: Union["DecisionNode", LeafNode] = None,
        right: Union["DecisionNode", LeafNode] = None,
        n_samples: int = 0,
    ) -> None:
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.n_samples = n_samples


# ============================================
# TASK 1: GINI IMPURITY
# ============================================

def gini_impurity(y: np.ndarray) -> float:
    """
    Compute the Gini Impurity for a set of labels.

        Gini(y) = 1 - sum_c (p_c)^2

    where p_c is the fraction of samples belonging to class c.

    Args:
        y (np.ndarray): Label array of shape (N,). Values are integer class
                        labels (e.g., 0, 1, 2, …).

    Returns:
        float: Gini impurity in [0, 1).
               Returns 0.0 if y is empty (no samples).

    Hint:
        - Use len(y) to check the number of elements in the array.
        - np.unique(y, return_counts=True) returns two arrays:
            classes, counts = np.unique(y, return_counts=True)
          where `classes` contains the distinct labels and `counts`
          contains how many times each label appears.
        - Compute fractions: p = counts / counts.sum()  (vectorised division).
        - Gini = 1 - np.sum(p ** 2). The ** operator squares element-wise.
        - Handle the edge case where y is empty (len(y) == 0) → return 0.0.
    """
    # TODO: Implement
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(1.0 - np.sum(p ** 2))
    # END TODO


# ============================================
# TASK 2: FIND BEST SPLIT
# ============================================

def find_best_split(
    X: np.ndarray,
    y: np.ndarray,
) -> Optional[Dict]:
    """
    Find the best binary split (feature_index, threshold) that minimises
    the weighted Gini impurity of the resulting children.

    For each feature j in {0, …, D-1}:
      - Extract the unique values in X[:, j].
      - For each unique value t (used as threshold):
          * Left  split: y[X[:, j] <= t]
          * Right split: y[X[:, j] >  t]
          * Compute weighted cost:
              J(j, t) = (N_L / N) * Gini(y_left) + (N_R / N) * Gini(y_right)
      - Track the (j, t) pair that gives the minimum cost.

    Args:
        X (np.ndarray): Feature matrix of shape (N, D).
        y (np.ndarray): Label vector of shape (N,). Integer class labels.

    Returns:
        Optional[Dict]: A dictionary with keys:
            {
                "feature_index": int,   # best feature j*
                "threshold":     float, # best threshold t*
                "cost":          float, # minimum weighted Gini cost
            }
        Returns None if no valid split is found (e.g., all features have a
        single unique value, so no split can separate the data).

    Hint:
        - Use boolean indexing to split the data without an explicit loop
          over samples:
            mask = X[:, j] <= t       # boolean array of shape (N,)
            y_left  = y[mask]          # labels where condition is True
            y_right = y[~mask]         # labels where condition is False
          The ~ operator flips True↔False in a boolean array.
        - len(y_left) and len(y_right) give the child sizes N_L, N_R.
        - A valid split requires both children to be non-empty;
          skip any threshold where len(y_left)==0 or len(y_right)==0.
        - np.unique(X[:, j]) returns the sorted unique values in a column,
          which serve as candidate thresholds.
        - Initialise best_cost = np.inf (positive infinity) and update
          whenever you find a cost strictly less than the current best.
        - Build the result as a Python dict literal:
            {"feature_index": j, "threshold": float(t), "cost": float(cost)}
        - Return None if no valid split was found (best_split is still None).

    VECTORISATION NOTE:
        The outer loop over features and inner loop over thresholds is
        acceptable here because D and the number of unique values are
        typically small. Within each iteration, the split and Gini
        computation should be vectorised (no loop over samples).
    """
    # TODO: Implement
    N, D = X.shape
    best_cost = np.inf
    best_split = None

    for j in range(D):
        thresholds = np.unique(X[:, j])
        for t in thresholds:
            mask = X[:, j] <= t
            y_left = y[mask]
            y_right = y[~mask]

            # Both children must be non-empty for a valid split
            if len(y_left) == 0 or len(y_right) == 0:
                continue

            N_L = len(y_left)
            N_R = len(y_right)
            cost = (N_L / N) * gini_impurity(y_left) + (N_R / N) * gini_impurity(y_right)

            if cost < best_cost:
                best_cost = cost
                best_split = {
                    "feature_index": j,
                    "threshold": float(t),
                    "cost": float(cost),
                }

    return best_split
    # END TODO


# ============================================
# TASK 3: DECISION TREE CLASSIFIER
# ============================================

class DecisionTree:
    """
    CART Decision Tree Classifier using Gini Impurity.

    Attributes:
        max_depth          (int):  Maximum depth of the tree. None = unlimited.
        min_samples_split  (int):  Minimum samples required to attempt a split.
        root (Union[DecisionNode, LeafNode, None]): The root of the fitted tree.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: Optional[Union[DecisionNode, LeafNode]] = None

    @staticmethod
    def _majority_class(y: np.ndarray) -> int:
        """
        Return the most frequent class label in y.

        Args:
            y (np.ndarray): Label array of shape (N,). Integer class labels.

        Returns:
            int: The class label that appears most frequently.

        Hint:
            - y.astype(int) converts float labels to integers (required by
              np.bincount).
            - np.bincount(y_int) returns an array where element i is the
              number of times value i appears in y_int.  For example,
              np.bincount([0,1,1,2]) → [1, 2, 1].
            - np.argmax(counts) returns the index of the largest element,
              which equals the most frequent class label.
        """
        # TODO: Implement
        counts = np.bincount(y.astype(int))
        return int(np.argmax(counts))
        # END TODO

    def _grow_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int = 0,
    ) -> Union[DecisionNode, LeafNode]:
        """
        Recursively grow the decision tree.

        Stopping criteria (return a LeafNode if ANY of these hold):
          1. max_depth is not None and depth >= max_depth.
          2. All labels in y are the same (pure node).
          3. Number of samples < min_samples_split.
          4. No valid split can be found (find_best_split returns None).

        If not stopping:
          1. Call find_best_split(X, y) to get the best (feature, threshold).
          2. Split X and y into left (X[:, j] <= t) and right (X[:, j] > t).
          3. Recursively call _grow_tree on the left and right subsets with
             depth + 1.
          4. Return a DecisionNode storing the split parameters and children.

        Args:
            X (np.ndarray):  Feature matrix of shape (N, D).
            y (np.ndarray):  Label vector of shape (N,). Integer class labels.
            depth (int):     Current depth in the tree (root = 0).

        Returns:
            Union[DecisionNode, LeafNode]: The constructed (sub)tree.

        Hint:
            - To create a leaf: LeafNode(predicted_class=self._majority_class(y),
              n_samples=len(y)).
            - Check purity: len(np.unique(y)) == 1 means all labels are the same.
            - find_best_split(X, y) returns a dict or None. Access the split
              parameters via best["feature_index"] and best["threshold"].
            - Boolean mask for split: mask = X[:, j] <= t
              Left:  X[mask], y[mask]      (rows where condition is True)
              Right: X[~mask], y[~mask]    (rows where condition is False)
            - Recurse: left_child  = self._grow_tree(X_left, y_left, depth + 1)
                       right_child = self._grow_tree(X_right, y_right, depth + 1)
            - Return: DecisionNode(feature_index=j, threshold=t,
                      left=left_child, right=right_child, n_samples=len(y))
        """
        # TODO: Implement
        N = len(y)

        # --- Stopping criteria ---
        # 1. Max depth reached
        if self.max_depth is not None and depth >= self.max_depth:
            return LeafNode(self._majority_class(y), N)

        # 2. Pure node (all labels identical)
        if len(np.unique(y)) == 1:
            return LeafNode(int(y[0]), N)

        # 3. Too few samples to split
        if N < self.min_samples_split:
            return LeafNode(self._majority_class(y), N)

        # 4. No valid split found
        best = find_best_split(X, y)
        if best is None:
            return LeafNode(self._majority_class(y), N)

        # --- Recurse ---
        j = best["feature_index"]
        t = best["threshold"]
        mask = X[:, j] <= t

        left_child = self._grow_tree(X[mask], y[mask], depth + 1)
        right_child = self._grow_tree(X[~mask], y[~mask], depth + 1)

        return DecisionNode(
            feature_index=j,
            threshold=t,
            left=left_child,
            right=right_child,
            n_samples=N,
        )
        # END TODO

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Build the decision tree from training data.

        Args:
            X (np.ndarray): Feature matrix of shape (N, D).
            y (np.ndarray): Label vector of shape (N,). Integer class labels.

        Returns:
            None (stores the tree in self.root).

        Hint:
            - Convert labels to ints before growing the tree:
              y.astype(int) converts a float array like [0.0, 1.0, 2.0] into
              the integer array [0, 1, 2].  This is needed because np.bincount
              only accepts integer arrays.
            - Call self._grow_tree(X, y_int, depth=0) and store the result
              in self.root.
        """
        # TODO: Implement
        self.root = self._grow_tree(X, y.astype(int), depth=0)
        # END TODO

    def _predict_single(
        self,
        x: np.ndarray,
        node: Union[DecisionNode, LeafNode],
    ) -> int:
        """
        Traverse the tree to predict the class for a single sample.

        Args:
            x (np.ndarray):  Single feature vector of shape (D,).
            node (Union[DecisionNode, LeafNode]): Current node in the tree.

        Returns:
            int: Predicted class label.

        Hint:
            - Use isinstance(node, LeafNode) to check the node type.
              isinstance returns True if `node` is an instance of the given
              class.  Example: isinstance(node, LeafNode) → True/False.
            - If the node is a LeafNode, simply return node.predicted_class.
            - If the node is a DecisionNode, read the split rule from
              node.feature_index and node.threshold, then compare:
                if x[node.feature_index] <= node.threshold:
                    → recurse into node.left
                else:
                    → recurse into node.right
            - This is a recursive function — it calls itself on child nodes
              until it reaches a LeafNode.
        """
        # TODO: Implement
        if isinstance(node, LeafNode):
            return node.predicted_class
        # DecisionNode: go left if x[j] <= t, else right
        if x[node.feature_index] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)
        # END TODO

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for a batch of samples.

        Args:
            X (np.ndarray): Feature matrix of shape (N, D).

        Returns:
            np.ndarray: Predicted labels of shape (N,). Integer class labels.

        Hint:
            - Apply _predict_single to each row of X.  Iterating over a 2D
              NumPy array (for x in X) yields one row at a time as a 1D array.
            - A list comprehension collects results:
                preds = [self._predict_single(x, self.root) for x in X]
            - Wrap in np.array(preds) to return an ndarray of shape (N,).
            - Note: a Python-level loop over rows is fine here because the
              tree traversal is inherently sequential per sample.
        """
        # TODO: Implement
        return np.array([self._predict_single(x, self.root) for x in X])
        # END TODO

    def get_depth(self) -> int:
        """
        Return the maximum depth of the fitted tree.

        Returns:
            int: Depth of the tree (root-only tree has depth 0).
                 Returns -1 if the tree has not been fitted yet.
        """
        if self.root is None:
            return -1

        def _depth(node: Union[DecisionNode, LeafNode]) -> int:
            if isinstance(node, LeafNode):
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))

        return _depth(self.root)

    def get_n_leaves(self) -> int:
        """
        Return the number of leaf nodes in the fitted tree.

        Returns:
            int: Number of leaves. Returns 0 if the tree has not been fitted.
        """
        if self.root is None:
            return 0

        def _count_leaves(node: Union[DecisionNode, LeafNode]) -> int:
            if isinstance(node, LeafNode):
                return 1
            return _count_leaves(node.left) + _count_leaves(node.right)

        return _count_leaves(self.root)
