"""
tree.py — Linear Regression Tree with local Ridge estimators.

Implements the CART framework where every node (not just leaves) fits a
local linear model via regularised least-squares on mean-centred data:

    y_hat = (x - x_bar)^T  w  +  y_bar

Tree growth uses an efficient O(N log N  +  N d^2)-per-feature splitting
algorithm that sweeps sorted thresholds while maintaining running
sufficient statistics (Welford-style updates) and matrix inverses
(Sherman-Morrison rank-1 updates).

Only dependency: NumPy.
"""

import numpy as np


# ======================================================================
#  Node data structure
# ======================================================================

class Node:
    """
    A single node in the LinearRegressionTree.

    Every node stores the parameters of a local linear model.  Internal
    nodes additionally store a split criterion (feature index + threshold)
    and references to their two children.

    Attributes
    ----------
    is_leaf : bool
        True if this node is a terminal leaf, False if it has children.
    w : np.ndarray, shape (d,)
        Weight vector of the centred local linear model.
    x_bar : np.ndarray, shape (d,)
        Empirical feature mean of the training data at this node.
    y_bar : float
        Empirical target mean of the training data at this node.
    feature : int or None
        Index of the feature used for splitting (None for leaves).
    threshold : float or None
        Split threshold (None for leaves).
        Points with ``x[feature] <= threshold`` are routed left.
    left : Node or None
        Left child node  (None for leaves).
    right : Node or None
        Right child node (None for leaves).
    """

    def __init__(self):
        self.is_leaf   = True
        self.w         = None
        self.x_bar     = None
        self.y_bar     = None
        self.feature   = None
        self.threshold = None
        self.left      = None
        self.right     = None


# ======================================================================
#  Linear Regression Tree
# ======================================================================

class LinearRegressionTree:
    """
    Regression Tree with local linear (Ridge) estimators at every node.

    Instead of predicting a constant mean at each leaf, every node fits
    a local linear model on mean-centred features:

        y_hat  =  (x  -  x_bar)^T  w  +  y_bar

    The weight vector ``w`` minimises the regularised loss:

        L(w) = || y_c  -  X_c w ||_2^2  +  lam || w ||_2^2

    where  X_c = X - x_bar ,  y_c = y - y_bar  are mean-centred.

    Splits are found with an efficient sequential sweep per feature:

    1. Sort the N samples by the candidate feature.
    2. Maintain running sufficient statistics (Welford-style online
       updates) for the left and right partitions.
    3. Maintain the matrix inverse (S_xx + lam I)^{-1} incrementally
       with the Sherman-Morrison rank-1 update — O(d^2) per step
       instead of O(d^3).

    Parameters
    ----------
    max_depth : int or None
        Maximum depth of the tree (root is depth 0).
        ``None`` means no depth limit — the tree grows until no
        improving split exists.
    lam : float
        Strictly positive L2 regularisation parameter (lambda).
    """

    def __init__(self, max_depth=None, lam=1.0):
        assert lam > 0, "Regularisation parameter lam must be > 0."
        self.max_depth   = max_depth
        self.lam         = lam
        self.root        = None
        self.n_features_ = None          # set at fit time

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Build the regression tree from training data.

        Parameters
        ----------
        X : np.ndarray, shape (N, d)
            Training feature matrix.  Each row is a sample.
        y : np.ndarray, shape (N,) or (N, 1)
            Training target vector.  Automatically ravelled to 1-D.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()       # ensure (N,)
        assert X.ndim == 2,                "X must be 2-D."
        assert X.shape[0] == y.shape[0],   "X and y must have equal N."
        self.n_features_ = X.shape[1]
        self.root = self._grow_tree(X, y, depth=0)
        return self

    def predict(self, X):
        """
        Predict target values for new data.

        Each test point is routed from root to a leaf.  The leaf's
        local linear model produces the prediction:

            y_hat = (x - x_bar_leaf)^T  w_leaf  +  y_bar_leaf

        Predictions are batched: all points are routed together, and
        subsets heading to the same child are grouped at each level.

        Parameters
        ----------
        X : np.ndarray, shape (M, d)
            Test feature matrix.

        Returns
        -------
        y_pred : np.ndarray, shape (M,)
            Predicted target values.
        """
        X = np.asarray(X, dtype=float)
        assert X.ndim == 2 and X.shape[1] == self.n_features_
        M = X.shape[0]
        y_pred = np.empty(M, dtype=float)
        self._predict_batch(X, np.arange(M), self.root, y_pred)
        return y_pred

    # ------------------------------------------------------------------
    # Internal — node statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_node_params(X, y, lam):
        """
        Compute all sufficient statistics and the optimal local linear
        model for a node containing the given data subset.

        Solves:
            w* = (X_c^T X_c + lam I)^{-1}  X_c^T y_c
        and evaluates the minimised loss:
            L* = y_c^T y_c  -  w*^T  X_c^T y_c

        Parameters
        ----------
        X : np.ndarray, shape (N, d)
            Feature matrix (uncentred).
        y : np.ndarray, shape (N,)
            Target vector (uncentred, 1-D).
        lam : float
            Regularisation parameter (lambda > 0).

        Returns
        -------
        x_bar : np.ndarray, shape (d,)
            Feature mean.
        y_bar : float
            Target mean.
        w : np.ndarray, shape (d,)
            Optimal weight vector.
        loss : float
            Minimised regularised loss  L*.
        S_xx : np.ndarray, shape (d, d)
            X_c^T X_c  — centred scatter matrix.
        S_xy : np.ndarray, shape (d,)
            X_c^T y_c  — centred cross-covariance vector.
        S_yy : float
            y_c^T y_c  — centred target sum of squares.
        inv : np.ndarray, shape (d, d)
            (S_xx + lam I)^{-1}.
        """
        N, d = X.shape

        x_bar = np.mean(X, axis=0)                 # (d,)
        y_bar = float(np.mean(y))                   # scalar

        X_c = X - x_bar                             # (N, d)
        y_c = y - y_bar                             # (N,)

        S_xx = X_c.T @ X_c                          # (d, d)
        S_xy = X_c.T @ y_c                          # (d,)
        S_yy = float(y_c @ y_c)                     # scalar

        inv  = np.linalg.inv(S_xx + lam * np.eye(d))  # (d, d)
        w    = inv @ S_xy                              # (d,)
        loss = S_yy - float(w @ S_xy)                  # scalar

        return x_bar, y_bar, w, loss, S_xx, S_xy, S_yy, inv

    # ------------------------------------------------------------------
    # Internal — efficient split search
    # ------------------------------------------------------------------

    def _find_best_split(self, X, y, parent_loss, parent_stats):
        """
        Search every feature and threshold for the split that minimises
        the combined child loss.

        Algorithm (per feature j):
          1. Sort the N samples by x_j.
          2. Initialise LEFT = empty, RIGHT = all N points
             (with pre-computed sufficient statistics).
          3. Sweep sorted order: at each step move one point from RIGHT
             to LEFT.  Update running statistics with Welford-style
             formulas (O(d) for means, O(d^2) for S_xx) and keep the
             matrix inverse current via Sherman-Morrison (O(d^2)).
          4. After each move (if the feature value changes), evaluate
             the candidate split cost  L*_L + L*_R.

        Parameters
        ----------
        X : np.ndarray, shape (N, d)
            Feature matrix at the current node (uncentred).
        y : np.ndarray, shape (N,)
            Target vector at the current node (1-D).
        parent_loss : float
            L* of the unsplit node.
        parent_stats : tuple
            (x_bar, y_bar, S_xx, S_xy, S_yy, inv) — pre-computed
            sufficient statistics of the current node.

        Returns
        -------
        split : dict or None
            ``{"feature": int, "threshold": float, "loss": float}``
            if an improving split exists, otherwise ``None``.
        """
        N, d = X.shape
        lam = self.lam

        (x_bar_all, y_bar_all,
         S_xx_all, S_xy_all, S_yy_all, inv_all) = parent_stats

        best_loss  = parent_loss          # only accept strict improvement
        best_split = None

        for j in range(d):
            # -------- sort by feature j --------
            order = np.argsort(X[:, j])   # (N,) index array
            X_s   = X[order]              # (N, d)  sorted rows
            y_s   = y[order]              # (N,)    sorted targets

            # ---- initialise LEFT partition (empty) ----
            n_L   = 0
            xb_L  = np.zeros(d)
            yb_L  = 0.0
            Sxx_L = np.zeros((d, d))
            Sxy_L = np.zeros(d)
            Syy_L = 0.0
            inv_L = np.eye(d) / lam       # (0 + lam I)^{-1}

            # ---- initialise RIGHT partition (all N points) ----
            n_R   = N
            xb_R  = x_bar_all.copy()
            yb_R  = float(y_bar_all)
            Sxx_R = S_xx_all.copy()
            Sxy_R = S_xy_all.copy()
            Syy_R = float(S_yy_all)
            inv_R = inv_all.copy()

            # ---- sweep from left to right ----
            for i in range(N - 1):
                xi = X_s[i]               # (d,)
                yi = y_s[i]               # scalar

                # =========== ADD point (xi, yi) to LEFT ===========
                dx_L = xi - xb_L          # delta_x  (d,)
                dy_L = yi - yb_L          # delta_y  scalar
                n_L += 1
                xb_L = xb_L + dx_L / n_L
                yb_L = yb_L + dy_L / n_L

                if n_L > 1:
                    alpha_L = (n_L - 1) / n_L           # n_old / n_new
                    Sxx_L += alpha_L * np.outer(dx_L, dx_L)    # (d, d)
                    Sxy_L += alpha_L * dy_L * dx_L             # (d,)
                    Syy_L += alpha_L * dy_L ** 2               # scalar

                    # Sherman-Morrison:  (A + u u^T)^{-1}
                    u_L = np.sqrt(alpha_L) * dx_L              # (d,)
                    v_L = inv_L @ u_L                          # (d,)
                    inv_L = inv_L - np.outer(v_L, v_L) / (1.0 + u_L @ v_L)
                # else: n_L == 1 → S_xx stays 0, inv stays I/lam

                # ========= REMOVE point (xi, yi) from RIGHT =========
                dx_R = xi - xb_R          # delta_x  (d,)
                dy_R = yi - yb_R          # delta_y  scalar
                n_R -= 1

                if n_R > 0:
                    xb_R = xb_R - dx_R / n_R
                    yb_R = yb_R - dy_R / n_R

                    alpha_R = (n_R + 1) / n_R               # n_old / n_new
                    Sxx_R -= alpha_R * np.outer(dx_R, dx_R)
                    Sxy_R -= alpha_R * dy_R * dx_R
                    Syy_R -= alpha_R * dy_R ** 2

                    # Sherman-Morrison:  (A - u u^T)^{-1}
                    u_R = np.sqrt(alpha_R) * dx_R              # (d,)
                    v_R = inv_R @ u_R                          # (d,)
                    inv_R = inv_R + np.outer(v_R, v_R) / (1.0 - u_R @ v_R)

                # ---- skip duplicate feature values ----
                if X_s[i, j] == X_s[i + 1, j]:
                    continue

                # ---- evaluate candidate split ----
                # Left child loss
                w_L    = inv_L @ Sxy_L                         # (d,)
                loss_L = Syy_L - float(w_L @ Sxy_L)           # scalar

                # Right child loss
                w_R    = inv_R @ Sxy_R                         # (d,)
                loss_R = Syy_R - float(w_R @ Sxy_R)           # scalar

                total = loss_L + loss_R

                if total < best_loss:
                    best_loss  = total
                    best_split = {
                        "feature":   j,
                        "threshold": 0.5 * (X_s[i, j] + X_s[i + 1, j]),
                        "loss":      total,
                    }

        return best_split

    # ------------------------------------------------------------------
    # Internal — recursive tree growth
    # ------------------------------------------------------------------

    def _grow_tree(self, X, y, depth):
        """
        Recursively build the tree for the given data partition.

        Stopping criteria (node is finalised as a leaf if ANY holds):
            1. ``max_depth`` is not None and ``depth >= max_depth``.
            2. No candidate split yields  L*_L + L*_R  <  L*_parent .

        Parameters
        ----------
        X : np.ndarray, shape (N, d)
            Feature matrix for the current partition.
        y : np.ndarray, shape (N,)
            Target vector for the current partition (1-D).
        depth : int
            Current depth (root = 0).

        Returns
        -------
        node : Node
            The constructed (sub-)tree rooted at this node.
        """
        N, d = X.shape

        # --- compute node parameters ---
        (x_bar, y_bar, w, loss,
         S_xx, S_xy, S_yy, inv) = self._compute_node_params(X, y, self.lam)

        node       = Node()
        node.x_bar = x_bar        # (d,)
        node.y_bar = y_bar        # scalar
        node.w     = w            # (d,)

        # Stopping criterion 1: maximum depth reached
        if self.max_depth is not None and depth >= self.max_depth:
            return node            # leaf

        # Attempt to find an improving split
        parent_stats = (x_bar, y_bar, S_xx, S_xy, S_yy, inv)
        best = self._find_best_split(X, y, loss, parent_stats)

        # Stopping criterion 2: no improving split
        if best is None:
            return node            # leaf

        # ---- execute the split ----
        j = best["feature"]
        t = best["threshold"]
        mask = X[:, j] <= t                # (N,) boolean

        node.is_leaf   = False
        node.feature   = j
        node.threshold = t
        node.left  = self._grow_tree(X[mask],  y[mask],  depth + 1)
        node.right = self._grow_tree(X[~mask], y[~mask], depth + 1)

        return node

    # ------------------------------------------------------------------
    # Internal — vectorised prediction
    # ------------------------------------------------------------------

    def _predict_batch(self, X, indices, node, y_pred):
        """
        Recursively route a batch of test points through the tree and
        write their predictions into ``y_pred`` in-place.

        At a leaf the vectorised prediction is:

            y_hat  =  (X_sub  -  1 x_bar^T)  w  +  y_bar      (K, d) @ (d,) → (K,)

        Parameters
        ----------
        X : np.ndarray, shape (M, d)
            Full test feature matrix (shared across recursion).
        indices : np.ndarray, shape (K,)
            Row indices into *X* of the points currently at this node.
        node : Node
            Current tree node.
        y_pred : np.ndarray, shape (M,)
            Output vector (modified in-place).
        """
        if len(indices) == 0:
            return

        if node.is_leaf:
            # Vectorised leaf prediction
            X_sub = X[indices]                                  # (K, d)
            y_pred[indices] = (X_sub - node.x_bar) @ node.w + node.y_bar
            return

        # Route to children based on split criterion
        feat_vals = X[indices, node.feature]                    # (K,)
        left_mask = feat_vals <= node.threshold                 # (K,) bool

        self._predict_batch(X, indices[left_mask],  node.left,  y_pred)
        self._predict_batch(X, indices[~left_mask], node.right, y_pred)


# ======================================================================
#  Hyperparameter configuration (Task 2)
# ======================================================================

def give_optimal_hyperparameters():
    """
    Return the tuned hyperparameters for the Segment dataset.

    These values should be determined experimentally by observing
    train/test MSE and the prediction-boundary plots generated by
    ``main.py``.  Hardcode your final choices here.

    Returns
    -------
    max_depth : int
        Optimal maximum tree depth.
    lam : float
        Optimal L2 regularisation strength (lambda).
    """
    max_depth = 3
    lam       = 0.1
    return max_depth, lam
