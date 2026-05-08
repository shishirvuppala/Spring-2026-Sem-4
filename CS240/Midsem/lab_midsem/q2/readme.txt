    ================================================================================
  Lab Midsem — Q2: Regression Trees: Local Linear Estimators
================================================================================

OVERVIEW
--------
You will implement and debug a variant of the CART Regression Tree in which
every node fits a LOCAL LINEAR (Ridge) model instead of predicting a constant
mean.  Given a partition D_i of the training data, the prediction at any node
is the mean-centred linear model:

    y_hat  =  (x - x_bar_i)^T  w_i  +  y_bar_i

where w_i is obtained by solving the regularised normal equations:

    w_i  =  (X_c^T X_c  +  lam * I)^{-1}  X_c^T y_c

Splitting is performed efficiently using Welford-style online updates for the
sufficient statistics (S_xx, S_xy, S_yy) and Sherman-Morrison rank-1 updates
to maintain the matrix inverse, avoiding a full O(d^3) solve at each step.

The experiment is evaluated on the Segment dataset — a 2-D synthetic terrain
surface with four piecewise-linear quadrants.  The goal is to see how well
the local-linear CART recovers these boundaries and to tune max_depth / lambda.

FILES
-----
  tree.py        — YOUR CODE GOES HERE (bugs to fix, hyperparams to tune)
  main.py        — Training / evaluation / plotting pipeline (DO NOT MODIFY)
  debug_check.py — Sanity-check harness against pre-computed references
                   (DO NOT MODIFY)
  utils.py       — Evaluation metrics and tree-inspection helpers
                   (DO NOT MODIFY)
  data/
    segment/     — train.csv, test.csv
  tests/         — Pre-generated reference test cases for debug_check.py
                   (DO NOT MODIFY)

RULES
-----
  * Only NumPy and the Python standard library are allowed.
    Do NOT import sklearn, scipy, or any other third-party library.
  * Use vectorised NumPy operations wherever possible.
  * Follow the exact function signatures, return types, and array shapes
    documented in the docstrings.  The autograder checks these strictly.
  * Do NOT modify main.py, debug_check.py, utils.py, setup_data.py, or
    anything inside data/ or tests/.

================================================================================
  TASK 1 — The Buggy Ride  (fix tree.py)
================================================================================

The file tree.py contains a complete skeleton of the LinearRegressionTree
class.  It has been deliberately seeded with logical / mathematical errors
across the following methods.  Your job is to identify and correct every bug
so that python debug_check.py passes all test cases.

--- tree.py  (LinearRegressionTree class) ---

  CLASS  Node
    A data-class holding the local linear model (w, x_bar, y_bar) and
    split metadata (feature, threshold, left, right) for one tree node.
    You should NOT need to modify this.

  METHOD  _compute_node_params(X, y, lam)  ->  (x_bar, y_bar, w, loss,
                                                 S_xx, S_xy, S_yy, inv)
    Compute the empirical means, mean-centred sufficient statistics, and
    the optimal Ridge weight vector for the data at one node.

  METHOD  _find_best_split(X, y, parent_loss, parent_stats)
    For each feature j, sort by x_j and sweep thresholds in order.
    Maintain running LEFT and RIGHT statistics with online update rules:

  METHOD  _grow_tree(X, y, depth)  ->  Node
    Recursively grow the tree.  Stop and return a leaf when:
      (a)  max_depth is not None  AND  depth >= max_depth
      (b)  _find_best_split returns None (no improving split found)


  METHOD  _predict_batch(X, indices, node, y_pred)
    Recursively route a batch of test points through the tree.
    At a leaf (vectorised):
        y_pred[indices]  =  (X[indices] - node.x_bar) @ node.w + node.y_bar
    At an internal node, split indices on node.feature / node.threshold
    and recurse to node.left and node.right.

DEBUGGING STRATEGY
------------------
  1. Read the mathematical specification in the problem sheet carefully.
     Write down expected shapes before touching the code.
  2. Go through each method line-by-line and compare with the equations.
     Common bug categories: wrong sign in update rule, wrong axis for mean/
     matmul, missing factor (n_old/n_new vs n_new/n_old), wrong Sherman-
     Morrison formula (ADD vs REMOVE confusion), wrong stopping condition.
  3. Run  python debug_check.py  after each fix attempt to see progress.

================================================================================
  TASK 2 — Split Segments  (tune hyperparameters)
================================================================================

Once tree.py is bug-free (debug_check.py passing), evaluate it on the
Segment dataset and tune the two hyperparameters to minimise test MSE:

  max_depth : int
      Controls tree complexity.  Too shallow → underfits.
              Too deep   → marginal gain (noise floor reached by depth 3+
              for this 4-quadrant dataset).

  lam : float (> 0)
      L2 regularisation on the local Ridge weights.  Too large → weights
      shrink to zero, tree degenerates to constant-mean CART.
              Too small  → weights overfit noise within each partition.

HOW TO TUNE
-----------
  1. Run  python main.py
       This uses the values returned by  give_optimal_hyperparameters()
       and generates five diagnostic plots into  plots/ :
         fig1_surface_comparison.png   — True vs predicted terrain (2-D map)
         fig2_prediction_diagnostics.png — Predicted vs Actual, Residual plot
  3. Update  give_optimal_hyperparameters()  at the bottom of tree.py
     with your chosen values and re-run main.py to confirm.

GRADING — quantitative MSE on unseen public and private test splits.

================================================================================
  HOW TO TEST
================================================================================
  1. Fix bugs in tree.py (_compute_node_params, _find_best_split,
                          _grow_tree, _predict_batch).
  2. Run  python debug_check.py   — must show all cases PASS.
  3. Run  python main.py          — inspect plots and printed metrics,
                                    then update give_optimal_hyperparameters().
================================================================================
