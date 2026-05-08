"""
classifiers.py
==============
Lab 5, Q1: Implement LDA (Generative) and Logistic Regression (Discriminative)
           classifiers, plus a Hybrid warm-start strategy.

Only NumPy is allowed. Do NOT import any other libraries.
"""

import numpy as np
from typing import Optional, Tuple, Dict, List


# ============================================
# TASK 1: LINEAR DISCRIMINANT ANALYSIS (LDA)
# ============================================

class LDAClassifier:
    """
    Linear Discriminant Analysis (Binary) with shared covariance.

    Attributes:
        mu_0     (np.ndarray): Mean vector for class 0, shape (D,).
        mu_1     (np.ndarray): Mean vector for class 1, shape (D,).
        sigma    (np.ndarray): Pooled (shared) covariance matrix, shape (D, D).
        pi_0     (float):      Prior probability of class 0.
        pi_1     (float):      Prior probability of class 1.
        w        (np.ndarray): Weight vector derived from LDA, shape (D,).
        b        (float):      Bias (intercept) derived from LDA.
    """

    def __init__(self) -> None:
        self.mu_0: Optional[np.ndarray] = None
        self.mu_1: Optional[np.ndarray] = None
        self.sigma: Optional[np.ndarray] = None
        self.pi_0: Optional[float] = None
        self.pi_1: Optional[float] = None
        self.w: Optional[np.ndarray] = None
        self.b: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Estimate LDA parameters from the training data and derive linear weights.

        This method should:
          1. Compute class priors (pi_0, pi_1).
          2. Compute class means (mu_0, mu_1).
          3. Compute the pooled covariance matrix (sigma).
          4. Derive the weight vector w and bias b.

        Args:
            X (np.ndarray): Feature matrix of shape (N, D).
            y (np.ndarray): Binary label vector of shape (N,). Values in {0, 1}.

        Returns:
            None (stores results in self.mu_0, self.mu_1, self.sigma,
                  self.pi_0, self.pi_1, self.w, self.b).

        Hint:
            - Use np.linalg.inv or np.linalg.pinv (pseudo-inverse) for matrix
              inversion. If the covariance matrix is poorly conditioned (nearly
              singular), pinv is more numerically stable.
        """
        # TODO: Implement parameter estimation and weight derivation


        # --- 1. Class priors ---

        # --- 2. Class means ---

        # --- 3. Pooled covariance ---

        # --- 4. Derive linear weights ---
        # NUMERICAL STABILITY NOTE:
        #   np.linalg.pinv is safer than np.linalg.inv when the covariance
        #   matrix is near-singular. For well-conditioned data inv is fine.

        # NUMERICAL STABILITY NOTE:
        #   When pi_0 or pi_1 is very small, log(pi_1/pi_0) could overflow.
        #   For balanced datasets this is not an issue. For extremely imbalanced
        #   data, consider np.log(pi_1 + 1e-15) - np.log(pi_0 + 1e-15).

        # END TODO

    def get_params(self) -> Tuple[np.ndarray, float]:
        """
        Return the LDA-derived weight vector and bias.

        Returns:
            Tuple[np.ndarray, float]:
                - w (np.ndarray): Weight vector of shape (D,).
                - b (float): Scalar bias.
        """
        return self.w, self.b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict P(y=1 | x) for each sample using the LDA linear model.

        The posterior is:  sigma(w^T x + b)  where sigma is the sigmoid function.

        Args:
            X (np.ndarray): Feature matrix of shape (N, D).

        Returns:
            np.ndarray: Predicted probabilities of shape (N,), values in [0, 1].

        Hint:
            - NUMERICAL STABILITY: np.clip(z, -500, 500) before computing
              exp(-z) prevents overflow in the exponential.
        """
        # TODO: Implement

        # END TODO

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary class labels (0 or 1) using a 0.5 threshold.

        Args:
            X (np.ndarray): Feature matrix of shape (N, D).

        Returns:
            np.ndarray: Predicted labels of shape (N,). Values in {0, 1}.
        """
        # TODO: Implement

        # END TODO


# ============================================
# TASK 2: LOGISTIC REGRESSION (Mini-Batch GD)
# ============================================

class LogisticRegression:
    """
    Binary Logistic Regression trained with Mini-Batch Gradient Descent.

    Attributes:
        lr        (float):  Learning rate (eta).
        max_iter  (int):    Number of epochs.
        batch_size(int):    Mini-batch size. Use full-batch if batch_size >= N.
        w         (np.ndarray): Weight vector of shape (D,).
        b         (float):      Bias (intercept).
        loss_history (List[float]): Training loss at the END of each epoch.
    """

    def __init__(
        self,
        lr: float = 0.01,
        max_iter: int = 1000,
        batch_size: int = 64,
    ) -> None:
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.w: Optional[np.ndarray] = None
        self.b: Optional[float] = None
        self.loss_history: List[float] = []

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Numerically stable sigmoid function.

        Args:
            z (np.ndarray): Input array of any shape.

        Returns:
            np.ndarray: Output array of the same shape, values in (0, 1).

        NUMERICAL STABILITY NOTE:
            For large positive z, exp(-z) underflows to 0 → sigmoid → 1 (fine).
            For large negative z, exp(-z) overflows → Inf.
            Strategy: clip z to [-500, 500] BEFORE computing exp to avoid overflow.
            Alternatively use the piecewise formula:
                sigmoid(z) = exp(z)/(1+exp(z))  for z < 0
                             1/(1+exp(-z))       for z >= 0
        Hint:
            - np.clip is your friend here.
        """
        # TODO: Implement


        # END TODO

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Binary Cross-Entropy (Negative Log-Likelihood) loss.

        J = -(1/N) * sum[ y*log(h) + (1-y)*log(1-h) ]

        Args:
            y_true (np.ndarray): True labels of shape (N,). Values in {0, 1}.
            y_pred (np.ndarray): Predicted probabilities of shape (N,), in (0, 1).

        Returns:
            float: Scalar loss value.

        NUMERICAL STABILITY NOTE:
            log(0) = -inf, which will corrupt the loss.
            Add a small epsilon (1e-9) before taking the log:
                np.log(y_pred + eps)   and   np.log(1 - y_pred + eps)
            Alternatively, use np.clip(y_pred, eps, 1 - eps).

        """
        # TODO: Implement

        # Clip predictions to avoid log(0)

        # END TODO

    def compute_gradient(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute gradients of the BCE loss w.r.t. weights w and bias b.

        dJ/dw = (1/N) * X^T @ (y_pred - y_true)    shape: (D,)
        dJ/db = (1/N) * sum(y_pred - y_true)         shape: scalar

        Args:
            X      (np.ndarray): Feature matrix of shape (N, D).
            y_true (np.ndarray): True labels of shape (N,).
            y_pred (np.ndarray): Predicted probabilities of shape (N,).

        Returns:
            Tuple[np.ndarray, float]:
                - grad_w (np.ndarray): Gradient w.r.t. w, shape (D,).
                - grad_b (float):      Gradient w.r.t. b, scalar.

        Hint:
            - The error vector is (y_pred - y_true) of shape (N,).
        """
        # TODO: Implement

        # END TODO

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        init_params: Optional[Tuple[np.ndarray, float]] = None,
    ) -> None:
        """
        Train the logistic regression model using Mini-Batch Gradient Descent.

        Steps per epoch:
          1. Shuffle the data (use a fresh permutation each epoch).
          2. Iterate over mini-batches.
          3. For each mini-batch: compute predictions, gradient, update params.
          4. After processing ALL mini-batches, compute the loss on the FULL
             training set and append it to self.loss_history.

        Args:
            X (np.ndarray): Feature matrix of shape (N, D).
            y (np.ndarray): Label vector of shape (N,). Values in {0, 1}.
            init_params (Optional[Tuple[np.ndarray, float]]):
                If None  → initialise w ~ N(0,1) of shape (D,), b = 0.0.
                           Use np.random.seed(42) before sampling for
                           reproducibility.
                If given → (w_init, b_init). Use these as the starting point.
                           w_init has shape (D,), b_init is a float.

        Returns:
            None (modifies self.w, self.b, self.loss_history in place).

        Hint:
            - np.random.permutation(N) gives a shuffled index array.
            - Slice indices for batch i: idx[i*bs : (i+1)*bs].
            - After the inner loop over batches, compute predictions on the
              FULL training set to get the epoch loss.
        """
        # TODO: Implement


        # --- Initialisation ---

            # Shuffle

            # Mini-batch iteration

                # Forward pass on batch

                # Gradient on batch

                # Parameter update

            # Epoch loss on FULL training set

        # END TODO

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict P(y=1 | x) for each sample.

        Args:
            X (np.ndarray): Feature matrix of shape (N, D).

        Returns:
            np.ndarray: Predicted probabilities of shape (N,).
        """
        # TODO: Implement

        # END TODO

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary class labels {0, 1} using a 0.5 threshold.

        Args:
            X (np.ndarray): Feature matrix of shape (N, D).

        Returns:
            np.ndarray: Predicted labels of shape (N,). Values in {0, 1}.
        """
        # TODO: Implement

        # END TODO
