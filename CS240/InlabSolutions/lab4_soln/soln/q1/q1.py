"""
q1.py
=====
Task: Implement Naive and Weighted Logistic Regression.
      Handle Covariate Shift where Train and Test distributions differ.
"""

import numpy as np

# ============================================
# PART 1: METRICS
# ============================================

def get_true_positives(y_true, y_pred):
    """ Count samples where y_true=1 and y_pred=1 
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
    Returns:
        Float/Int
    """
    return np.sum((y_true == 1) & (y_pred == 1))

def get_false_positives(y_true, y_pred):
    """ Count samples where y_true=0 and y_pred=1 
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
    Returns:
        Float/Int
    """
    return np.sum((y_true == 0) & (y_pred == 1))

def get_false_negatives(y_true, y_pred):
    """ Count samples where y_true=1 and y_pred=0 
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
    Returns:
        Float/Int
    """
    return np.sum((y_true == 1) & (y_pred == 0))

def get_true_negatives(y_true, y_pred):
    """ Count samples where y_true=0 and y_pred=0 
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
    Returns:
        Float/Int
    """
    return np.sum((y_true == 0) & (y_pred == 0))

def get_precision(y_true, y_pred):
    """
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
    Returns:
        float: Precision score.
    """
    tp = get_true_positives(y_true, y_pred)
    fp = get_false_positives(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def get_recall(y_true, y_pred):
    """
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
    Returns:
        float: Recall score.
    """
    tp = get_true_positives(y_true, y_pred)
    fn = get_false_negatives(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def get_f1_score(y_true, y_pred):
    """
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
    Returns:
        float: F1 score.
    """
    precision = get_precision(y_true, y_pred)
    recall    = get_recall(y_true, y_pred)
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

def get_multiclass_accuracy(y_true, y_pred):
    """
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
    Returns:
        float: Fraction of correctly classified samples.
    """
    # Ensure this function works for multi-class as well.
    return np.mean(y_true == y_pred)


# ============================================
# PART 2: THE CLASS-WEIGHTED BINARY CLASSIFIER 
# ============================================

class BaseLogisticClassifier:
    """
    The Base Engine. Implements Gradient Descent.
    Sample weights are passed directly to fit().
    """
    def __init__(self, lr=0.5, max_iter=4000, tol=1e-6):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.loss_history = []

    def sigmoid(self, z):
        """
        Args:
            z (np.ndarray): Input array.
        Returns:
            np.ndarray: Values between 0 and 1.
        """
        z = np.clip(z, -500, 500)  # prevent overflow in exp
        return 1.0 / (1.0 + np.exp(-z))

    def compute_loss(self, y_true, y_pred, sample_weights):
        """
        Computes the Weighted Binary Cross-Entropy Loss.
        
        Args:
            y_true (np.ndarray): True labels (0 or 1). Shape (N,).
            y_pred (np.ndarray): Predicted probabilities. Shape (N,).
            sample_weights (np.ndarray): Weights for each sample. Shape (N,).
            
        Returns:
            float: The weighted average loss.
        """
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)  # avoid log(0)
        per_sample_loss = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        return -np.sum(sample_weights * per_sample_loss) / np.sum(sample_weights)

    def compute_gradient(self, X, y_true, y_pred, sample_weights):
        """
        Computes the weighted gradient of the loss with respect to weights w.
        
        Args:
            X (np.ndarray): Feature matrix (N, D).
            y_true (np.ndarray): True labels (N,).
            y_pred (np.ndarray): Predicted probabilities (N,).
            sample_weights (np.ndarray): Sample weights (N,).
            
        Returns:
            np.ndarray: Gradient vector of shape (D,).
        """

        error = y_pred - y_true  # shape (N,)
        return (sample_weights * error) @ X / np.sum(sample_weights)

    def fit(self, X, y, sample_weights):
        """
        Main optimization loop.
        """
        self.w = np.zeros(X.shape[1])
        self.loss_history = []

        for i in range(self.max_iter):
            y_pred = self.predict_proba(X)
            loss   = self.compute_loss(y, y_pred, sample_weights)
            self.loss_history.append(loss)

            grad  = self.compute_gradient(X, y, y_pred, sample_weights)
            self.w = self.w - self.lr * grad

            if np.linalg.norm(grad) < self.tol:
                break

    def predict_proba(self, X):
        """
        Predict probability of class 1.
        Args:
            X (np.ndarray): Feature matrix (N, D).
        Returns:
            np.ndarray: Probabilities (N,).
        """
        return self.sigmoid(X @ self.w)


# ============================================
# PART 3: MULTI-CLASS WRAPPER (ONE-VS-REST)
# ============================================

class OneVsRestClassifier:
    """
    Single Class to handle both Naive and Weighted Multi-class Strategies.
    """
    def __init__(self, mode='naive', lr=0.5, max_iter=4000, test_ratios=None):
        """
        Args:
            mode (str): 'naive' or 'weighted'.
            lr (float): Learning rate.
            max_iter (int): Maximum iterations.
            test_ratios (dict): The target distribution for the 'weighted' mode.
                                Example: {0: 0.33, 1: 0.33, 2: 0.33}
        """
        self.mode = mode
        self.lr = lr
        self.max_iter = max_iter
        self.test_ratios = test_ratios
        self.logistic_models = {} #A dictionary with Key: class label (int), Value: an object of type BaseLogisticClassifier

    def _get_binary_weights(self, y_binary, class_label):
        """
        Helper to calculate sample weights for a specific binary problem.
        
        Args:
            y_binary (np.ndarray): Binary targets (0 or 1) for the current class.
            class_label (int/str): The current class label being trained.
            
        Returns:
            np.ndarray: Array of weights, one for each sample. shape (N,)
        """
        if self.mode == 'naive':
            return np.full(len(y_binary), 0.5)

        # Weighted mode: importance weights to correct for prior probability shift
        N = len(y_binary)

        # Training priors (empirical)
        train_prior_pos = np.sum(y_binary == 1) / N  # P_train(y=k)
        train_prior_neg = 1 - train_prior_pos         # P_train(y!=k)

        # Test priors from the provided ratios
        test_prior_pos = self.test_ratios[class_label]  # P_test(y=k)
        test_prior_neg = 1 - test_prior_pos              # P_test(y!=k)

        # Importance weight = P_test / P_train for each sample's class
        weight_pos = test_prior_pos / train_prior_pos if train_prior_pos > 0 else 1.0
        weight_neg = test_prior_neg / train_prior_neg if train_prior_neg > 0 else 1.0

        weights = np.where(y_binary == 1, weight_pos, weight_neg)
        return weights

    def fit(self, X, y):
        """
        Trains K binary classifiers.
        
        Args:
            X (np.ndarray): Feature matrix of shape (N, D).
            y (np.ndarray): Multiclass labels of shape (N,).
        """
        classes = np.unique(y)

        for k in classes:
            y_binary = (y == k).astype(int)
            weights  = self._get_binary_weights(y_binary, k)

            model = BaseLogisticClassifier(lr=self.lr, max_iter=self.max_iter)
            model.fit(X, y_binary, weights)
            self.logistic_models[k] = model

    def predict(self, X):
        """
        Predict class with highest probability.
        
        Args:
            X (np.ndarray): Feature matrix of shape (N, D).
            
        Returns:
            np.ndarray: Predicted class labels of shape (N,).
        """
        class_labels = list(self.logistic_models.keys())
        probs = np.array([self.logistic_models[k].predict_proba(X) for k in class_labels]).T
        best_class_idx = probs.argmax(axis=1)
        return np.array(class_labels)[best_class_idx]
