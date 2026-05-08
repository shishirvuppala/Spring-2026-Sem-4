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
    # TODO
    pass

def get_false_positives(y_true, y_pred):
    """ Count samples where y_true=0 and y_pred=1 
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
    Returns:
        Float/Int
    """
    # TODO
    pass

def get_false_negatives(y_true, y_pred):
    """ Count samples where y_true=1 and y_pred=0 
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
    Returns:
        Float/Int
    """
    # TODO
    pass

def get_true_negatives(y_true, y_pred):
    """ Count samples where y_true=0 and y_pred=0 
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
    Returns:
        Float/Int
    """
    # TODO
    pass

def get_precision(y_true, y_pred):
    """
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
    Returns:
        float: Precision score.
    """
    # TODO
    pass

def get_recall(y_true, y_pred):
    """
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
    Returns:
        float: Recall score.
    """
    # TODO
    pass

def get_f1_score(y_true, y_pred):
    """
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
    Returns:
        float: F1 score.
    """
    # TODO
    pass

def get_multiclass_accuracy(y_true, y_pred):
    """
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
    Returns:
        float: Fraction of correctly classified samples.
    """
    # TODO. Ensure this function works for multi-class as well.
    pass


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
        # TODO: Implement sigmoid (use np.clip to prevent overflow)
        pass

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
        # TODO: Implement weighted binary cross entropy.
        pass

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
        # TODO: Compute weighted gradient
        pass

    def fit(self, X, y, sample_weights):
        """
        Main optimization loop.
        """
        #TODO: Initialize self.w and self.loss_history and any other variables, initialize weights to zero
        pass 

        for i in range(self.max_iter):
            # TODO: Implement the training loop
            
            pass

    def predict_proba(self, X):
        """
        Predict probability of class 1.
        Args:
            X (np.ndarray): Feature matrix (N, D).
        Returns:
            np.ndarray: Probabilities (N,).
        """
        # TODO
        pass


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
        # TODO: Return weights based on self.mode
        
        # If mode is 'naive', return an array of 0.5's
        
        # If mode is 'weighted', calculate importance weights using the scheme listed in PS.

        
        pass

    def fit(self, X, y):
        """
        Trains K binary classifiers.
        
        Args:
            X (np.ndarray): Feature matrix of shape (N, D).
            y (np.ndarray): Multiclass labels of shape (N,).
        """
        classes = np.unique(y)
        #TODO:
        pass
        
        for k in classes:
            #TODO:
            pass

    def predict(self, X):
        """
        Predict class with highest probability.
        
        Args:
            X (np.ndarray): Feature matrix of shape (N, D).
            
        Returns:
            np.ndarray: Predicted class labels of shape (N,).
        """
        # TODO: Iterate over self.logistic_models
        # Return the class with the maximum probability.
        # To get the keys of a dictionary dict.keys() may be helpful... Read the python docs for more
        pass
