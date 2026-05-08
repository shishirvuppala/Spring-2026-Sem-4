import numpy as np
import pandas as pd

class KNN:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Store the training data and labels.
        Parameters:
            X: Training features (numpy array)
            y: Training labels (numpy array)
        """
        self.X_train = X
        self.y_train = y

    def predict_L2(self, X_test, k):
        """
        Predict labels for the test set using L2 (Euclidean) distance.
        Parameters:
            X_test: Test features (numpy array)
            k: Number of neighbors
        Returns:
            y_pred: Predicted labels for X_test (numpy array of +1 or -1)
        """
        # TODO: Implement vectorized L2 distance and majority vote
        pass

    def predict_L1(self, X_test, k):
        """
        Predict labels for the test set using L1 (Manhattan) distance.
        Parameters:
            X_test: Test features (numpy array)
            k: Number of neighbors
        Returns:
            y_pred: Predicted labels for X_test (numpy array of +1 or -1)
        """
        # TODO: Implement vectorized L1 distance and majority vote
        pass

def compute_accuracy(y_true, y_pred):
    """
    Calculate the percentage of correct predictions.
    Parameters:
        y_true: Ground truth labels
        y_pred: Predicted labels
    Returns:
        accuracy: Float representing accuracy
    """
    # TODO: Implement accuracy calculation
    pass

def standardize(X_train, X_test):
    """
    Standardize features to mean 0 and variance 1.
    Parameters:
        X_train: Raw training features
        X_test: Raw test features
    Returns:
        X_train_std, X_test_std: Standardized feature arrays
    """
    # TODO: Standardize X_test using statistics derived ONLY from X_train
    pass

def get_pearson_indices(X, y, m):
    """
    Select top m features based on absolute Pearson correlation with label y.
    Parameters:
        X: Feature array
        y: Label array
        m: Number of features to select
    Returns:
        indices: Array of indices for the top m features
    """
    # TODO: Implement vectorized Pearson correlation and return top m indices
    pass

if __name__ == "__main__":
    # you are allowed to use loops here

    # TODO: Load data using pandas
    
    # TODO: Execute Task A (Vary k, use L2)

    # TODO: Execute Task B (Standardize, then vary m for Pearson selection, use k=20, L2)

    # TODO: Execute Task C (Standardize, use all features, use k=20, L1)
    pass