"""
q1_solve.py
============
Students must implement the following functions and classes:
- load_data
- train_test_split_custom
- OLSClassifier
- LogisticRegressionGD
"""
import numpy as np

# ============================================
# TASK 0: DATA LOADING
# ============================================

def load_data(file_path: str):
    """
    Load dataset from a given CSV file.

    Parameters
    ----------
    file_path : str
        Path to CSV file. Last column should be the label (y in {-1, +1})

    Returns
    -------
    X : np.ndarray
        Features of shape (N, d)
    y : np.ndarray
        Labels of shape (N,), values in {-1, +1}
    """
    # TODO: Implement CSV loading
    data = np.loadtxt(file_path, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


# ============================================
# TASK 0.5: TRAIN TEST SPLIT
# ============================================

def test_train_split(X: np.ndarray, y: np.ndarray, test_size=0.2):
    """
    Split dataset into train and test sets.

    Parameters
    ----------
    X : np.ndarray
        Features (N, d)
    y : np.ndarray
        Labels (N,)
    test_size : float
        Fraction of dataset to assign to test set

    X_train must have shape (N_train, d)
    Where N_train = floor(N * (1 - test_size))
    and N_test = N - N_train

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarrays
        Split datasets
        X_train : (N_train, d)
        X_test : (N_test, d)
        y_train : (N_train,)
        y_test : (N_test,)
    
    DO NOT alter the order of samples in X and y
    """
    # TODO: Split according to test_size
    N = X.shape[0]
    N_train = int(N * (1 - test_size))
    X_train = X[:N_train]
    X_test = X[N_train:]
    y_train = y[:N_train]
    y_test = y[N_train:]
    return X_train, X_test, y_train, y_test



def normalized_test_train_split(X: np.ndarray, y: np.ndarray, test_size=0.2, test_train_split_func=test_train_split):
    """
    Split dataset into train and test sets and normalize features.

    Parameters
    ----------
    X : np.ndarray
        Features (N, d)
    y : np.ndarray
        Labels (N,)
    test_size : float
        Fraction of dataset to assign to test set
    test_train_split_func : function to use for splitting
        Function that takes in X, y, test_size and returns X_train, X_test, y_train, y_test
        
    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarrays
        Split datasets
        X_train : (N_train, d)
        X_test : (N_test, d)
        y_train : (N_train,)
        y_test : (N_test,)
    """
    # TODO
    X_train, X_test, y_train, y_test = test_train_split(X, y, test_size)
    X_train_norm = (X_train - X_train.mean()) / (X_train.std())
    X_test_norm = (X_test - X_test.mean()) / (X_test.std())
    return X_train_norm, X_test_norm, y_train, y_test


# ============================================
# TASK 1: LEAST SQUARES CLASSIFICATION
# ============================================

class OLSClassifier:
    """
    Ordinary Least Squares classifier using gradient descent.
    Predicts labels {-1, +1}.
    lr: learning rate
    max_iter: maximum number of iterations
    tol: tolerance for stopping criterion, (If the norm of the change in weights is less than tol, stop)
    """

    def __init__(self, lr=0.01, max_iter=1000, tol=1e-6):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.w = None


    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the linear model using gradient descent.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (N, d)
        y : np.ndarray
            Labels {-1, +1}
        """
        # TODO: Implement newton's method for least squares
        N = X.shape[0]
        d = X.shape[1]
        self.w = np.array([0] * d)
        for _ in range(self.max_iter):
            gradient = (-2 / N) * (X.T @ (y - X @ self.w))
            hessian = (2 / N) * (X.T @ X)
            w_new = self.w - self.lr * np.linalg.inv(hessian) @ gradient
            if np.linalg.norm(w_new - self.w) < self.tol:
                break
            self.w = w_new

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels {-1, +1}.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (N, d)

        Returns
        -------
        y_pred : np.ndarray 
            Predicted labels (N,)
        """ 
        # TODO: Return the predictied labels
        return np.where(X @ self.w >= 0, 1, -1)

# ============================================
# TASK 2: LOGISTIC REGRESSION
# ============================================

class LogisticRegressionGD:
    """
    Logistic Regression classifier using batch gradient descent.
    """

    def __init__(self, lr=0.01, max_iter=1000, tol=1e-6):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.logistic_loss_history = []


    def logistic_gradient(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute gradient of logistic loss.

        Parameters
        ----------
        w : np.ndarray
            Weights (d,)q
        X : np.ndarray
            Feature matrix (N, d)
        y : np.ndarray
            Labels {-1, +1} (N,)

        Returns
        -------
        grad : np.ndarray
            Gradient of shape (d,)
        """

        # TODO: Compute logistic gradient
        N = X.shape[0]
        margins = y * (X @ w)
        probs = 1 / (1 + np.exp(margins))
        grad = - (1 / N) * (X.T @ (y * probs))
        return grad


    def compute_logistic_loss(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute logistic loss.

        Parameters
        ----------
        w : np.ndarray
            Weights (d,)
        X : np.ndarray
            Feature matrix (N, d)
        y : np.ndarray
            Labels {-1, +1} (N,)

        Returns
        -------
        loss : float
            Logistic loss
        """
        N = X.shape[0]
        margins = y * (X @ w)
        loss = np.sum(np.log(1 + np.exp(-margins))) / N
        return loss
    


    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit logistic regression using batch gradient descent, while adding logistic loss to history.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (N, d)
        y : np.ndarray
            Labels {-1, +1} (N,)
        """


        # TODO: Implement gradient descent for logistic regression
        N = X.shape[0]
        d = X.shape[1]
        self.w = np.array([0] * d)
        for _ in range(self.max_iter):
            grad = self.logistic_gradient(self.w, X, y)
            w_new = self.w - self.lr * grad
            loss = self.compute_logistic_loss(self.w, X, y)
            self.logistic_loss_history.append(loss)
            if np.linalg.norm(w_new - self.w) < self.tol:
                break
            self.w = w_new

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels {-1, +1}
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (N, d)
        
        Returns
        -------
        y_pred : np.ndarray
            Predicted labels (N,)
        """

        # TODO: Implement predict method
        return np.where(X @ self.w >= 0, 1, -1)