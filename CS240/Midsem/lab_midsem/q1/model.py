import numpy as np


class LogisticModel:
    """
    Logistic classifier with ℓ4 regularization.

        f(x) = w^T x + b
    """

    def __init__(self, d, lam=0.1, lr=0.01):
        """
        Inputs:
            d   : number of features
            lam : regularization strength
            lr  : learning rate
        """
        self.w = np.zeros(d)
        self.b = 0.0
        self.lam = lam
        self.lr = lr

        self.mean = None
        self.std = None

    def fit_standardization(self, X: np.ndarray):
        """
        Inputs:
            X : (n, d) training data

        Compute feature mean and std.

        Returns Nothing.
        """
        # TODO
        pass

    def transform(self, X: np.ndarray):
        """
        Inputs:
            X : (n, d)

        Returns:
            X_standardized: np.ndarray: (n,d) 
            The standardized features. Subtract mean and divide by standard deviation for each column.
        """
        # TODO
        pass


    def predict(self, X: np.ndarray):
        """
        Inputs:
            X : (n, d) : Not standardized.

        Returns:
            y_pred : np.ndarray: (n,)
            predicted labels are in {+1, -1}
        """
        # TODO
        pass

    def loss_batch(self, X: np.ndarray, y: np.ndarray):
        """
        Inputs:
            X : standardized batch features (n,d)
            y : Array (n,) of labels in {+1, -1}

        Returns:
            regularized logistic loss (As defined in problem statement) : float
        """
        # TODO
        pass

    def gradients(self, X: np.ndarray, y: np.ndarray):
        """
        Inputs:
            X : standardized batch features (n,d)
            y : labels (n,)

        Returns:
            gradient w.r.t w and b
            grad1: np.ndarray: (d,) The gradient of loss with respect to w
            grad2: float : The gradient of loss with respect to b.
            Actually Returns: A tuple containing grad1 and grad2
                (grad1, grad2)
        """
        # TODO
        pass

    def step(self, X: np.ndarray, y: np.ndarray):
        """
        Performs one gradient update.
        Inputs:
            X : standardized batch features (n,d)
            y: labels (n,) in {-1, 1}
        """
        # TODO
        pass


def train(model: LogisticModel, X: np.ndarray, y: np.ndarray, batch_size=64, epochs=40):
    """
    Inputs:
        model      : LogisticModel
        X          : training features : np.ndarray (n,d)
        y          : training labels : np.ndarray (n,)
        batch_size : mini-batch size
        epochs     : number of passes

    Train using mini-batch gradient descent.
    """

    # TODO: fit standardization
    # TODO: transform training data

    n = X.shape[0]

    rng = np.random.RandomState(0)   # keep deterministic

    for _ in range(epochs):
        perm = rng.permutation(n)
        #TODO perform batch gradient descent from here, dont change the perm here, we need it for autograding
        X_shuffle = X[perm]
        y_shuffle = y[perm]

        # On every batch derived from X_shuffle and y_shuffl run a step of your model


def accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Inputs:
        y_true : true labels : np.ndarray : (n,)
        y_pred : predicted labels : np.ndarray : (n,)

    Returns:
        classification accuracy
    """
    # TODO
    pass


def give_optimal_hyperparameter(Xtr: np.ndarray, ytr: np.ndarray, Xval: np.ndarray, yval: np.ndarray):
    """
    Inputs:
        Xtr, ytr   : training data : (N_train, d) and (N_train,)
        Xval, yval : validation data : (N_val, d) and (N_val,)

    Returns:
        best λ based on validation performance
    """
    lambdas = np.arange(0, 50, 1) # sweep within this range

    best_lam = None

    for lam in lambdas:
        np.random.seed(0)   # ensures fair comparison

        # TODO:
        # 1. create model with lam
        # 2. train on training data
        # 3. evaluate on validation data
        # 4. track best λ

        pass

    return best_lam