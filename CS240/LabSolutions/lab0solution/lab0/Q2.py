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

        X_test = X_test[:,None,:]
        diff = (X_test - self.X_train)**2
        dist = np.sqrt(np.sum(diff,axis=2))

        parti = np.argpartition(dist,k-1,axis=1)[:,:k]

        labels = np.sum(self.y_train[parti],axis=1)
        labels =  (labels > 0).astype(int) * 2 -1

        return labels
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

        X_test = X_test[:,None,:]
        diff = np.abs(X_test - self.X_train)
        dist = np.sum(diff,axis=2)

        parti = np.argpartition(dist,k-1,axis=1)[:,:k]

        labels = np.sum(self.y_train[parti],axis=1)
        labels =  (labels > 0).astype(int) * 2 -1

        return labels
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
    lol = (y_true == y_pred)
    return np.sum(lol.astype(int))/(y_true.shape[0])

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
    mu = np.mean(X_train,axis=0)
    stdv = np.std(X_train,axis=0)
    return (X_test - mu)/stdv

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
    Xmean = np.mean(X,axis = 0)
    ymean = np.mean(y,axis = 0)
    yn = y - ymean
    Xn = X - Xmean
    Xstdev = np.std(X,axis = 0)
    Ystdev = np.std(y,axis = 0)
    all = (yn[:,None]*Xn)/(Xstdev*Ystdev)

    rvals = np.abs(np.sum(all,axis=0))

    return np.argpartition(rvals,-m)[-m:]
    # N,d X and N,1 Y


    pass

if __name__ == "__main__":
    # you are allowed to use loops here

    # TODO: Load data using pandas

    tests = pd.read_csv("q2_test.csv")
    trains = pd.read_csv("q2_train.csv")

    X_test = tests.iloc[:,:-1].to_numpy()
    Y_test = tests.iloc[:,-1].to_numpy()

    X_train = trains.iloc[:,:-1].to_numpy()
    Y_train = trains.iloc[:,-1].to_numpy()
    
    # TODO: Execute Task A (Vary k, use L2)
    knn = KNN()
    knn.fit(X_train,Y_train)
    
    kvals = [1,2,5,10,100]
    for k in kvals:
        y_pred = knn.predict_L2(X_test,k)
        acc = compute_accuracy(Y_test,y_pred)
        print("when k = ",k," acc = ",acc)

    # TODO: Execute Task B (Standardize, then vary m for Pearson selection, use k=20, L2)


    X_test_nor = standardize(X_train,X_test)
    X_train_nor = standardize(X_train,X_train)

    

    mvals = [30,50,70,100,150,200]
    for m in mvals:
        indice = get_pearson_indices(X_train_nor,Y_train,m)
        X_train_nor1 = X_train_nor[:,indice]
        X_test_nor1 = X_test_nor[:,indice]
        
        knnB = KNN()
        knnB.fit(X_train_nor1,Y_train)
        y_pred = knnB.predict_L2(X_test_nor1,20)
        acc = compute_accuracy(Y_test,y_pred)
        print("TASK B: when m = ",m," k = 20 acc = ",acc)


    # TODO: Execute Task C (Standardize, use all features, use k=20, L1)
    knnC = KNN()
    knnC.fit(X_train_nor,Y_train)
    y_pred = knn.predict_L1(X_test_nor,20)
    acc = compute_accuracy(Y_test,y_pred)
    print("TASK C: when k = 20 acc = ",acc)
    pass