import numpy as np

def get_knn_accuracy(X_train, y_train, X_test, y_test, k, metric='L2', method='broadcasting'):
    N, D = X_test.shape
    M = X_train.shape[0]
    
    if method == 'expansion':
        if metric == 'L2':
            # Space: O(NM) - Only possible for L2
            dot_product = np.dot(X_test, X_train.T)
            test_sq = np.sum(X_test**2, axis=1, keepdims=True)
            train_sq = np.sum(X_train**2, axis=1)
            dists = np.sqrt(np.maximum(0, test_sq + train_sq - 2 * dot_product))
        else:
            raise ValueError("Expansion trick is only mathematically valid for L2 metric.")
        
    
    #the next few methods are tradeoffs, the standard for method uses much lesser space, but is slower, try to observe this tradeoff by playing
    #around with the sizes of the array to check this
    elif method == 'broadcasting':
        # Space: O(NMD), faster by stride allocation, please read up on how broadcasting works, is extremely useful
        ord_val = 2 if metric == 'L2' else 1
        dists = np.linalg.norm(X_test[:, np.newaxis, :] - X_train, ord=ord_val, axis=2)

    elif method == 'tiled':
        # Space: O(NMD) with explicit memory allocation
        X_test_3d = np.tile(X_test[:, np.newaxis, :], (1, M, 1))
        X_train_3d = np.tile(X_train[np.newaxis, :, :], (N, 1, 1))
        ord_val = 2 if metric == 'L2' else 1
        dists = np.linalg.norm(X_test_3d - X_train_3d, ord=ord_val, axis=2)

    nn_indices = np.argsort(dists, axis=1)[:, :k]
    nn_labels = y_train[nn_indices]
    
    predictions = np.sign(np.sum(nn_labels, axis=1))
    predictions[predictions == 0] = 1 
    
    return np.mean(predictions == y_test)


#the rest of the code for the experiments can be implemented on your own, just write the main function to run the experiments with hyperparameterization