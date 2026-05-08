import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

from ffnn import FeedForwardNN 

def run_binary_classification():
    print("=========================================")
    print("TASK 1: BINARY CLASSIFICATION (Two Moons)")
    print("=========================================\n")
    
    #load data
    data = np.genfromtxt('two_moons_data.csv', delimiter=',', skip_header=1)
    X = data[:, :2]
    Y = data[:, 2]
    Y = Y.reshape(-1, 1)

    #build model
    classifier = FeedForwardNN(
        layer_sizes=[2, 32, 32, 1], 
        hidden_activation='relu', 
        output_activation='sigmoid', 
        loss_type='bce', 
        learning_rate=0.1
    )

    #train
    classifier.train(X, Y, epochs=5000, print_freq=1000)

    #Evaluate & Plot
    predictions = classifier.predict(X)
    accuracy = np.mean(predictions == Y) * 100
    print(f"\nFinal Classification Accuracy: {accuracy:.2f}%")

    # Plotting Decision Boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=Y.ravel(), cmap=plt.cm.Spectral, edgecolors='k')
    plt.title("Binary Classification Decision Boundary")
    plt.show()

def run_deep_regression():
    print("\n=========================================")
    print("TASK 2: DEEP NEURAL NETWORK REGRESSION")
    print("=========================================\n")
    
    # Generate Dataset (Noisy Sine Wave)
    X = np.random.uniform(-3, 3, (400,1))
    X = np.sort(X, axis=0)
    Y = np.sin(X) + np.random.randn(*X.shape) * 0.15

    # Build tthe model
    regressor = FeedForwardNN(
        layer_sizes=[1, 32, 32, 32, 1], 
        hidden_activation='relu', 
        output_activation='linear', 
        loss_type='mse', 
        learning_rate=0.01
    )

    # Train
    regressor.train(X, Y, epochs=15000, print_freq=3000)

    # Evaluate & Plot
    predictions = regressor.predict(X)
    mse = np.mean((predictions - Y)**2)
    print(f"\nFinal Regression Mean Squared Error: {mse:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, color='blue', alpha=0.5, label='Actual Data (Noisy)')
    plt.plot(X, predictions, color='red', linewidth=3, label='NN Prediction Curve')
    plt.title("Deep Neural Network Regression Fit")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_binary_classification()
    run_deep_regression()