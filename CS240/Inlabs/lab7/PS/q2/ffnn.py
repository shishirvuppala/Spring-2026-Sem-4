import numpy as np

class FeedForwardNN:
    """
    A Feed-Forward Neural Network implemented from scratch using NumPy.
    """
    def __init__(self, layer_sizes, hidden_activation='relu', output_activation='sigmoid', loss_type='bce', learning_rate=0.01):
        """
        Instance Variables:
            self.layer_sizes (list of int): Stored layer dimensions.
            self.hidden_act (str):  hidden activation type. ('relu' or 'sigmoid')
            self.output_act (str):  output activation type. ('sigmoid' or 'linear')
            self.loss_type (str):  loss function type. ('mse' or 'bce')
            self.learning_rate (float):  learning rate alpha.
            self.L (int): Total number of layers (number of weight matrices).
            self.weights (list of numpy.ndarray): Weight matrices. weights[l] = W for layer l.
            self.biases (list of numpy.ndarray): Bias vectors. biases[l] = b for layer l.
            self.pre_activations (list of numpy.ndarray): Pre-activations z stored during forward pass.
            self.activations (list of numpy.ndarray): Activations a stored during forward pass.
                                                      activations[0] = X (network input).
        """
        self.layer_sizes = layer_sizes
        self.hidden_act = hidden_activation
        self.output_act = output_activation
        self.loss_type = loss_type
        self.learning_rate = learning_rate
        self.L = len(layer_sizes) - 1

        self.weights = []
        self.biases = []
        self.pre_activations = []
        self.activations = []

        self.initialize_parameters()

    def initialize_parameters(self):
        """
        TODO 1: Initialize weights and biases for each layer and append them to
        self.weights and self.biases.
        """
        np.random.seed(42)
        # --- YOUR CODE HERE ---
        pass
        # ----------------------

    def relu(self, z):
        """
        TODO 2a: Implement the ReLU activation.

        Args:
            z (numpy.ndarray): Pre-activation values.

        Returns:
            numpy.ndarray: Activated values.
        """
        # --- YOUR CODE HERE ---
        pass
        # ----------------------

    def relu_derivative(self, z):
        """
        TODO 2b: Implement the derivative of ReLU.

        Args:
            z (numpy.ndarray): Pre-activation values.

        Returns:
            numpy.ndarray: Element-wise derivative values.
        """
        # --- YOUR CODE HERE ---
        pass
        # ----------------------

    def sigmoid(self, z):
        """
        TODO 3a: Implement the Sigmoid activation.

        Args:
            z (numpy.ndarray): Pre-activation values.

        Returns:
            numpy.ndarray: Activated values.
        """
        # --- YOUR CODE HERE ---
        pass
        # ----------------------

    def sigmoid_derivative(self, z):
        """
        TODO 3b: Implement the derivative of Sigmoid.

        Args:
            z (numpy.ndarray): Pre-activation values.

        Returns:
            numpy.ndarray: Element-wise derivative values.
        """
        # --- YOUR CODE HERE ---
        pass
        # ----------------------

    def activate(self, z, activation_type):
        """
        Routes to the correct activation function.

        Args:
            z (numpy.ndarray): Pre-activation values.
            activation_type (str): One of 'relu', 'sigmoid', or 'linear'.

        Returns:
            numpy.ndarray: Activated values.
        """
        if activation_type == 'relu': return self.relu(z)
        elif activation_type == 'sigmoid': return self.sigmoid(z)
        elif activation_type == 'linear': return z
        else: raise ValueError("Unsupported activation")

    def forward_propagation(self, X):
        """
        TODO 4: Implement forward propagation.

        Args:
            X (numpy.ndarray): Input data of shape (num_samples, num_features).

        Returns:
            numpy.ndarray: Final output predictions.
        """
        self.activations = [X]
        self.pre_activations = []

        # --- YOUR CODE HERE ---
        pass
        # ----------------------

    def compute_loss(self, output, y):
        """
        TODO 5: Compute and return the loss (MSE or BCE).

        Args:
            output (numpy.ndarray): Predictions from the forward pass.
            y (numpy.ndarray): True labels/targets.

        Returns:
            float: Scalar loss value.
        """
        # --- YOUR CODE HERE ---
        pass
        # ----------------------

    def backward_propagation(self, y):
        """
        TODO 6: Implement backpropagation.

        Args:
            y (numpy.ndarray): True labels/targets.

        Returns:
            tuple: (grad_weights, grad_biases) where each is a list of gradients,
                   one per layer.
        """
        grad_weights = [np.zeros_like(w) for w in self.weights]
        grad_biases  = [np.zeros_like(b) for b in self.biases]

        # --- YOUR CODE HERE ---
        pass
        # ----------------------

    def update_parameters(self, grad_weights, grad_biases):
        """
        TODO 7: Update weights and biases using gradient descent.

        Args:
            grad_weights (list of numpy.ndarray): Weight gradients per layer.
            grad_biases  (list of numpy.ndarray): Bias gradients per layer.
        """
        # --- YOUR CODE HERE ---
        pass
        # ----------------------


    def train(self, X, y, epochs=10000, print_freq=1000):
        """
        TODO 8: Implement the training loop.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): True labels/targets.
            epochs (int): Number of training iterations.
            print_freq (int): Frequency of printing loss.
        """
        # --- YOUR CODE HERE ---
        pass
        # ----------------------

    def predict(self, X, threshold=0.5):
        """
        TODO 9: Implement the prediction function.

        Args:
            X (numpy.ndarray): Input data.
            threshold (float): Threshold for binary classification.

        Returns:
            numpy.ndarray: Predicted labels or regression outputs.
        """
        # --- YOUR CODE HERE ---
        pass
        # ----------------------
