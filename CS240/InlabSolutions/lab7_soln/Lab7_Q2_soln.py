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
        for l in range(self.L):
            fan_in = self.layer_sizes[l]
            fan_out = self.layer_sizes[l + 1]
            # He initialization (good for ReLU); works fine for sigmoid too
            W = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, fan_out))
            self.weights.append(W)
            self.biases.append(b)
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
        return np.maximum(0, z)
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
        return (z > 0).astype(float)
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
        # Clip to avoid overflow in exp
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))
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
        s = self.sigmoid(z)
        return s * (1.0 - s)
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
        for l in range(self.L):
            z = self.activations[l] @ self.weights[l] + self.biases[l]
            self.pre_activations.append(z)
            # Use hidden activation for all layers except the last
            act_type = self.output_act if l == self.L - 1 else self.hidden_act
            a = self.activate(z, act_type)
            self.activations.append(a)

        return self.activations[-1]
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
        m = y.shape[0]
        if self.loss_type == 'mse':
            return np.sum((output - y) ** 2) / (2 * m)
        elif self.loss_type == 'bce':
            output_clipped = np.clip(output, 1e-12, 1 - 1e-12)
            return -np.sum(y * np.log(output_clipped) + (1 - y) * np.log(1 - output_clipped)) / m
        else:
            raise ValueError("Unsupported loss type")
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
        m = y.shape[0]
        y_hat = self.activations[-1]

        # Output delta: works for MSE+linear and BCE+sigmoid
        delta = y_hat - y  # shape (m, output_size)

        for l in reversed(range(self.L)):
            grad_weights[l] = (self.activations[l].T @ delta) / m
            grad_biases[l]  = np.sum(delta, axis=0, keepdims=True) / m

            if l > 0:
                # Backpropagate delta to previous layer
                delta = (delta @ self.weights[l].T)
                # Apply derivative of hidden activation at layer l-1
                act_type = self.hidden_act  # all non-output layers use hidden_act
                if act_type == 'relu':
                    delta = delta * self.relu_derivative(self.pre_activations[l - 1])
                elif act_type == 'sigmoid':
                    delta = delta * self.sigmoid_derivative(self.pre_activations[l - 1])
                # 'linear' derivative is 1, no change needed

        return grad_weights, grad_biases
        # ----------------------

    def update_parameters(self, grad_weights, grad_biases):
        """
        TODO 7: Update weights and biases using gradient descent.

        Args:
            grad_weights (list of numpy.ndarray): Weight gradients per layer.
            grad_biases  (list of numpy.ndarray): Bias gradients per layer.
        """
        # --- YOUR CODE HERE ---
        for l in range(self.L):
            self.weights[l] -= self.learning_rate * grad_weights[l]
            self.biases[l]  -= self.learning_rate * grad_biases[l]
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
        costs = []
        for epoch in range(1, epochs + 1):
            output = self.forward_propagation(X)
            loss = self.compute_loss(output, y)
            grad_weights, grad_biases = self.backward_propagation(y)
            self.update_parameters(grad_weights, grad_biases)

            if epoch % print_freq == 0:
                print(f"Epoch {epoch}/{epochs} — Loss: {loss:.6f}")
                costs.append(loss)
        return costs
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
        output = self.forward_propagation(X)
        if self.output_act == 'sigmoid':
            return (output >= threshold).astype(int)
        else:
            return output
        # ----------------------