import numpy as np

class FeedForwardNN:

    def __init__(self, layer_sizes, hidden_activation='relu', output_activation='sigmoid', learning_rate=0.01):

        self.layer_sizes = layer_sizes
        self.hidden_act = hidden_activation
        self.output_act = output_activation
        self.learning_rate = learning_rate

        self.L = len(layer_sizes) - 1

        self.weights = []
        self.biases = []

        self.pre_activations = []
        self.activations = []

        self.initialize_parameters()


    def initialize_parameters(self):

        np.random.seed(42)

        for l in range(self.L):

            fan_in = self.layer_sizes[l]
            fan_out = self.layer_sizes[l+1]

            W = np.random.randn(fan_in, fan_out) * np.sqrt(2 / fan_in)
            b = np.zeros((1, fan_out))

            self.weights.append(W)
            self.biases.append(b)


    def relu(self, z):
        return np.maximum(0, z)


    def relu_derivative(self, z):
        return (z > 0).astype(float)


    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))


    def sigmoid_derivative(self, z):

        s = self.sigmoid(z)
        return s * (1 - s)


    def softmax(self, z):

        z_shift = z - np.max(z, axis=1, keepdims=True)
        exp = np.exp(z_shift)

        return exp / np.sum(exp, axis=1, keepdims=True)


    def activate(self, z, activation_type):

        if activation_type == 'relu':
            return self.relu(z)

        elif activation_type == 'sigmoid':
            return self.sigmoid(z)

        elif activation_type == 'linear':
            return z

        elif activation_type == 'softmax':
            return self.softmax(z)

        else:
            raise ValueError("Unsupported activation")


    def forward_propagation(self, X):

        self.activations = [X]
        self.pre_activations = []

        A = X

        for l in range(self.L):

            W = self.weights[l]
            b = self.biases[l]

            Z = A @ W + b

            self.pre_activations.append(Z)

            if l == self.L - 1:
                A = self.activate(Z, self.output_act)
            else:
                A = self.activate(Z, self.hidden_act)

            self.activations.append(A)

        return A


    def backward_propagation(self, grads):

        grad_weights = [np.zeros_like(w) for w in self.weights]
        grad_biases  = [np.zeros_like(b) for b in self.biases]

        delta = grads

        for l in reversed(range(self.L)):

            Z = self.pre_activations[l]
            A_prev = self.activations[l]

            if l == self.L - 1:

                if self.output_act == "softmax":

                    s = self.activations[l+1]

                    dot = np.sum(delta * s, axis=1, keepdims=True)

                    delta = s * (delta - dot)

                elif self.output_act == "sigmoid":

                    delta = delta * self.sigmoid_derivative(Z)

                elif self.output_act == "linear":

                    pass

            else:

                if self.hidden_act == "relu":
                    delta = delta * self.relu_derivative(Z)

                elif self.hidden_act == "sigmoid":
                    delta = delta * self.sigmoid_derivative(Z)

            grad_weights[l] = A_prev.T @ delta / A_prev.shape[0]
            grad_biases[l]  = np.sum(delta, axis=0, keepdims=True) / A_prev.shape[0]

            if l > 0:
                delta = delta @ self.weights[l].T

        return grad_weights, grad_biases


    def update_parameters(self, grad_weights, grad_biases):

        for l in range(self.L):

            self.weights[l] -= self.learning_rate * grad_weights[l]
            self.biases[l]  -= self.learning_rate * grad_biases[l]


    def train(self, X, y, epochs=10000):

        for epoch in range(epochs):

            y_hat = self.forward_propagation(X)

            grads = y_hat - y

            grad_weights, grad_biases = self.backward_propagation(grads)

            self.update_parameters(grad_weights, grad_biases)

            if epoch % 1000 == 0:
                loss = np.mean((y_hat - y)**2)
                print(f"Epoch {epoch} Loss {loss}")
