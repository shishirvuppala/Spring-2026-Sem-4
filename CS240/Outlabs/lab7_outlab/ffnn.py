import numpy as np

class FeedForwardNN:

    def __init__(self, layer_sizes, hidden_activation='relu', output_activation='sigmoid', learning_rate=0.01):
self.layer_sizes = layer_sizes self.hidden_act = hidden_activation
        self.output_act = output_activation
        self.learning_rate = learning_rate

        self.L = len(layer_sizes) - 1 self.weights = []
        self.biases = []

        self.pre_activations = []
        self.activations = []

        self.initialize_parameters()


    def initialize_parameters(self):
        """
        Initialize all the parameters of the neural network.
        All the weights with appropriate shapes
        Do random initialization using standard normal distribution
        and all the biases.
        For weigths initialize them with a scale of  2 / sqrt(in) (To mantain variance)
        """

        # DO NOT REMOVE
        np.random.seed(42)

        #TODO 1
        pass


    def relu(self, z: np.ndarray):
        """
        Implement reLu on z
        z: np.ndarray
        output: np.ndarray with relu applied to it (Same shape as z)
        """

        # TODO 2
        pass


    def relu_derivative(self, z: np.ndarray):
        """
        Implement the derivative of relu on z
        z: np.ndarray
        output: np.ndarray with relu's derivative (Same shape as z)
        """

        # TODO 3
        pass


    def sigmoid(self, z: np.ndarray):
        """
        Implement sigmoid on z
        z: np.ndarray
        output: np.ndarray with sigmoid applied to it (Same shape as z)
        don't forget to clip
        """

        #TODO 4
        pass


    def sigmoid_derivative(self, z: np.ndarray):
        """
        Implement sigmoid derivative on z
        z: np.ndarray
        output: np.ndarray with sigmoid derivative applied to it (Same shape as z)
        don't forget to clip
        """

        #TODO 5
        pass


    def softmax(self, z: np.ndarray):
        """
        Implement softmax on Z: np.ndarray (N, d)
        Output: np.ndarray with softmax taken on each row (N, d)
        don't forget to make it numerically stable
        """

        # TOOD 6
        pass


    def activate(self, z: np.ndarray, activation_type: str):
        """
        Handle the correct activation function for z.

        Return the value of applying said activation function to z

        """


        #TODO 7
        pass


    def forward_propagation(self, X: np.ndarray):
        """
        Implement Forward propogation.

        Fill in the activations and pre-activations arrays appropriately. 

        return the final output of the neural net (after the output layer activation)

        """
        # TODO 8
        pass


    def backward_propagation(self, grads: np.ndarray):
        """
        Implement backward propogation

        grads is the gradient of the loss with respect to the output of the final layer. (It is slightly different in semantics from delta so keep that in mind)

        propogate the grad_weights and grad_biases arrays appropriately

        return grad_weights, grad_biases

        """

        grad_weights = [np.zeros_like(w) for w in self.weights]
        grad_biases  = [np.zeros_like(b) for b in self.biases]


        # TODO 9

        return grad_weights, grad_biases




    def update_parameters(self, grad_weights: list, grad_biases: list):
        """
        Update all the paramters according to grad_weights and grad_biases (with learning rate)

        """

        # TODO 10
        pass

    def train(self, X: np.ndarray, y: np.ndarray, epochs=10000):
        """
        Train the neural network. For now no need to implement any batching etc. In each epoch run forward backward and update on the whole dataset.
        Assume that we will only do regression with MSE loss.
        """

        for epoch in range(epochs):
            # TODO 11

            if epoch % 1000 == 0:
                # FILL in with appropriate losss
                loss = 0
                print(f"Epoch {epoch} Loss {loss}")
