import numpy as np
from ffnn import FeedForwardNN

class MoE:

    def __init__(self, input_dim, hidden_dim, num_experts, expert_layers, learning_rate=0.01, router_reg=0.01):

        self.num_experts = num_experts
        self.learning_rate = learning_rate
        self.router_reg = router_reg

        # Router network
        self.router = FeedForwardNN(
            layer_sizes=[input_dim, hidden_dim, num_experts],
            hidden_activation='relu',
            output_activation='softmax',
            learning_rate=learning_rate
        )

        # Experts
        self.experts = [
            FeedForwardNN(
                layer_sizes=expert_layers,
                hidden_activation='relu',
                output_activation='linear',
                learning_rate=learning_rate
            )
            for _ in range(num_experts)
        ]


    def forward(self, X):

        gates = self.router.forward_propagation(X)     # (N,K)

        expert_outputs = []

        for expert in self.experts:
            out = expert.forward_propagation(X)        # (N,D)
            expert_outputs.append(out)

        expert_outputs = np.stack(expert_outputs, axis=1)  # (N,K,D)

        # Mixture
        y_hat = np.sum(
            gates[:, :, None] * expert_outputs,
            axis=1
        )                                              # (N,D)

        self.gates = gates
        self.expert_outputs = expert_outputs

        return y_hat


    def backward(self, grad_output):

        N, D = grad_output.shape
        K = self.num_experts

        # Gradient wrt router gates
        eps = 1e-12
        g = np.clip(self.gates, eps, 1)

        entropy_grad = -(np.log(g) + 1)

        grad_gates = np.sum(
            grad_output[:, None, :] * self.expert_outputs,
            axis=2
        )

        grad_gates -= self.router_reg * entropy_grad
        # Backprop router
        grad_w, grad_b = self.router.backward_propagation(grad_gates)
        self.router.update_parameters(grad_w, grad_b)

        # Backprop experts
        for k, expert in enumerate(self.experts):

            grad_expert = self.gates[:, k:k+1] * grad_output   # (N,D)

            grad_w, grad_b = expert.backward_propagation(grad_expert)
            expert.update_parameters(grad_w, grad_b)


    def compute_loss(self, y_pred, y):

        return (np.mean((y_pred - y) ** 2) / 2) 


    def router_entropy(self):
        eps = 1e-12

        g = np.clip(self.gates, eps, 1)
        entropy = -np.sum(g * np.log(g), axis=1)
        return np.mean(entropy)


    def loss_grad(self, y_pred, y):
        return (y_pred - y) 


    def train(self, X, y, epochs=1000, print_freq=1000):

        for epoch in range(epochs):

            y_pred = self.forward(X)

            loss = self.compute_loss(y_pred, y)

            grad_output = self.loss_grad(y_pred, y)

            self.backward(grad_output)

            if epoch % print_freq == 0:
                print(f"Epoch {epoch} | Loss {loss:.6f}")
