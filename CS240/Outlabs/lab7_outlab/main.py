import numpy as np
from ffnn import FeedForwardNN
from moe import MoE




data = np.loadtxt("./testcases/data.csv", delimiter=",")


X = data[:, :-1]
y = data[:, -1:]
print(X)
print(y)

# Split
split = 1600
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# FFNN
ffnn = FeedForwardNN(
    layer_sizes=[1, 16, 16, 1],
    hidden_activation='relu',
    output_activation='linear',
    learning_rate=0.01
)
ffnn.train(X_train, y_train, epochs=10000)
# Test MSE
ffnn_pred = ffnn.forward_propagation(X_test)
ffnn_mse = np.mean((ffnn_pred - y_test) ** 2)
print(ffnn_mse)


# MoE
moe = MoE(
    input_dim=1,
    hidden_dim=16,
    num_experts=3,
    expert_layers=[1, 32, 1],
    learning_rate=0.1,
    router_reg=0.01
)
moe.train(X_train, y_train, epochs=10000)


moe_pred = moe.forward(X_test)
moe_mse = np.mean((moe_pred - y_test) ** 2)

print(f"\nTest MSE — FFNN: {ffnn_mse:.6f} | MoE: {moe_mse:.6f}")
print(f"MoE improvement: {(ffnn_mse - moe_mse) / ffnn_mse * 100:.1f}%")
