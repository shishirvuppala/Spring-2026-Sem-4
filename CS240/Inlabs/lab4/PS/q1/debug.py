from q1 import *


print("-" * 50)
print("Testing get_accuracy function")
y_true = np.array([0, 2, 3, 4, 5])
y_pred = np.array([0, 2, 1, 4, 5])
accuracy = get_multiclass_accuracy(y_true, y_pred)
print(f"Accuracy: {accuracy}")  # Expected output: 0.8
print("-" * 50)

# Here we are also checking for numerical stability. A correct implementation should not give any warnings here.
print("Testing sigmoid function")
z = np.array([-1, 100000000, 500000000, -1000000000000])
logistic_classifier = BaseLogisticClassifier()
sigmoid_output = logistic_classifier.sigmoid(z)
print(f"Sigmoid Output: {sigmoid_output}")  # Expected output: array close to [0, 1, 1, 0]
assert(sigmoid_output.shape == (4,))
print("-" * 50)