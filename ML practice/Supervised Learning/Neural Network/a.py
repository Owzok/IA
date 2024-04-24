import numpy as np

# Input data (4 input neurons)
input_data = np.array([0.5, 0.3, 0.2, 0.7])

# Target output (3 output neurons)
target_output = np.array([0.8, 0.1, 0.4])

# Weights and biases for a single hidden layer (4 hidden neurons)
hidden_weights = np.random.randn(4, 4)
hidden_biases = np.random.randn(4)

# Weights and biases for the output layer (3 output neurons)
output_weights = np.random.randn(4, 3)
output_biases = np.random.randn(3)

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Forward pass
def forward_pass(input_data):
    global output_weights
    global output_biases
    global hidden_biases
    global hidden_weights
    global target_output
    hidden_activations = np.dot(input_data, hidden_weights) + hidden_biases
    hidden_outputs = sigmoid(hidden_activations)

    output_activations = np.dot(hidden_outputs, output_weights) + output_biases
    output_outputs = sigmoid(output_activations)

    return hidden_outputs, output_outputs

# Backpropagation
def backpropagation(input_data, target_output):
    global output_weights
    global output_biases
    global hidden_biases
    global hidden_weights
    hidden_outputs, output_outputs = forward_pass(input_data)

    # Calculate output layer error and deltas
    output_error = target_output - output_outputs
    output_deltas = output_error * sigmoid_derivative(output_outputs)

    # Calculate hidden layer error and deltas
    hidden_error = np.dot(output_deltas, output_weights.T)
    hidden_deltas = hidden_error * sigmoid_derivative(hidden_outputs)

    # Update output weights and biases
    output_weights += np.dot(hidden_outputs.reshape(-1, 1), output_deltas.reshape(1, -1))
    output_biases += output_deltas

    # Update hidden weights and biases
    hidden_weights += np.dot(input_data.reshape(-1, 1), hidden_deltas.reshape(1, -1))
    hidden_biases += hidden_deltas

# Training loop (for demonstration purposes)
epochs = 1000
for epoch in range(epochs):
    backpropagation(input_data, target_output)

# Perform forward pass after training
final_hidden_outputs, final_output_predictions = forward_pass(input_data)

print("Input Data:", input_data)
print("Final Hidden Outputs:", final_hidden_outputs)
print("Final Output Predictions:", final_output_predictions)