import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Load and preprocess the data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

train_Shaped = train_images.reshape((train_images.shape[0], 28 * 28))
test_Shaped = test_images.reshape((test_images.shape[0], 28 * 28))

# Initialize parameters for two layers
def initialize_parameters(input_size, layer_sizes):
    weights = []
    biases = []

    for size in layer_sizes:
        weight = np.random.randn(input_size, size) * 0.01
        bias = np.zeros(size)
        weights.append(weight)
        biases.append(bias)
        input_size = size

    return weights, biases

# Define activation functions
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Forward pass
def forward_pass(input, weights, biases):
    layer_outputs = []
    x = input

    for i in range(len(weights)):
        x = np.dot(x, weights[i]) + biases[i]
        if i < len(weights) - 1:
            x = relu(x)  # Apply ReLU activation except for the last layer
        else:
            x = softmax(x)  # Apply softmax activation for the output layer
        layer_outputs.append(x)

    return layer_outputs

# Loss function
def loss(y_hat, y):
    l = -np.sum(y * np.log(y_hat + 1e-10))  # Add a small epsilon to prevent log(0)
    return l
def lossfunction(input_vec , target):
    lozz = 0
    for i in range(len(input_vec)):
        lozz += loss(input_vec[i],target[i])
    lozz = lozz/len(input_vec)
    return(lozz)

def one_hot_encode(label, num_classes = 10):
    one_hot = np.zeros(num_classes)
    one_hot[label] = 1
    return one_hot


# Backpropagation

def calculate_average_loss(train_data, train_labels, weights, bias):
    total_loss = 0

    for i in range(len(train_data)):
        output = forward_pass(train_data[i], weights, bias)[-1]
        target = one_hot_encode(train_labels[i])
        loss_value = lossfunction(output, target)
        total_loss += loss_value

    average_loss = total_loss / len(train_data)
    return average_loss

# Usage
weights, bias = initialize_parameters(784, [128, 256, 10])
avg_loss = calculate_average_loss(train_Shaped, train_labels, weights, bias)
print("Average Loss:", avg_loss)

# Training


