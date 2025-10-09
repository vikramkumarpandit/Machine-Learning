# ==========================================
# Tiny Neural Network Example using NumPy
# Author: Vikram Kumar
# ==========================================

import numpy as np

# -------------------------------
# 1️⃣ Activation Function
# -------------------------------
def sigmoid(x):
    """
    Sigmoid activation function
    f(x) = 1 / (1 + e^(-x))
    Maps input to range 0–1
    """
    return 1 / (1 + np.exp(-x))

# -------------------------------
# 2️⃣ Neuron Class
# -------------------------------
class Neuron:
    """
    A simple neuron with multiple inputs, weights, bias, and sigmoid activation
    """
    def __init__(self, weights, bias):
        """
        Constructor: Initialize neuron with weights and bias
        """
        self.weights = weights
        self.bias = bias
    
    def feedforward(self, inputs):
        """
        Compute weighted sum + bias, then apply sigmoid activation
        """
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

# -------------------------------
# 3️⃣ Tiny Neural Network Class
# -------------------------------
class OurNeuralNetwork:
    """
    A neural network with:
      - 2 inputs
      - a hidden layer with 2 neurons (h1, h2)
      - an output layer with 1 neuron (o1)
    Each neuron uses the same weights and bias for simplicity
    """
    def __init__(self):
        # Same weights and bias for all neurons (for simplicity)
        weights = np.array([0, 1])  # w1=0, w2=1
        bias = 4                     # b=4
        
        # Hidden layer neurons
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        
        # Output neuron
        self.o1 = Neuron(weights, bias)
    
    def feedforward(self, x):
        """
        Perform a forward pass through the network
        """
        # Hidden layer outputs
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)
        
        # Output layer input is outputs from hidden neurons
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))
        
        return out_o1

# -------------------------------
# 4️⃣ Testing the Network
# -------------------------------
network = OurNeuralNetwork()

x = np.array([2, 3])  # Input to the network

output = network.feedforward(x)

print("Network output for input", x, ":", output)
