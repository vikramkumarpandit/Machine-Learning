# ==========================================
# Simple Neuron Example using NumPy
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
# 2️⃣ Neuron Class Definition
# -------------------------------
class Neuron:
    """
    A simple neuron with multiple inputs, weights, bias and sigmoid activation
    """
    def __init__(self, weights, bias):
        """
        Constructor: Initialize neuron with weights and bias
        """
        self.weights = weights
        self.bias = bias
    
    def predict(self, inputs):
        """
        Compute weighted sum + bias, then apply sigmoid activation
        """
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

# -------------------------------
# 3️⃣ Define Inputs, Weights, Bias
# -------------------------------
inputs = np.array([2, 3])         # Input values: x1=2, x2=3
weights = np.array([0, 1])        # Weights: w1=0, w2=1
bias = 4                           # Bias: b=4

# -------------------------------
# 4️⃣ Create Neuron Object
# -------------------------------
neuron1 = Neuron(weights, bias)

# -------------------------------
# 5️⃣ Make Prediction
# -------------------------------
output1 = neuron1.predict(inputs)

# -------------------------------
# 6️⃣ Display Results
# -------------------------------
print("Inputs:", inputs)
print("Weights:", weights)
print("Bias:", bias)
print("Neuron output:", output1)
