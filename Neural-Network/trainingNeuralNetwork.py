# ======================================================
# Neural Network from Scratch using NumPy
# Author: Vikram Kumar
# ======================================================

import numpy as np

# ------------------------------------------------------
# 1️⃣ Activation Functions
# ------------------------------------------------------
def sigmoid(x):
    """Sigmoid activation function: f(x) = 1 / (1 + e^(-x))"""
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    """Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))"""
    fx = sigmoid(x)
    return fx * (1 - fx)

# ------------------------------------------------------
# 2️⃣ Loss Function
# ------------------------------------------------------
def mse_loss(y_true, y_pred):
    """Mean Squared Error (MSE) loss"""
    return ((y_true - y_pred) ** 2).mean()

# ------------------------------------------------------
# 3️⃣ Neural Network Definition
# ------------------------------------------------------
class OurNeuralNetwork:
    """
    A simple Neural Network with:
      - 2 input features
      - 1 hidden layer with 2 neurons (h1, h2)
      - 1 output neuron (o1)
    
    Each neuron uses a sigmoid activation.
    This is a learning example, not for production use.
    """

    def __init__(self):
        # Initialize weights (randomly small normal values)
        self.w1, self.w2, self.w3 = np.random.normal(), np.random.normal(), np.random.normal()
        self.w4, self.w5, self.w6 = np.random.normal(), np.random.normal(), np.random.normal()

        # Initialize biases
        self.b1, self.b2, self.b3 = np.random.normal(), np.random.normal(), np.random.normal()

    # ---------------------------
    # Forward Pass
    # ---------------------------
    def feedforward(self, x):
        """Compute the output of the network for a given input x"""
        # Hidden layer
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        
        # Output neuron
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    # ---------------------------
    # Training Function
    # ---------------------------
    def train(self, data, all_y_trues, learn_rate=0.1, epochs=1000):
        """
        Train the neural network using gradient descent.

        data         : numpy array (n_samples x 2)
        all_y_trues  : true labels (n_samples)
        learn_rate   : step size for updates
        epochs       : number of training iterations
        """
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # --- Feedforward
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                # --- Compute partial derivatives
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Output neuron gradients
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)
                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                # Hidden neuron 1 gradients
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                # Hidden neuron 2 gradients
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # --- Update weights and biases (Gradient Descent)
                # Hidden layer
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Output layer
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

            # --- Track loss
            if epoch % 10 == 0:
                y_preds = np.array([self.feedforward(x) for x in data])
                loss = mse_loss(all_y_trues, y_preds)
                print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")

# ------------------------------------------------------
# 4️⃣ Dataset Definition
# ------------------------------------------------------
data = np.array([
    [-2, -1],  # Alice
    [25, 6],   # Bob
    [17, 4],   # Charlie
    [-15, -6], # Diana
])

all_y_trues = np.array([
    1,  # Alice
    0,  # Bob
    0,  # Charlie
    1,  # Diana
])

# ------------------------------------------------------
# 5️⃣ Train the Network
# ------------------------------------------------------
network = OurNeuralNetwork()
network.train(data, all_y_trues)

# ------------------------------------------------------
# 6️⃣ Test Predictions
# ------------------------------------------------------
test_inputs = np.array([8, 3])
pred = network.feedforward(test_inputs)

print("\nTest Input:", test_inputs)
print("Predicted Output:", round(pred, 3))
