import numpy as np

class Softmax:
    """
    Fully connected output layer with softmax activation.
    Converts raw scores into class probabilities.
    """

    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        """
        Performs a forward pass of the softmax layer.
        """
        input = input.flatten()
        totals = np.dot(input, self.weights) + self.biases

        # Stability trick
        exp = np.exp(totals - np.max(totals))
        return exp / np.sum(exp, axis=0)


# Example usage
if __name__ == "__main__":
    softmax = Softmax(4, 3)
    x = np.array([2.0, 1.0, 0.1, -1.5])
    output = softmax.forward(x)
    print("Softmax Output:", output)
    print("Sum of Probabilities:", np.sum(output))
