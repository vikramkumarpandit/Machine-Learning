import numpy as np
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax
from tensorflow.keras.datasets import mnist  # Using TensorFlow to load MNIST

# -------------------------------
# Load dataset
# -------------------------------
(_, _), (test_images, test_labels) = mnist.load_data()  # Load test data
test_images = test_images[:1000]  # Use first 1000 images
test_labels = test_labels[:1000]

# -------------------------------
# Initialize layers
# -------------------------------
conv = Conv3x3(8)                  # 28x28x1 -> 26x26x8
pool = MaxPool2()                  # 26x26x8 -> 13x13x8
softmax = Softmax(13 * 13 * 8, 10) # 13x13x8 -> 10

# -------------------------------
# Forward pass function
# -------------------------------
def forward(image, label):
    # Normalize image to [-0.5, 0.5]
    out = conv.forward((image / 255) - 0.5)  # Convolution
    out = pool.forward(out)                  # Max Pooling
    out = softmax.forward(out)               # Fully connected + softmax

    # Cross-entropy loss
    loss = -np.log(out[label])
    # Accuracy
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc

print('MNIST CNN initialized!')

# -------------------------------
# Evaluate on test images
# -------------------------------
loss = 0
num_correct = 0

for i, (im, label) in enumerate(zip(test_images, test_labels)):
    _, l, acc = forward(im, label)
    loss += l
    num_correct += acc

    if i % 100 == 99:
        print('[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
              (i + 1, loss / 100, num_correct))
        loss = 0
        num_correct = 0


