# ==========================================
# NumPy Basics for Machine Learning
# Author: Vikram Kumar
# ==========================================

import numpy as np

# -------------------------------
# 1️⃣ Creating NumPy Arrays
# -------------------------------
# Create 1D array
a = np.array([1, 2, 3, 4])
print("1D Array:\n", a)

# Create 2D array
b = np.array([[1, 2, 3], [4, 5, 6]])
print("\n2D Array:\n", b)

# Check dimensions, shape, and data type
print("\nDimensions:", b.ndim)   # 2
print("Shape:", b.shape)         # (2,3)
print("Data type:", b.dtype)     # int32 or int64

# -------------------------------
# 2️⃣ Creating Special Arrays
# -------------------------------
print("\nZeros:\n", np.zeros((2, 3)))
print("Ones:\n", np.ones((2, 3)))
print("Identity Matrix:\n", np.eye(3))
print("Arange:\n", np.arange(0, 10, 2))     # [0 2 4 6 8]
print("Linspace:\n", np.linspace(0, 1, 5))  # [0. 0.25 0.5 0.75 1.]

# -------------------------------
# 3️⃣ Indexing and Slicing
# -------------------------------
arr = np.array([[10, 20, 30],
                [40, 50, 60],
                [70, 80, 90]])
print("\nElement at (1,2):", arr[1, 2]) 
print("First row:", arr[0, :])
print("First column:", arr[:, 0])
print("Sub-array:\n", arr[0:2, 1:3])  # rows 0-1, cols 1-2

# -------------------------------
# 4️⃣ Mathematical Operations
# -------------------------------
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

print("\nAddition:", x + y)
print("Subtraction:", y - x)
print("Multiplication:", x * y)
print("Division:", y / x)
print("Exponent:", x ** 2)
print("Dot Product:", np.dot(x, y))

# -------------------------------
# 5️⃣ Statistical Functions
# -------------------------------
data = np.array([10, 20, 30, 40, 50])
print("\nMean:", np.mean(data))
print("Median:", np.median(data))
print("Standard Deviation:", np.std(data))
print("Sum:", np.sum(data))
print("Max:", np.max(data))
print("Min:", np.min(data))

# -------------------------------
# 6️⃣ Reshaping and Flattening
# -------------------------------
a = np.arange(1, 7)  # [1 2 3 4 5 6]
b = a.reshape(2, 3)
print("\nReshaped Array (2x3):\n", b)
print("Flattened Array:", b.flatten())

# -------------------------------
# 7️⃣ Broadcasting
# -------------------------------
A = np.array([[1, 2, 3],
              [4, 5, 6]])
B = np.array([10, 20, 30])
print("\nBroadcasted Addition:\n", A + B)

# -------------------------------
# 8️⃣ Random Numbers (for ML)
# -------------------------------
np.random.seed(42)  # for reproducibility
print("\nRandom floats:\n", np.random.rand(2, 3))
print("Random normal dist:\n", np.random.randn(2, 3))
print("Random integers:\n", np.random.randint(1, 10, size=(2, 3)))

# -------------------------------
# 9️⃣ Normalization (ML preprocessing)
# -------------------------------
X = np.array([10, 20, 30, 40, 50])
X_norm = (X - np.min(X)) / (np.max(X) - np.min(X))
print("\nNormalized data (0–1 scale):\n", X_norm)

# -------------------------------
# 🔟 Simple ML-style computation
# -------------------------------
# Linear equation: y = w*x + b
x = np.array([1, 2, 3, 4, 5])
w = 2.5   # weight
b = 1.0   # bias
y_pred = w * x + b
print("\nPredicted values (y = 2.5x + 1):\n", y_pred)

# -------------------------------
# 1️⃣1️⃣ Matrix Operations (used in ML)
# -------------------------------
M1 = np.array([[1, 2],
               [3, 4]])
M2 = np.array([[5, 6],
               [7, 8]])

print("\nMatrix Multiplication:\n", np.dot(M1, M2))
print("Transpose of M1:\n", M1.T)
print("Inverse of M1:\n", np.linalg.inv(M1))

# -------------------------------
# 1️⃣2️⃣ Summary
# -------------------------------
print("\n✅ NumPy basics covered successfully! Ready for ML projects.")
