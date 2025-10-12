# ================================================================
# 🧠 PYTHON DATA TYPES TUTORIAL (For Machine Learning Beginners)
# Author: Vikram Kumar
# ================================================================

import numpy as np

# ------------------------------------------------
# 1️⃣ Numeric Types: int, float, bool
# ------------------------------------------------

# Integer
age = 25
print("Integer:", age, type(age))

# Float
height = 5.9
print("Float:", height, type(height))

# Boolean
is_student = True
print("Boolean:", is_student, type(is_student))

# Basic Operations
x, y = 10, 3
print("Addition:", x + y)
print("Division:", x / y)
print("Floor Division:", x // y)
print("Power:", x ** y)
print("Modulo:", x % y)

# ------------------------------------------------
# 2️⃣ String Type: str
# ------------------------------------------------
name = "Vikram Kumar"
print("\nString:", name, type(name))
print("Length of string:", len(name))
print("First 6 letters:", name[:6])
print("Uppercase:", name.upper())
print("Lowercase:", name.lower())
print("Replace:", name.replace("Kumar", "Singh"))

# String formatting
print(f"My name is {name} and I am {age} years old.")

# ------------------------------------------------
# 3️⃣ List: Mutable Ordered Collection
# ------------------------------------------------
scores = [90, 85, 88, 92]
print("\nList:", scores, type(scores))
print("First element:", scores[0])
scores.append(95)
print("After append:", scores)
scores[1] = 99
print("After update:", scores)
print("Slicing:", scores[1:3])

# Useful operations
print("Sum:", sum(scores))
print("Max:", max(scores))
print("Min:", min(scores))

# ------------------------------------------------
# 4️⃣ Tuple: Immutable Ordered Collection
# ------------------------------------------------
coordinates = (10, 20)
print("\nTuple:", coordinates, type(coordinates))
# coordinates[0] = 5  # ❌ Error (Tuples are immutable)

# ------------------------------------------------
# 5️⃣ Set: Unordered Unique Elements
# ------------------------------------------------
features = {"height", "weight", "age", "age"}
print("\nSet (unique values only):", features, type(features))
features.add("income")
print("After add:", features)
features.remove("height")
print("After remove:", features)

# ------------------------------------------------
# 6️⃣ Dictionary: Key-Value Pairs
# ------------------------------------------------
student = {
    "name": "Vikram",
    "age": 25,
    "course": "Machine Learning",
}
print("\nDictionary:", student, type(student))
print("Access by key:", student["name"])
student["age"] = 26
print("Updated age:", student["age"])
student["grade"] = "A"
print("Added new key:", student)
print("All keys:", student.keys())
print("All values:", student.values())

# ------------------------------------------------
# 7️⃣ NumPy Array: For Machine Learning Computations
# ------------------------------------------------
arr = np.array([1, 2, 3, 4])
print("\nNumPy Array:", arr, type(arr))
print("Shape:", arr.shape)
print("Data Type:", arr.dtype)
print("Sum:", np.sum(arr))
print("Mean:", np.mean(arr))
print("Reshape Example:", np.array([[1, 2], [3, 4]]))

# Element-wise operation
arr2 = np.array([10, 20, 30, 40])
print("Addition:", arr + arr2)
print("Multiplication:", arr * arr2)

# ------------------------------------------------
# 8️⃣ Type Conversion (Casting)
# ------------------------------------------------
print("\nType Conversion Examples:")
print("String to Int:", int("10"))
print("Float to Int:", int(5.9))
print("Int to Float:", float(10))
print("List to Set:", set([1, 2, 2, 3]))
print("List to Tuple:", tuple([1, 2, 3]))

# ------------------------------------------------
# 9️⃣ NoneType (Represents No Value)
# ------------------------------------------------
x = None
print("\nNoneType Example:", x, type(x))

# ================================================================
# END OF FILE
# ================================================================
