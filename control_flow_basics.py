# ==========================================
# 🌟 Python Control Flow Tutorial
# Author: Vikram Kumar
# ==========================================

# ------------------------------------------
# 1️⃣ Conditional Statements (if / elif / else)
# ------------------------------------------
x = 10

if x > 0:
    print("x is positive")
elif x == 0:
    print("x is zero")
else:
    print("x is negative")

# Nested if example
if x > 0:
    if x % 2 == 0:
        print("x is positive and even")
    else:
        print("x is positive and odd")

# ------------------------------------------
# 2️⃣ For Loop
# ------------------------------------------
print("\nFor loop example:")
for i in range(5):  # range(5) → 0,1,2,3,4
    print("Iteration:", i)

# Looping through a list
fruits = ["apple", "banana", "mango"]
for fruit in fruits:
    print("Fruit:", fruit)

# Using enumerate() to get index + value
for index, fruit in enumerate(fruits):
    print(f"Fruit {index}: {fruit}")

# ------------------------------------------
# 3️⃣ While Loop
# ------------------------------------------
print("\nWhile loop example:")
count = 0
while count < 3:
    print("Count is:", count)
    count += 1

# ------------------------------------------
# 4️⃣ Break, Continue, and Pass
# ------------------------------------------
print("\nBreak / Continue / Pass example:")
for num in range(10):
    if num == 3:
        continue  # skip number 3
    if num == 7:
        break     # stop loop at 7
    if num == 5:
        pass      # do nothing (placeholder)
    print("Number:", num)

# ------------------------------------------
# 5️⃣ Range Function
# ------------------------------------------
print("\nRange examples:")
print(list(range(5)))         # [0,1,2,3,4]
print(list(range(2, 10)))     # [2..9]
print(list(range(2, 10, 2)))  # [2,4,6,8]

# ------------------------------------------
# 6️⃣ List Comprehensions
# ------------------------------------------
print("\nList Comprehension examples:")
nums = [1, 2, 3, 4, 5]
squares = [n**2 for n in nums]
print("Squares:", squares)

# Conditional list comprehension
even_squares = [n**2 for n in nums if n % 2 == 0]
print("Even squares:", even_squares)

# Nested list comprehension (2D matrix)
matrix = [[i * j for j in range(3)] for i in range(3)]
print("Matrix:", matrix)

# ------------------------------------------
# 7️⃣ Try / Except / Finally
# ------------------------------------------
print("\nException Handling Example:")
try:
    val = int("hello")  # will raise ValueError
except ValueError:
    print("⚠️ ValueError: Cannot convert string to int")
finally:
    print("This will always execute")

# ------------------------------------------
# 8️⃣ Match-Case (Python 3.10+)
# ------------------------------------------
print("\nMatch-Case example:")
day = "Sunday"

match day:
    case "Monday":
        print("Start of the week!")
    case "Friday":
        print("Almost weekend!")
    case "Sunday" | "Saturday":
        print("Weekend vibes 😎")
    case _:
        print("Midweek days!")

# ------------------------------------------
# 9️⃣ Combining Everything: Example
# ------------------------------------------
print("\nCombined Example: Loop + Condition + Comprehension")

numbers = list(range(-3, 6))
positive_even_squares = [n**2 for n in numbers if n > 0 and n % 2 == 0]

print("Numbers:", numbers)
print("Positive even squares:", positive_even_squares)

# ------------------------------------------
# 🔟 While + Try Example (Input validation)
# ------------------------------------------
print("\nInput validation example (commented out to prevent blocking):")
# while True:
#     try:
#         num = int(input("Enter a number between 1 and 5: "))
#         if 1 <= num <= 5:
#             print("Valid number!")
#             break
#         else:
#             print("Out of range. Try again.")
#     except ValueError:
#         print("Invalid input. Please enter a number.")

print("\n✅ All control flow concepts covered successfully!")
