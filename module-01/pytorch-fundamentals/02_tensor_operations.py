"""
Exercise 2: Tensor Operations
PyTorch Fundamentals - Module 1

This exercise covers:
- Basic arithmetic operations
- Matrix operations
- Reduction operations
- Comparison operations

PyTorch 2.0 Note: All operations in this file are compatible with PyTorch 2.0.
The @ operator for matrix multiplication is recommended for both PyTorch 1.x and 2.0.
"""

import torch

# Set random seed for reproducibility
torch.manual_seed(42)

# ============================================
# Part 1: Basic Arithmetic Operations
# ============================================

print("=" * 60)
print("Part 1: Basic Arithmetic Operations")
print("=" * 60)

# Create test tensors
x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
y = torch.tensor([5, 6, 7, 8], dtype=torch.float32)

print(f"x: {x}")
print(f"y: {y}")

# TODO: Addition
add_result = x + y
print(f"\nAddition (x + y): {add_result}")

# TODO: Subtraction
sub_result = x - y
print(f"Subtraction (x - y): {sub_result}")

# TODO: Element-wise multiplication
mul_result = x * y
print(f"Multiplication (x * y): {mul_result}")

# TODO: Element-wise division
div_result = x / y
print(f"Division (x / y): {div_result}")

# TODO: Power
pow_result = x ** 2
print(f"Power (x^2): {pow_result}")


# ============================================
# Part 2: Matrix Operations
# ============================================

print("\n" + "=" * 60)
print("Part 2: Matrix Operations")
print("=" * 60)

# Create matrices
A = torch.randn(3, 4)
B = torch.randn(4, 5)

print(f"Matrix A shape: {A.shape}")
print(f"Matrix B shape: {B.shape}")

# TODO: Matrix multiplication
matmul_result = A @ B
print(f"\nMatrix multiplication A @ B shape: {matmul_result.shape}")

# TODO: Transpose
A_T = A.T
print(f"Transpose of A shape: {A_T.shape}")

# TODO: Element-wise min/max
min_val = torch.min(A)
max_val = torch.max(A)
print(f"\nMin value in A: {min_val:.4f}")
print(f"Max value in A: {max_val:.4f}")


# ============================================
# Part 3: Reduction Operations
# ============================================

print("\n" + "=" * 60)
print("Part 3: Reduction Operations")
print("=" * 60)

# Create a 2D tensor
tensor = torch.randn(3, 4)
print(f"Tensor:\n{tensor}")

# TODO: Sum all elements
sum_all = torch.sum(tensor)
print(f"\nSum of all elements: {sum_all:.4f}")

# TODO: Sum along rows (dim=0)
sum_rows = torch.sum(tensor, dim=0)
print(f"Sum along rows: {sum_rows}")

# TODO: Sum along columns (dim=1)
sum_cols = torch.sum(tensor, dim=1)
print(f"Sum along columns: {sum_cols}")

# TODO: Mean
mean_all = torch.mean(tensor)
print(f"\nMean of all elements: {mean_all:.4f}")

# TODO: Standard deviation
std_all = torch.std(tensor)
print(f"Standard deviation: {std_all:.4f}")


# ============================================
# Part 4: Comparison Operations
# ============================================

print("\n" + "=" * 60)
print("Part 4: Comparison Operations")
print("=" * 60)

x = torch.tensor([1, 2, 3, 4, 5])
y = torch.tensor([2, 2, 4, 4, 6])

print(f"x: {x}")
print(f"y: {y}")

# TODO: Element-wise comparison
eq_result = x == y
print(f"\nEqual (x == y): {eq_result}")

gt_result = x > y
print(f"Greater than (x > y): {gt_result}")

lt_result = x < y
print(f"Less than (x < y): {lt_result}")

# TODO: Clamp values
clamped = torch.clamp(x, min=2, max=4)
print(f"\nClamp x to [2, 4]: {clamped}")


# ============================================
# Part 5: Broadcasting
# ============================================

print("\n" + "=" * 60)
print("Part 5: Broadcasting")
print("=" * 60)

# Create tensors with different shapes
x = torch.randn(3, 1)
y = torch.randn(1, 4)

print(f"x shape: {x.shape}")
print(f"y shape: {y.shape}")

# TODO: Broadcast addition
result = x + y
print(f"\nBroadcast result shape: {result.shape}")
print(f"Result:\n{result}")


# ============================================
# Exercises
# ============================================

print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

# Exercise 1: Given two matrices A (4x3) and B (3x5), compute C = A @ B
print("\nExercise 1: Matrix multiplication")
A = torch.randn(4, 3)
B = torch.randn(3, 5)
# Compute C = A @ B

# Exercise 2: Normalize a tensor to have zero mean and unit variance
print("\nExercise 2: Normalize tensor")
tensor = torch.randn(100, 10)
# Normalize: (x - mean) / std

# Exercise 3: Find the indices of the top 3 values in a tensor
print("\nExercise 3: Find top 3 indices")
tensor = torch.randn(10)
# Find indices of top 3 values

# Exercise 4: Implement softmax function
print("\nExercise 4: Implement softmax")
logits = torch.randn(2, 5)  # Batch of 2, 5 classes
# Implement: softmax(x) = exp(x) / sum(exp(x))

# Exercise 5: Compute cosine similarity between two vectors
print("\nExercise 5: Cosine similarity")
v1 = torch.randn(100)
v2 = torch.randn(100)
# Compute cosine similarity: (v1 . v2) / (||v1|| * ||v2||)


print("\n" + "=" * 60)
print("Exercise 2 Complete!")
print("=" * 60)
