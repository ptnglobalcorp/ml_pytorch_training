"""
Exercise 1: Tensor Creation
PyTorch Fundamentals - Module 1

This exercise covers:
- Creating tensors from different sources
- Understanding tensor attributes
- Creating tensors with specific values
"""

import torch
import numpy as np

# ============================================
# Part 1: Creating Tensors from Data
# ============================================

print("=" * 60)
print("Part 1: Creating Tensors from Data")
print("=" * 60)

# TODO: Create a scalar tensor (0-dimensional)
scalar = torch.tensor(7)
print(f"Scalar: {scalar}")
print(f"Scalar ndim: {scalar.ndim}")

# TODO: Create a vector tensor (1-dimensional)
vector = torch.tensor([1, 2, 3, 4, 5])
print(f"\nVector: {vector}")
print(f"Vector shape: {vector.shape}")

# TODO: Create a matrix tensor (2-dimensional)
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"\nMatrix:\n{matrix}")
print(f"Matrix shape: {matrix.shape}")

# TODO: Create a 3D tensor
tensor_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"\n3D Tensor:\n{tensor_3d}")
print(f"3D Tensor shape: {tensor_3d.shape}")


# ============================================
# Part 2: Creating Tensors with Specific Values
# ============================================

print("\n" + "=" * 60)
print("Part 2: Creating Tensors with Specific Values")
print("=" * 60)

# TODO: Create a tensor filled with zeros
zeros = torch.zeros(3, 4)
print(f"Zeros (3x4):\n{zeros}")

# TODO: Create a tensor filled with ones
ones = torch.ones(2, 3)
print(f"\nOnes (2x3):\n{ones}")

# TODO: Create a tensor filled with a specific value
full = torch.full((2, 3), 7)
print(f"\nFull with 7s (2x3):\n{full}")

# TODO: Create identity matrix
identity = torch.eye(4)
print(f"\nIdentity matrix (4x4):\n{identity}")

# TODO: Create a tensor with values from 0 to 9
range_tensor = torch.arange(0, 10)
print(f"\nRange (0-9):\n{range_tensor}")

# TODO: Create evenly spaced values
linspace = torch.linspace(0, 10, steps=5)
print(f"\nLinspace (0-10, 5 steps):\n{linspace}")


# ============================================
# Part 3: Creating Random Tensors
# ============================================

print("\n" + "=" * 60)
print("Part 3: Creating Random Tensors")
print("=" * 60)

# TODO: Set random seed for reproducibility
torch.manual_seed(42)

# TODO: Create random tensor from uniform distribution [0, 1)
rand_uniform = torch.rand(3, 4)
print(f"Random uniform (3x4):\n{rand_uniform}")

# TODO: Create random tensor from normal distribution
rand_normal = torch.randn(2, 3)
print(f"\nRandom normal (2x3):\n{rand_normal}")

# TODO: Create random integers
rand_int = torch.randint(low=0, high=10, size=(3, 3))
print(f"\nRandom integers (0-9, 3x3):\n{rand_int}")


# ============================================
# Part 4: Working with NumPy
# ============================================

print("\n" + "=" * 60)
print("Part 4: Working with NumPy")
print("=" * 60)

# TODO: Create a NumPy array
np_array = np.array([[1, 2, 3], [4, 5, 6]])
print(f"NumPy array:\n{np_array}")

# TODO: Convert NumPy array to PyTorch tensor
tensor_from_np = torch.from_numpy(np_array)
print(f"\nTensor from NumPy:\n{tensor_from_np}")

# TODO: Convert PyTorch tensor to NumPy array
np_from_tensor = tensor_from_np.numpy()
print(f"\nNumPy from tensor:\n{np_from_tensor}")


# ============================================
# Part 5: Tensor Attributes
# ============================================

print("\n" + "=" * 60)
print("Part 5: Tensor Attributes")
print("=" * 60)

tensor = torch.randn(2, 3, 4)

# TODO: Print tensor shape
print(f"Shape: {tensor.shape}")

# TODO: Print tensor data type
print(f"Dtype: {tensor.dtype}")

# TODO: Print tensor device
print(f"Device: {tensor.device}")

# TODO: Print number of dimensions
print(f"Number of dimensions: {tensor.ndim}")

# TODO: Print total number of elements
print(f"Total elements: {tensor.numel()}")


# ============================================
# Exercises
# ============================================

print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

# Exercise 1: Create a 4x4 tensor filled with the value 3.14
print("\nExercise 1: Create a 4x4 tensor filled with 3.14")
# Your code here

# Exercise 2: Create a random tensor of shape (5, 5) with values between 0 and 1
print("\nExercise 2: Create random 5x5 tensor [0, 1)")
# Your code here

# Exercise 3: Create a tensor with values from 10 to 50 (exclusive) with step 5
print("\nExercise 3: Range from 10 to 50 with step 5")
# Your code here

# Exercise 4: Create a 3D tensor of shape (2, 3, 4) filled with ones
print("\nExercise 4: Create 3D tensor (2, 3, 4) of ones")
# Your code here

# Exercise 5: Check if a GPU is available and create a tensor on it if possible
print("\nExercise 5: Create tensor on GPU if available")
# Your code here


print("\n" + "=" * 60)
print("Exercise 1 Complete!")
print("=" * 60)
