"""
Exercise 3: Tensor Manipulation
PyTorch Fundamentals - Module 1

This exercise covers:
- Reshaping tensors
- Indexing and slicing
- Transposing and permuting
- Squeezing and unsqueezing
- Concatenating and splitting tensors

PyTorch 2.0 Note: All manipulation operations are compatible with PyTorch 2.0.
See the device management section for torch.set_default_device() (PyTorch 2.0+ feature).
"""

import torch

# Set random seed for reproducibility
torch.manual_seed(42)

# ============================================
# Part 1: Reshaping
# ============================================

print("=" * 60)
print("Part 1: Reshaping Tensors")
print("=" * 60)

# Create a 1D tensor
x = torch.arange(12)
print(f"Original tensor: {x}")
print(f"Original shape: {x.shape}")

# TODO: Reshape to 3x4
reshaped = x.reshape(3, 4)
print(f"\nReshaped to (3, 4):\n{reshaped}")

# TODO: Reshape to 2x6
reshaped_2x6 = x.reshape(2, 6)
print(f"\nReshaped to (2, 6):\n{reshaped_2x6}")

# TODO: Flatten a 2D tensor
flat = reshaped.flatten()
print(f"\nFlattened: {flat}")

# TODO: Reshape with view (only works on contiguous tensors)
viewed = reshaped.view(-1)  # -1 means infer dimension
print(f"\nViewed as 1D: {viewed}")


# ============================================
# Part 2: Squeeze and Unsqueeze
# ============================================

print("\n" + "=" * 60)
print("Part 2: Squeeze and Unsqueeze")
print("=" * 60)

# Create tensor with singleton dimensions
x = torch.randn(2, 1, 4, 1)
print(f"Original shape: {x.shape}")

# TODO: Remove all singleton dimensions
squeezed = x.squeeze()
print(f"After squeeze: {squeezed.shape}")

# TODO: Remove specific singleton dimension
squeezed_dim = x.squeeze(dim=1)
print(f"After squeeze(dim=1): {squeezed_dim.shape}")

# TODO: Add dimension at position 0
unsqueezed_0 = squeezed.unsqueeze(dim=0)
print(f"\nAfter unsqueeze(dim=0): {unsqueezed_0.shape}")

# TODO: Add dimension at last position
unsqueezed_last = squeezed.unsqueeze(dim=-1)
print(f"After unsqueeze(dim=-1): {unsqueezed_last.shape}")


# ============================================
# Part 3: Transpose and Permute
# ============================================

print("\n" + "=" * 60)
print("Part 3: Transpose and Permute")
print("=" * 60)

# Create a 3D tensor
x = torch.randn(2, 3, 4)
print(f"Original shape: {x.shape}")

# TODO: Transpose dimensions 0 and 1
transposed = torch.transpose(x, 0, 1)
print(f"After transpose(0, 1): {transposed.shape}")

# TODO: Permute all dimensions
permuted = x.permute(2, 0, 1)
print(f"After permute(2, 0, 1): {permuted.shape}")

# TODO: Matrix transpose for 2D tensor
matrix = torch.randn(3, 4)
print(f"\nMatrix shape: {matrix.shape}")
print(f"Matrix.T shape: {matrix.T.shape}")


# ============================================
# Part 4: Indexing and Slicing
# ============================================

print("\n" + "=" * 60)
print("Part 4: Indexing and Slicing")
print("=" * 60)

# Create a 2D tensor
x = torch.arange(20).reshape(4, 5)
print(f"Tensor:\n{x}")

# TODO: Get element at position (1, 2)
element = x[1, 2]
print(f"\nElement at [1, 2]: {element}")

# TODO: Get first row
first_row = x[0, :]
print(f"\nFirst row: {first_row}")

# TODO: Get last column
last_col = x[:, -1]
print(f"Last column: {last_col}")

# TODO: Get submatrix (rows 1-2, cols 2-4)
submatrix = x[1:3, 2:5]
print(f"\nSubmatrix [1:3, 2:5]:\n{submatrix}")

# TODO: Boolean indexing
mask = x > 10
values = x[mask]
print(f"\nValues > 10: {values}")


# ============================================
# Part 5: Concatenating and Splitting
# ============================================

print("\n" + "=" * 60)
print("Part 5: Concatenating and Splitting")
print("=" * 60)

# Create tensors to concatenate
x1 = torch.randn(2, 3)
x2 = torch.randn(2, 3)
x3 = torch.randn(2, 3)

print(f"x1 shape: {x1.shape}")
print(f"x2 shape: {x2.shape}")

# TODO: Stack along new dimension (dim=0)
stacked = torch.stack([x1, x2, x3], dim=0)
print(f"\nStacked along dim=0: {stacked.shape}")

# TODO: Concatenate along existing dimension (dim=0)
concat_dim0 = torch.cat([x1, x2, x3], dim=0)
print(f"Concatenated along dim=0: {concat_dim0.shape}")

# TODO: Split tensor into chunks
x = torch.randn(6, 4)
chunks = torch.chunk(x, chunks=3, dim=0)
print(f"\nSplit into 3 chunks along dim=0:")
for i, chunk in enumerate(chunks):
    print(f"  Chunk {i} shape: {chunk.shape}")

# TODO: Split at specific indices
split_sections = torch.split(x, [2, 4], dim=0)
print(f"\nSplit into sections of size [2, 4]:")
for i, section in enumerate(split_sections):
    print(f"  Section {i} shape: {section.shape}")


# ============================================
# Exercises
# ============================================

print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

# Exercise 1: Given a tensor of shape (64, 1, 28, 28), remove the singleton dimension
print("\nExercise 1: Remove singleton dimension")
x = torch.randn(64, 1, 28, 28)
# Your code here

# Exercise 2: Add a batch dimension to a single image tensor of shape (3, 224, 224)
print("\nExercise 2: Add batch dimension")
image = torch.randn(3, 224, 224)
# Your code here - should result in (1, 3, 224, 224)

# Exercise 3: Convert HWC format to CHW format
print("\nExercise 3: HWC to CHW conversion")
hwc_image = torch.randn(224, 224, 3)
# Your code here - should result in (3, 224, 224)

# Exercise 4: Extract the diagonal elements from a square matrix
print("\nExercise 4: Extract diagonal")
matrix = torch.randn(5, 5)
# Your code here

# Exercise 5: Create a batch of 10 random images and stack them
print("\nExercise 5: Stack images into batch")
images = [torch.randn(3, 224, 224) for _ in range(10)]
# Your code here - should result in (10, 3, 224, 224)


print("\n" + "=" * 60)
print("Exercise 3 Complete!")
print("=" * 60)
