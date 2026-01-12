# Tensor Basics

## Learning Objectives

By the end of this lesson, you will be able to:
- Understand what tensors are and why they're fundamental to PyTorch
- Create tensors using various methods
- Understand tensor attributes (shape, dtype, device)
- Convert between NumPy arrays and PyTorch tensors

## Introduction to Tensors

### What is a Tensor?

A **tensor** is a multi-dimensional array that is the fundamental data structure in PyTorch. Tensors are similar to NumPy arrays but with additional capabilities:

- **GPU Acceleration**: Tensors can be moved to and operated on GPUs
- **Automatic Differentiation**: Tensors can track gradients for neural network training
- **Optimized Performance**: Tensors are optimized for deep learning operations

> **Note**: You can think of tensors as generalized scalars, vectors, and matrices:
> - **Scalar**: 0-dimensional tensor (single number)
> - **Vector**: 1-dimensional tensor (array of numbers)
> - **Matrix**: 2-dimensional tensor (grid of numbers)
> - **Tensor**: N-dimensional tensor (multi-dimensional array)

### Tensor Dimensions

| Dimension | Name | Example | Notation |
|-----------|------|---------|----------|
| 0D | Scalar | `5` | `torch.tensor(5)` |
| 1D | Vector | `[1, 2, 3]` | `torch.tensor([1, 2, 3])` |
| 2D | Matrix | `[[1, 2], [3, 4]]` | `torch.tensor([[1, 2], [3, 4]])` |
| 3D | 3-Tensor | Image with RGB channels | `torch.randn(3, 224, 224)` |
| ND | N-Tensor | Batch of images | `torch.randn(32, 3, 224, 224)` |

## Creating Tensors

### 1. From Data

```python
import torch

# From a list
scalar = torch.tensor(7)
vector = torch.tensor([1, 2, 3])
matrix = torch.tensor([[1, 2], [3, 4]])

print(scalar)
print(vector)
print(matrix)
```

**Output:**
```
tensor(7)
tensor([1, 2, 3])
tensor([[1, 2],
        [3, 4]])
```

### 2. With Specific Data Types

```python
# Explicit dtype
float_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)

print(float_tensor.dtype)  # torch.float32
print(int_tensor.dtype)     # torch.int32
```

**Common Data Types:**

| Dtype | Description | Use Case |
|-------|-------------|----------|
| `torch.float32` | 32-bit floating point | Default for most operations |
| `torch.float64` | 64-bit floating point | High precision calculations |
| `torch.int32` | 32-bit integer | Indexing, counting |
| `torch.int64` | 64-bit integer | Large indices |
| `torch.bool` | Boolean | Masks, conditional operations |

### 3. From NumPy Arrays

```python
import numpy as np

# NumPy to Tensor
np_array = np.array([1, 2, 3])
tensor_from_np = torch.from_numpy(np_array)

# Tensor to NumPy
np_from_tensor = tensor_from_np.numpy()

print(tensor_from_np)
print(np_from_tensor)
```

> **Warning**: When converting from NumPy, be aware that NumPy uses `float64` by default while PyTorch uses `float32`. This can cause type mismatches in your code.

### 4. Creating Tensors with Specific Values

```python
# Zeros
zeros = torch.zeros(size=(3, 4))

# Ones
ones = torch.ones(size=(3, 4))

# Full (custom value)
full = torch.full(size=(3, 4), fill_value=7)

# Range
range_tensor = torch.arange(start=0, end=10, step=2)

# Linspace (evenly spaced)
linspace = torch.linspace(start=0, end=10, steps=5)

# Identity matrix
identity = torch.eye(n=3)

print("Zeros:\n", zeros)
print("Ones:\n", ones)
print("Range:", range_tensor)
print("Linspace:", linspace)
print("Identity:\n", identity)
```

### 5. Creating Random Tensors

```python
# Random uniform [0, 1)
random_uniform = torch.rand(size=(3, 4))

# Random normal distribution (mean=0, std=1)
random_normal = torch.randn(size=(3, 4))

# Random integers
random_int = torch.randint(low=0, high=10, size=(3, 4))

# With specific shape
random_like = torch.randn_like(input=torch.ones(3, 4))

print("Random uniform:\n", random_uniform)
print("Random normal:\n", random_normal)
print("Random integers:\n", random_int)
```

## Tensor Attributes

Every tensor has three key attributes:

```python
tensor = torch.randn(3, 4)

print(f"Shape: {tensor.shape}")      # torch.Size([3, 4])
print(f"Dtype: {tensor.dtype}")      # torch.float32
print(f"Device: {tensor.device}")    # cpu
```

### Shape

The **shape** (or size) tells you the dimensions of the tensor:

```python
scalar = torch.tensor(7)
vector = torch.tensor([1, 2, 3, 4])
matrix = torch.tensor([[1, 2], [3, 4]])

print(scalar.shape)   # torch.Size([])      - 0 dimensions
print(vector.shape)   # torch.Size([4])     - 1 dimension, 4 elements
print(matrix.shape)   # torch.Size([2, 2])  - 2 dimensions, 2x2 elements
```

### Dtype

The **dtype** specifies the data type of the tensor elements:

```python
# Default dtype depends on the input
float_tensor = torch.tensor([1.0, 2.0, 3.0])   # torch.float32
int_tensor = torch.tensor([1, 2, 3])            # torch.int64 (long)

print(float_tensor.dtype)   # torch.float32
print(int_tensor.dtype)     # torch.int64
```

> **Tip**: For deep learning, you'll almost always use `torch.float32` for your data and tensors.

### Device

The **device** indicates where the tensor is stored (CPU or GPU):

```python
# CPU tensor (default)
cpu_tensor = torch.tensor([1, 2, 3])

# Move to GPU (if available)
if torch.cuda.is_available():
    gpu_tensor = cpu_tensor.to('cuda')
    print(gpu_tensor.device)  # cuda:0

# Create directly on GPU
if torch.cuda.is_available():
    gpu_tensor = torch.tensor([1, 2, 3], device='cuda')
```

## Common Operations on Shape

```python
# Get number of dimensions (rank)
tensor = torch.randn(2, 3, 4)
print(f"Dimensions: {tensor.ndim}")  # 3

# Get total number of elements
print(f"Total elements: {tensor.numel()}")  # 24

# Get size of specific dimension
print(f"First dimension size: {tensor.shape[0]}")  # 2
print(f"Second dimension size: {tensor.size(1)}")  # 3
```

## Practical Examples

### Example 1: Creating Image-like Tensors

```python
# Grayscale image (1 channel, 28x28 pixels)
grayscale = torch.randn(1, 28, 28)

# RGB image (3 channels, 224x224 pixels)
rgb_image = torch.randn(3, 224, 224)

# Batch of 32 RGB images
batch = torch.randn(32, 3, 224, 224)

print(f"Grayscale shape: {grayscale.shape}")
print(f"RGB image shape: {rgb_image.shape}")
print(f"Batch shape: {batch.shape}")
```

### Example 2: Creating Sequence-like Tensors

```python
# Time series data (100 time steps, 5 features)
time_series = torch.randn(100, 5)

# Batch of sequences (32 sequences, 100 time steps, 5 features)
batch_sequences = torch.randn(32, 100, 5)

print(f"Time series shape: {time_series.shape}")
print(f"Batch shape: {batch_sequences.shape}")
```

## Key Takeaways

| Concept | Description |
|---------|-------------|
| **Tensors** are multi-dimensional arrays, the core data structure in PyTorch |
| **Dimensions** range from 0D (scalar) to ND (multi-dimensional) |
| **Creation methods** include: from data, zeros/ones, random, ranges |
| **Key attributes**: shape, dtype, device |
| **NumPy integration** for easy conversion between libraries |
| **GPU support** for accelerated computation |

## Practice Exercises

1. Create a 3x3 tensor filled with the value 5
2. Create a random tensor of shape (2, 4, 6) and print its shape and dtype
3. Create a tensor with values from 0 to 100 (exclusive) with step 5
4. Convert a NumPy array of shape (5, 5) to a PyTorch tensor
5. Check if GPU is available and create a tensor on the GPU if possible

## Next Steps

- [Tensor Operations Guide](tensor-operations.md) - Learn mathematical and comparison operations
- [Tensor Manipulation](tensor-manipulation.md) - Reshaping, indexing, and slicing

---

**Last Updated**: January 2026
