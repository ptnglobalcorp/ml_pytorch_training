# Tensor Foundations

## Learning Objectives

By the end of this lesson, you will be able to:
- Understand tensors as multi-dimensional numerical representations
- Master the three critical tensor attributes: Shape, Rank, and Device
- Create tensors using essential methods
- Convert between NumPy arrays and PyTorch tensors

## What is a Tensor?

A **tensor** is a multi-dimensional array that is the fundamental data structure in PyTorch. Tensors are similar to NumPy arrays but with additional capabilities:

- **GPU Acceleration**: Tensors can be moved to and operated on GPUs
- **Automatic Differentiation**: Tensors can track gradients for neural network training
- **Optimized Performance**: Tensors are optimized for deep learning operations

You can think of tensors as generalized scalars, vectors, and matrices:

| Dimension | Name | Example | Notation |
|-----------|------|---------|----------|
| 0D | Scalar | `5` | `torch.tensor(5)` |
| 1D | Vector | `[1, 2, 3]` | `torch.tensor([1, 2, 3])` |
| 2D | Matrix | `[[1, 2], [3, 4]]` | `torch.tensor([[1, 2], [3, 4]])` |
| 3D | 3-Tensor | RGB Image | `torch.randn(3, 224, 224)` |
| ND | N-Tensor | Batch of images | `torch.randn(32, 3, 224, 224)` |

## The Three Critical Tensor Attributes

Every tensor has three attributes that you must understand. These will determine how you structure your data and where computations happen.

### 1. Shape

The **shape** tells you the dimensions of the tensor—how many elements exist in each dimension.

```python
import torch

scalar = torch.tensor(7)
vector = torch.tensor([1, 2, 3, 4])
matrix = torch.tensor([[1, 2], [3, 4]])
tensor_3d = torch.randn(2, 3, 4)

print(f"Scalar shape: {scalar.shape}")    # torch.Size([])
print(f"Vector shape: {vector.shape}")    # torch.Size([4])
print(f"Matrix shape: {matrix.shape}")    # torch.Size([2, 2])
print(f"3D Tensor shape: {tensor_3d.shape}")  # torch.Size([2, 3, 4])
```

**Understanding shapes:**
- `[]` = 0 dimensions (scalar)
- `[4]` = 1 dimension, 4 elements
- `[2, 2]` = 2 dimensions, 2×2 grid
- `[2, 3, 4]` = 3 dimensions, 2×3×4 block

**Practical examples:**
```python
# Grayscale image: 1 channel, 28×28 pixels
grayscale = torch.randn(1, 28, 28)

# RGB image: 3 channels, 224×224 pixels
rgb_image = torch.randn(3, 224, 224)

# Batch of 32 RGB images
batch = torch.randn(32, 3, 224, 224)

print(f"Grayscale: {grayscale.shape}")
print(f"RGB: {rgb_image.shape}")
print(f"Batch: {batch.shape}")
```

### 2. Rank (ndim)

The **rank** (or number of dimensions) tells you how many indices are needed to access a single element.

```python
scalar = torch.tensor(7)
vector = torch.tensor([1, 2, 3])
matrix = torch.tensor([[1, 2], [3, 4]])

print(f"Scalar ndim: {scalar.ndim}")  # 0
print(f"Vector ndim: {vector.ndim}")  # 1
print(f"Matrix ndim: {matrix.ndim}")  # 2
```

**Relationship between shape and ndim:**
- `ndim` = `len(shape)`
- A tensor with shape `[2, 3, 4]` has 3 dimensions

### 3. Device

The **device** indicates where the tensor is stored—CPU or GPU.

```python
# CPU tensor (default)
cpu_tensor = torch.tensor([1, 2, 3])
print(f"Device: {cpu_tensor.device}")  # cpu

# Check if GPU is available
if torch.cuda.is_available():
    # Move to GPU
    gpu_tensor = cpu_tensor.to('cuda')
    print(f"Device: {gpu_tensor.device}")  # cuda:0

    # PyTorch 2.0+: Set default device (optional)
    torch.set_default_device('cuda')
    auto_gpu = torch.randn(3, 4)  # Automatically on GPU
```

> **Important:** Tensors must be on the same device to perform operations together. You'll get an error if you try to add a CPU tensor to a GPU tensor.

## Creating Tensors

### From Data

Create tensors directly from Python data structures:

```python
# From a list
scalar = torch.tensor(7)
vector = torch.tensor([1, 2, 3])
matrix = torch.tensor([[1, 2], [3, 4]])
```

### With Specific Values

Five essential creation methods you'll use frequently:

```python
# Zeros
zeros = torch.zeros(3, 4)
# tensor([[0., 0., 0., 0.],
#         [0., 0., 0., 0.],
#         [0., 0., 0., 0.]])

# Ones
ones = torch.ones(2, 3)

# Random uniform [0, 1)
rand = torch.rand(3, 4)

# Random normal distribution (mean=0, std=1)
randn = torch.randn(3, 4)

# Range
arange = torch.arange(0, 10, 2)
# tensor([0, 2, 4, 6, 8])
```

### With Specific Data Types

```python
# Explicit dtype
float_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)

print(float_tensor.dtype)  # torch.float32
print(int_tensor.dtype)    # torch.int64
```

**Common data types:**

| Dtype | Description | Use Case |
|-------|-------------|----------|
| `torch.float32` | 32-bit floating point | Default for most operations |
| `torch.float64` | 64-bit floating point | High precision calculations |
| `torch.int64` | 64-bit integer | Indices, counting |
| `torch.bool` | Boolean | Masks, conditional operations |

> **Tip:** For deep learning, you'll almost always use `torch.float32` for your data and tensors.

### From NumPy Arrays

```python
import numpy as np

# NumPy to Tensor
np_array = np.array([1, 2, 3])
tensor_from_np = torch.from_numpy(np_array)

# Tensor to NumPy
np_from_tensor = tensor_from_np.numpy()

print(f"NumPy array: {np_array}")
print(f"From NumPy: {tensor_from_np}")
print(f"To NumPy: {np_from_tensor}")
```

> **Warning:** NumPy uses `float64` by default while PyTorch uses `float32`. This can cause type mismatches. Specify `dtype=np.float32` when creating NumPy arrays for PyTorch.

## Essential Tensor Operations

### Getting Information About Tensors

```python
tensor = torch.randn(2, 3, 4)

# Shape (dimensions)
print(f"Shape: {tensor.shape}")        # torch.Size([2, 3, 4])

# Number of dimensions (rank)
print(f"Dimensions: {tensor.ndim}")    # 3

# Total number of elements
print(f"Elements: {tensor.numel()}")   # 24

# Data type
print(f"Dtype: {tensor.dtype}")        # torch.float32

# Device
print(f"Device: {tensor.device}")      # cpu
```

### Accessing Specific Dimensions

```python
tensor = torch.randn(2, 3, 4)

# Size of specific dimension
print(f"First dim: {tensor.shape[0]}")      # 2
print(f"Second dim: {tensor.size(1)}")      # 3
print(f"Third dim: {tensor.size(2)}")       # 4
```

## Practical Examples

### Example 1: Image Data

Images are commonly represented as 3D tensors:

```python
# Single RGB image: (channels, height, width)
image = torch.randn(3, 224, 224)
print(f"Image shape: {image.shape}")  # torch.Size([3, 224, 224])

# Batch of images: (batch_size, channels, height, width)
batch = torch.randn(32, 3, 224, 224)
print(f"Batch shape: {batch.shape}")  # torch.Size([32, 3, 224, 224])
```

### Example 2: Sequence Data

Time series or text sequences are 2D or 3D tensors:

```python
# Time series: (time_steps, features)
time_series = torch.randn(100, 5)  # 100 time steps, 5 features
print(f"Time series shape: {time_series.shape}")

# Batch of sequences: (batch_size, time_steps, features)
batch_sequences = torch.randn(32, 100, 5)
print(f"Batch shape: {batch_sequences.shape}")
```

## Key Takeaways

| Concept | Description |
|---------|-------------|
| **Tensors** are multi-dimensional arrays, the core data structure in PyTorch |
| **Dimensions** range from 0D (scalar) to ND (multi-dimensional) |
| **Three Critical Attributes**: Shape (size of each dimension), Rank (number of dimensions), Device (CPU vs GPU) |
| **Creation Methods**: `zeros()`, `ones()`, `rand()`, `randn()`, `arange()` are the essentials |
| **NumPy Integration**: Easy conversion between NumPy and PyTorch |

## Practice Exercises

1. Create a 3×3 tensor filled with the value 5
2. Create a random tensor of shape (2, 4, 6) and print its shape, ndim, and device
3. Create a tensor with values from 0 to 100 (exclusive) with step 5
4. Convert a NumPy array of shape (5, 5) to a PyTorch tensor
5. Check if a GPU is available and create a tensor on the GPU if possible

## Next Steps

Now that you understand tensor creation and attributes, learn how to perform operations on them:

- [Tensor Operations](06-tensor-operations.md) - Mathematical operations and linear algebra
- [Tensor Manipulation](07-tensor-manipulation.md) - Reshaping, indexing, and device management

---

**Last Updated**: January 2026
