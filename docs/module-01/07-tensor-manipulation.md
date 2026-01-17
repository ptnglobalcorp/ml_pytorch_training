# Tensor Manipulation

> **PyTorch 2.0 Note:** All tensor manipulation operations are fully compatible with PyTorch 2.0. PyTorch 2.0 introduces `torch.set_default_device()` for improved device management (see Device Management section).

## Learning Objectives

By the end of this lesson, you will be able to:
- Reshape tensors while preserving data
- Index and slice tensors to access specific elements
- Manipulate tensor dimensions
- Join and split tensors
- Move tensors between CPU and GPU

## Reshaping Tensors

### Reshape and View

```python
import torch

x = torch.arange(12)
print("Original:", x)
print("Shape:", x.shape)  # torch.Size([12])

# Reshape to 3x4
reshaped = x.reshape(3, 4)
print("Reshaped to (3, 4):\n", reshaped)

# View (similar to reshape, but shares memory)
viewed = x.view(3, 4)
print("Viewed as (3, 4):\n", viewed)

# Flatten to 1D
flat = reshaped.flatten()
print("Flattened:", flat)
```

> **Note**: `view()` requires the tensor to be contiguous in memory. Use `reshape()` for a safer option that handles non-contiguous tensors.

### Reshape Methods Comparison

| Method | Description | Use Case |
|--------|-------------|----------|
| `reshape()` | Can return view or copy | General reshaping |
| `view()` | Always returns a view | Performance, contiguous tensors |
| `flatten()` | Collapse to 1D | Before fully connected layers |
| `squeeze()` | Remove dimensions of size 1 | Clean up singleton dimensions |
| `unsqueeze()` | Add dimension of size 1 | Add batch/channel dimensions |

### Squeeze and Unsqueeze

```python
# Create tensor with shape (1, 3, 1, 5)
x = torch.randn(1, 3, 1, 5)
print("Original shape:", x.shape)  # torch.Size([1, 3, 1, 5])

# Remove all dimensions of size 1
squeezed = x.squeeze()
print("After squeeze:", squeezed.shape)  # torch.Size([3, 5])

# Remove specific dimension
squeezed_dim = x.squeeze(dim=0)
print("After squeeze(dim=0):", squeezed_dim.shape)  # torch.Size([3, 1, 5])

# Add dimension at position 0
unsqueezed = x.unsqueeze(dim=0)
print("After unsqueeze(dim=0):", unsqueezed.shape)  # torch.Size([1, 1, 3, 1, 5])

# Add dimension at position -1 (last)
unsqueezed_last = x.unsqueeze(dim=-1)
print("After unsqueeze(dim=-1):", unsqueezed_last.shape)  # torch.Size([1, 3, 1, 5, 1])
```

### Transpose and Permute

```python
x = torch.randn(2, 3, 4)
print("Original shape:", x.shape)  # torch.Size([2, 3, 4])

# Transpose two dimensions
transposed = torch.transpose(x, 0, 1)
print("After transpose(0, 1):", transposed.shape)  # torch.Size([3, 2, 4])

# Alternative syntax
transposed = x.transpose(0, 1)
print("Using .transpose():", transposed.shape)  # torch.Size([3, 2, 4])

# Permute all dimensions (general transpose)
permuted = x.permute(2, 0, 1)
print("After permute(2, 0, 1):", permuted.shape)  # torch.Size([4, 2, 3])
```

> **Tip**: Use `permute()` when you need to rearrange multiple dimensions at once. The argument specifies the new order of dimensions.

## Indexing and Slicing

### Basic Indexing

```python
x = torch.arange(12).reshape(3, 4)
print("Tensor:\n", x)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

# Single element
element = x[1, 2]  # 6

# Row
row = x[1, :]  # tensor([4, 5, 6, 7])

# Column
col = x[:, 2]  # tensor([2, 6, 10])

# Submatrix
submatrix = x[0:2, 1:3]
# tensor([[1, 2],
#         [5, 6]])

print("Element at [1, 2]:", element)
print("Row 1:", row)
print("Column 2:", col)
print("Submatrix [0:2, 1:3]:\n", submatrix)
```

### Advanced Indexing

```python
x = torch.arange(24).reshape(4, 6)
print("Tensor:\n", x)

# Index with a list
selected_rows = x[[0, 2], :]
print("Rows 0 and 2:\n", selected_rows)

# Index with a tensor
indices = torch.tensor([0, 2, 5])
selected_cols = x[:, indices]
print("Columns 0, 2, 5:\n", selected_cols)

# Boolean indexing
mask = x > 10
print("Boolean mask:\n", mask)
values = x[mask]
print("Values > 10:", values)
```

### Ellipsis and Newaxis

```python
x = torch.randn(2, 3, 4, 5)

# Ellipsis (...) expands to fill all dimensions
# Take all elements in dimensions 1 and 2, first element in dim 0, last in dim 3
result = x[0, ..., -1]
print("Shape with ellipsis:", result.shape)  # torch.Size([3, 4])

# None adds a new dimension (similar to unsqueeze)
result = x[:, None, :, :, :]
print("Shape with None:", result.shape)  # torch.Size([2, 1, 3, 4, 5])
```

## Dimension Manipulation

### Stack and Cat

```python
# Stack: Join tensors along a new dimension
x1 = torch.tensor([1, 2, 3])
x2 = torch.tensor([4, 5, 6])

# Stack along dimension 0 (default)
stacked = torch.stack([x1, x2], dim=0)
print("Stacked (dim=0):\n", stacked)  # shape: (2, 3)

# Stack along dimension 1
stacked = torch.stack([x1, x2], dim=1)
print("Stacked (dim=1):\n", stacked)  # shape: (3, 2)

# Cat: Join tensors along an existing dimension
x1 = torch.randn(2, 3)
x2 = torch.randn(2, 3)

# Concatenate along dimension 0
cat_dim0 = torch.cat([x1, x2], dim=0)
print("Cat (dim=0) shape:", cat_dim0.shape)  # (4, 3)

# Concatenate along dimension 1
cat_dim1 = torch.cat([x1, x2], dim=1)
print("Cat (dim=1) shape:", cat_dim1.shape)  # (2, 6)
```

### Split and Chunk

```python
x = torch.arange(12).reshape(3, 4)
print("Original:\n", x)

# Split into sections along dimension 0
split_dim0 = torch.split(x, split_size_or_sections=1, dim=0)
print("Split into 3 parts:")
for i, part in enumerate(split_dim0):
    print(f"Part {i}:\n{part}")

# Split into unequal sections
split_unequal = torch.split(x, [1, 2], dim=0)
print("\nSplit into [1, 2] sections:")
for i, part in enumerate(split_unequal):
    print(f"Part {i} shape: {part.shape}")

# Chunk: split into equal-sized chunks (last chunk may be smaller)
chunks = torch.chunk(x, chunks=2, dim=1)
print("\nChunked into 2 along dim 1:")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i} shape: {chunk.shape}")
```

### Unbind

```python
x = torch.arange(12).reshape(3, 4)
print("Original:\n", x)

# Unbind removes a dimension and returns a tuple of tensors
unbound = torch.unbind(x, dim=0)
print("\nUnbound along dim 0:")
for i, tensor in enumerate(unbound):
    print(f"Tensor {i}: {tensor}")
```

## Device Management

### Moving Tensors Between Devices

```python
# Create tensor on CPU
x = torch.randn(3, 4)
print("Device:", x.device)  # cpu

# Check if GPU is available
if torch.cuda.is_available():
    # Move to GPU
    x_gpu = x.to('cuda')
    print("Device:", x_gpu.device)  # cuda:0

    # Move back to CPU
    x_cpu = x_gpu.to('cpu')
    # OR
    x_cpu = x_gpu.cpu()
    print("Device:", x_cpu.device)  # cpu

    # Create directly on GPU
    y = torch.randn(3, 4, device='cuda')
    print("Device:", y.device)  # cuda:0

    # Specify GPU index
    if torch.cuda.device_count() > 1:
        y = torch.randn(3, 4, device='cuda:1')
        print("Device:", y.device)  # cuda:1
```

### Device-Specific Operations

```python
# Tensors must be on the same device for operations
x = torch.randn(3, 4).to('cuda' if torch.cuda.is_available() else 'cpu')
y = torch.randn(3, 4).to('cuda' if torch.cuda.is_available() else 'cpu')

# This works because both are on the same device
result = x + y
print("Operation successful:", result.shape)

# Common pattern: ensure tensors are on the same device
def ensure_same_device(*tensors):
    """Move all tensors to the device of the first tensor"""
    device = tensors[0].device
    return [t.to(device) for t in tensors]

x, y = ensure_same_device(x, y)
result = x + y
```

### PyTorch 2.0+ Device Management

```python
# PyTorch 2.0+: Set default device for cleaner code
if torch.cuda.is_available():
    # Set CUDA as the default device
    torch.set_default_device('cuda')

    # All subsequent tensors are created on CUDA by default
    x = torch.randn(3, 4)  # Automatically on GPU!
    y = torch.randn(3, 4)  # Automatically on GPU!

    print(f"x device: {x.device}")  # cuda:0
    print(f"y device: {y.device}")  # cuda:0

    # No need for explicit .to('cuda') calls
    result = x + y  # Works automatically

    # Reset to CPU if needed
    torch.set_default_device('cpu')
    z = torch.randn(2, 3)
    print(f"z device: {z.device}")  # cpu
```

> **Note:** `torch.set_default_device()` is available in PyTorch 2.0+. For earlier versions, use explicit `.to(device)` calls as shown in the examples above.

## Practical Examples

### Example 1: Preparing Image Batches

```python
# Simulate loading images (each is 3x224x224)
images = [torch.randn(3, 224, 224) for _ in range(8)]

# Stack into a batch
batch = torch.stack(images, dim=0)
print("Batch shape:", batch.shape)  # torch.Size([8, 3, 224, 224])

# Common pattern: add batch dimension to single image
single_image = torch.randn(3, 224, 224)
single_batch = single_image.unsqueeze(dim=0)
print("Single batch shape:", single_batch.shape)  # torch.Size([1, 3, 224, 224])
```

### Example 2: Changing Image Layout

```python
# HWC to CHW conversion (Height-Width-Channels to Channels-Height-Width)
hwc_image = torch.randn(224, 224, 3)  # Common in PIL/OpenCV
chw_image = hwc_image.permute(2, 0, 1)
print("HWC shape:", hwc_image.shape)  # torch.Size([224, 224, 3])
print("CHW shape:", chw_image.shape)  # torch.Size([3, 224, 224])

# CHW to HWC
hwc_again = chw_image.permute(1, 2, 0)
print("Back to HWC:", hwc_again.shape)  # torch.Size([224, 224, 3])
```

### Example 3: Working with Sequences

```python
# Simulate sequences of different lengths
seq1 = torch.randn(10, 5)   # Length 10, 5 features
seq2 = torch.randn(15, 5)   # Length 15, 5 features
seq3 = torch.randn(8, 5)    # Length 8, 5 features

# Pad sequences to same length
max_len = 15
seq1_padded = torch.nn.functional.pad(seq1, (0, 0, 0, max_len - seq1.shape[0]))
seq2_padded = seq2  # Already max length
seq3_padded = torch.nn.functional.pad(seq3, (0, 0, 0, max_len - seq3.shape[0]))

# Stack into batch
batch = torch.stack([seq1_padded, seq2_padded, seq3_padded], dim=0)
print("Batch shape:", batch.shape)  # torch.Size([3, 15, 5])
```

### Example 4: Batching Data

```python
# Simulate data samples
data = [torch.randn(5) for _ in range(100)]

# Create batches
batch_size = 10
num_batches = len(data) // batch_size

batches = []
for i in range(num_batches):
    batch = torch.stack(data[i * batch_size : (i + 1) * batch_size], dim=0)
    batches.append(batch)

print(f"Created {len(batches)} batches")
print("First batch shape:", batches[0].shape)  # torch.Size([10, 5])
print("Last batch shape:", batches[-1].shape)  # torch.Size([10, 5])
```

## Key Takeaways

| Operation | Description | Common Use Case |
|-----------|-------------|-----------------|
| `reshape()` | Change tensor shape | Preparing data for layers |
| `view()` | Non-copy reshape | Memory-efficient reshaping |
| `squeeze()`/`unsqueeze()` | Add/remove size-1 dims | Batch/channel operations |
| `transpose()`/`permute()` | Swap dimensions | Image format conversion |
| `stack()` | Join along new dim | Creating batches |
| `cat()` | Join along existing dim | Concatenating sequences |
| `split()`/`chunk()` | Divide into parts | Processing large batches |
| `to(device)` | Move to GPU/CPU | Device management |

## Practice Exercises

1. Create a tensor of shape (64, 1, 28, 28) and remove the singleton dimension
2. Given a batch of images with shape (32, 3, 224, 224), extract the first 5 images
3. Create two tensors of shape (10, 5) and concatenate them along dimension 1
4. Implement a function that converts HWC format to CHW format and vice versa
5. Given a tensor of shape (2, 3, 4, 5), permute it to (5, 4, 2, 3)

## Next Steps

- [Tensor Basics](tensor-basics.md) - Tensor creation and attributes
- [Tensor Operations](tensor-operations.md) - Mathematical and comparison operations
- [Module 2: PyTorch Workflow Fundamentals](../module-02/README.md) - Data preparation and model building

---

**Last Updated**: January 2026
