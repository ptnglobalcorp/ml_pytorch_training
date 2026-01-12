# Tensor Operations

## Learning Objectives

By the end of this lesson, you will be able to:
- Perform basic mathematical operations on tensors
- Understand broadcasting rules in PyTorch
- Use comparison and logical operations
- Apply reduction operations
- Perform linear algebra operations

## Basic Mathematical Operations

### Element-wise Operations

PyTorch supports standard arithmetic operations that work element-wise:

```python
import torch

x = torch.tensor([1, 2, 3, 4])
y = torch.tensor([5, 6, 7, 8])

# Addition
add = x + y
# OR
add = torch.add(x, y)

# Subtraction
sub = x - y
# OR
sub = torch.sub(x, y)

# Multiplication (element-wise)
mul = x * y
# OR
mul = torch.mul(x, y)

# Division
div = x / y
# OR
div = torch.div(x, y)

print("Addition:", add)      # tensor([ 6,  8, 10, 12])
print("Subtraction:", sub)   # tensor([-4, -4, -4, -4])
print("Multiplication:", mul) # tensor([ 5, 12, 21, 32])
print("Division:", div)      # tensor([0.2000, 0.3333, 0.4286, 0.5000])
```

### Scalar Operations

```python
tensor = torch.tensor([1, 2, 3, 4])

# Addition with scalar
result = tensor + 10  # tensor([11, 12, 13, 14])

# Multiplication with scalar
result = tensor * 2   # tensor([2, 4, 6, 8])

# Power
result = tensor ** 2  # tensor([ 1,  4,  9, 16])
```

### In-place Operations

Operations ending with `_` modify the tensor in-place (save memory):

```python
x = torch.tensor([1, 2, 3, 4])

# In-place addition
x.add_(5)  # x is now [6, 7, 8, 9]

# In-place multiplication
x.mul_(2)  # x is now [12, 14, 16, 18]

print(x)  # tensor([12, 14, 16, 18])
```

> **Warning**: In-place operations can be problematic with autograd. Use them carefully during gradient computation.

## Advanced Mathematical Operations

### Basic Math Functions

```python
x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)

# Absolute value
abs_x = torch.abs(x)

# Square root
sqrt_x = torch.sqrt(x)

# Power (x^2)
pow_x = torch.pow(x, 2)

# Exponential
exp_x = torch.exp(x)

# Logarithm (natural log)
log_x = torch.log(x)

# Log base 10
log10_x = torch.log10(x)

# Round
round_x = torch.round(x)

# Floor
floor_x = torch.floor(x)

# Ceiling
ceil_x = torch.ceil(x)

print("Absolute:", abs_x)
print("Square root:", sqrt_x)
print("Exponential:", exp_x)
print("Logarithm:", log_x)
```

### Trigonometric Functions

```python
x = torch.tensor([0, torch.pi/2, torch.pi], dtype=torch.float32)

# Sine
sin_x = torch.sin(x)

# Cosine
cos_x = torch.cos(x)

# Tangent
tan_x = torch.tan(x)

print("Sine:", sin_x)
print("Cosine:", cos_x)
```

### Statistical Operations

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

# Mean
mean_all = torch.mean(x)           # 3.5
mean_dim0 = torch.mean(x, dim=0)   # tensor([2.5, 3.5, 4.5])
mean_dim1 = torch.mean(x, dim=1)   # tensor([2., 5.])

# Sum
sum_all = torch.sum(x)             # 21
sum_dim0 = torch.sum(x, dim=0)     # tensor([5., 7., 9.])
sum_dim1 = torch.sum(x, dim=1)     # tensor([ 6., 15.])

# Min/Max
min_val = torch.min(x)             # 1.
max_val = torch.max(x)             # 6.

# Min/Max with indices
min_val, min_idx = torch.min(x, dim=1)
max_val, max_idx = torch.max(x, dim=1)

# Standard deviation
std_x = torch.std(x)

# Variance
var_x = torch.var(x)

# Median
median_x = torch.median(x)

# Product
prod_x = torch.prod(x)  # 720
```

## Comparison Operations

### Element-wise Comparisons

```python
x = torch.tensor([1, 2, 3, 4])
y = torch.tensor([2, 2, 2, 5])

# Equal
eq = x == y  # tensor([False,  True, False, False])

# Not equal
ne = x != y  # tensor([ True, False,  True,  True])

# Greater than
gt = x > y   # tensor([False, False,  True, False])

# Less than
lt = x < y   # tensor([ True, False, False,  True])

# Greater or equal
ge = x >= y  # tensor([False,  True,  True, False])

# Less or equal
le = x <= y  # tensor([ True,  True, False,  True])
```

### Special Comparison Operations

```python
x = torch.tensor([1, 2, 3, 4])

# Element-wise minimum
min_result = torch.minimum(x, torch.tensor([3, 3, 3, 3]))
# tensor([1, 2, 3, 3])

# Element-wise maximum
max_result = torch.maximum(x, torch.tensor([3, 3, 3, 3]))
# tensor([3, 3, 3, 4])

# Clamp values to a range
clamped = torch.clamp(x, min=2, max=3)
# tensor([2, 2, 3, 3])
```

## Reduction Operations

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

# Sum all elements
sum_all = torch.sum(x)  # 21.0

# Sum along specific dimension
sum_dim0 = torch.sum(x, dim=0)  # tensor([5., 7., 9.])
sum_dim1 = torch.sum(x, dim=1)  # tensor([ 6., 15.])

# Keep dimensions (useful for broadcasting)
sum_dim0_kept = torch.sum(x, dim=0, keepdim=True)
# tensor([[5., 7., 9.]])

# Count non-zero elements
nonzero = torch.nonzero(x)

# Number of non-zero elements
count_nonzero = torch.count_nonzero(x)
```

## Linear Algebra Operations

### Matrix Operations

```python
# Matrix multiplication
A = torch.randn(3, 4)
B = torch.randn(4, 5)

# Method 1: Using @ operator (recommended)
C = A @ B  # Shape: (3, 5)

# Method 2: Using torch.matmul
C = torch.matmul(A, B)

# Method 3: Using torch.mm (for 2D tensors only)
C = torch.mm(A, B)

print("Matrix A shape:", A.shape)  # torch.Size([3, 4])
print("Matrix B shape:", B.shape)  # torch.Size([4, 5])
print("Result C shape:", C.shape)  # torch.Size([3, 5])
```

### Vector Operations

```python
# Dot product
v1 = torch.tensor([1, 2, 3])
v2 = torch.tensor([4, 5, 6])

dot_product = torch.dot(v1, v2)  # 32

# Outer product
outer = torch.outer(v1, v2)
# tensor([[ 4,  5,  6],
#         [ 8, 10, 12],
#         [12, 15, 18]])
```

### Matrix Decompositions

```python
# Transpose
A = torch.randn(3, 4)
A_T = A.T  # or torch.transpose(A, 0, 1)

# Singular Value Decomposition
A = torch.randn(3, 3)
U, S, V = torch.svd(A)

# Eigenvalues and eigenvectors (symmetric matrices only)
eigenvalues, eigenvectors = torch.linalg.eig(A.T @ A)

# Matrix inverse (square matrices)
A = torch.randn(3, 3)
A_inv = torch.inverse(A)

# Matrix determinant
det = torch.det(A)

# Matrix rank
rank = torch.matrix_rank(A)
```

## Broadcasting

Broadcasting allows operations on tensors of different shapes:

```python
# Scalar broadcasting
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
result = x + 5
# tensor([[ 6,  7,  8],
#         [ 9, 10, 11]])

# Vector broadcasting
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
y = torch.tensor([10, 20, 30])
result = x + y
# tensor([[11, 22, 33],
#         [14, 25, 36]])

# Compatible shapes
x = torch.randn(3, 1, 2)  # Shape: (3, 1, 2)
y = torch.randn(   5, 2)  # Shape:    (5, 2)
result = x + y             # Shape: (3, 5, 2)
```

### Broadcasting Rules

| Shape 1 | Shape 2 | Result | Compatible? |
|---------|---------|--------|-------------|
| `(3, 4)` | `(3, 4)` | `(3, 4)` | ✅ Identical |
| `(3, 1)` | `(3, 4)` | `(3, 4)` | ✅ One dimension is 1 |
| `(1, 4)` | `(3, 4)` | `(3, 4)` | ✅ One dimension is 1 |
| `(3, 4)` | `(1,)` | `(3, 4)` | ✅ Missing dimension treated as 1 |
| `(3, 4)` | `(4, 3)` | - | ❌ Incompatible |

> **Tip**: Broadcasting follows NumPy's rules: dimensions are compatible when they are equal, or one of them is 1.

## Practical Examples

### Example 1: Normalizing Data

```python
# Normalize to zero mean and unit variance
data = torch.randn(100, 10)  # 100 samples, 10 features

# Calculate mean and std
mean = torch.mean(data, dim=0)
std = torch.std(data, dim=0)

# Normalize
normalized = (data - mean) / std

print("Normalized mean:", torch.mean(normalized, dim=0))  # Should be ~0
print("Normalized std:", torch.std(normalized, dim=0))    # Should be ~1
```

### Example 2: Computing Cosine Similarity

```python
def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    dot_product = torch.dot(a, b)
    norm_a = torch.norm(a)
    norm_b = torch.norm(b)
    return dot_product / (norm_a * norm_b)

v1 = torch.tensor([1.0, 2.0, 3.0])
v2 = torch.tensor([4.0, 5.0, 6.0])

similarity = cosine_similarity(v1, v2)
print(f"Cosine similarity: {similarity:.4f}")
```

### Example 3: Softmax Activation

```python
def softmax(x):
    """Compute softmax values for each set of scores in x"""
    exp_x = torch.exp(x)
    return exp_x / torch.sum(exp_x, dim=1, keepdim=True)

# Batch of logits
logits = torch.tensor([[2.0, 1.0, 0.1],
                       [1.0, 3.0, 0.2]])

probabilities = softmax(logits)
print("Probabilities:")
print(probabilities)
print("Sum:", torch.sum(probabilities, dim=1))  # Should sum to 1
```

## Key Takeaways

| Operation Category | Common Functions |
|--------------------|------------------|
| **Basic Arithmetic** | `+`, `-`, `*`, `/`, `torch.add`, `torch.mul` |
| **Math Functions** | `torch.sqrt`, `torch.exp`, `torch.log`, `torch.pow` |
| **Statistics** | `torch.mean`, `torch.std`, `torch.min`, `torch.max` |
| **Comparison** | `==`, `!=`, `<`, `>`, `torch.minimum`, `torch.maximum` |
| **Linear Algebra** | `@`, `torch.matmul`, `torch.mm`, `torch.inverse` |
| **Reductions** | `torch.sum`, `torch.prod`, `torch.count_nonzero` |

## Practice Exercises

1. Create two tensors of shape (3, 4) with random values and compute their element-wise product
2. Implement a function that computes the L2 norm of a tensor
3. Create a function that normalizes a batch of data using min-max scaling
4. Given a batch of logits (shape: [batch_size, num_classes]), compute softmax probabilities
5. Implement matrix multiplication for a batch of matrices: A shape (batch, m, n) @ B shape (batch, n, p)

## Next Steps

- [Tensor Basics](tensor-basics.md) - Understanding tensor creation and attributes
- [Tensor Manipulation](tensor-manipulation.md) - Reshaping, indexing, and slicing

---

**Last Updated**: January 2026
