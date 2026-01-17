# Module 1 Exercises

This page provides a quick reference to all hands-on exercises for Module 1: Deep Learning Foundations with PyTorch.

## Exercise Files Location

All exercise files are located in:
```
module-01/pytorch-fundamentals/
```

## Exercise Map

| Exercise File | Concepts Covered | Documentation Section | Difficulty |
|---------------|------------------|----------------------|------------|
| `01_tensor_creation.py` | Tensor creation, attributes, NumPy integration | [Tensor Foundations](05-tensor-foundations.md) | Beginner |
| `02_tensor_operations.py` | Math operations, matrix multiplication, broadcasting | [Tensor Operations](06-tensor-operations.md) | Beginner |
| `03_tensor_manipulation.py` | Reshaping, indexing, device handling | [Tensor Manipulation](07-tensor-manipulation.md) | Intermediate |

## Running the Exercises

### Option 1: Run Individually

```bash
cd module-01/pytorch-fundamentals
python 01_tensor_creation.py
python 02_tensor_operations.py
python 03_tensor_manipulation.py
```

### Option 2: Run All at Once

```bash
cd module-01/pytorch-fundamentals
python 01_tensor_creation.py && python 02_tensor_operations.py && python 03_tensor_manipulation.py
```

## Exercise Overview

### Exercise 1: Tensor Creation

**File:** `01_tensor_creation.py`

**What you'll learn:**
- Creating tensors from data (scalars, vectors, matrices, 3D tensors)
- Creating tensors with specific values (zeros, ones, ranges, identity)
- Creating random tensors (uniform, normal, integer)
- Working with NumPy arrays
- Understanding tensor attributes (shape, dtype, device, ndim, numel)

**Key exercises:**
1. Create a 4×4 tensor filled with the value 3.14
2. Create a random tensor of shape (5, 5) with values between 0 and 1
3. Create a tensor with values from 10 to 50 (exclusive) with step 5
4. Create a 3D tensor of shape (2, 3, 4) filled with ones
5. Check if a GPU is available and create a tensor on it if possible

### Exercise 2: Tensor Operations

**File:** `02_tensor_operations.py`

**What you'll learn:**
- Basic arithmetic operations (addition, subtraction, multiplication, division)
- Matrix operations (multiplication, transpose)
- Reduction operations (sum, mean, standard deviation)
- Comparison operations
- Broadcasting with different shaped tensors

**Key exercises:**
1. Matrix multiplication: Given A (4×3) and B (3×5), compute C = A @ B
2. Normalize a tensor to have zero mean and unit variance
3. Find the indices of the top 3 values in a tensor
4. Implement softmax function: `softmax(x) = exp(x) / sum(exp(x))`
5. Compute cosine similarity between two vectors

### Exercise 3: Tensor Manipulation

**File:** `03_tensor_manipulation.py`

**What you'll learn:**
- Reshaping tensors (reshape, view, flatten)
- Squeeze and unsqueeze operations
- Transpose and permute dimensions
- Indexing and slicing
- Concatenating and splitting tensors

**Key exercises:**
1. Remove singleton dimension from tensor of shape (64, 1, 28, 28)
2. Add batch dimension to image tensor of shape (3, 224, 224)
3. Convert HWC format to CHW format
4. Extract diagonal elements from a square matrix
5. Stack 10 random images into a batch

## Exercise Tips

### Apply the Learning Methodology

Remember the three mottos from [Learning Methodology](03-learning-methodology.md):

1. **If in doubt, run the code!**
   - Don't just read the exercise files—run them and observe the output
   - Modify values and see how the output changes

2. **Experiment, experiment, experiment!**
   - Try different tensor shapes and values
   - Break the code intentionally to understand error messages
   - Combine operations in new ways

3. **Visualize, visualize, visualize!**
   - Print tensor shapes at every step
   - Use `print()` to inspect intermediate values
   - For complex tensors, print summaries (shape, dtype, device)

### Getting the Most Out of Exercises

**Before coding:**
- Read the relevant documentation section first
- Review the learning objectives
- Understand what the exercise is trying to teach

**While coding:**
- Read the comments and TODO markers
- Run the code frequently to see outputs
- Don't skip the exercises at the end of each file

**After completing:**
- Try the challenge exercises
- Modify the code to test your understanding
- Create your own small variations

## Challenge Exercises

Once you've completed the basic exercises, try these challenges:

### Challenge 1: Tensor Memory Efficiency
```python
# Given: A large tensor of shape (1000, 1000, 1000)
# Task: Reshape it to (1000, 1000000) without copying data
# Hint: Use view() instead of reshape()
```

### Challenge 2: Batch Matrix Multiplication
```python
# Given: Two batches of matrices
# A shape: (10, 5, 3)  # 10 matrices of 5×3
# B shape: (10, 3, 7)  # 10 matrices of 3×7
# Task: Compute batched matrix multiplication
# Expected result shape: (10, 5, 7)
```

### Challenge 3: Device-Aware Operations
```python
# Write a function that multiplies two tensors,
# automatically handling device mismatches:
def safe_multiply(a, b):
    # If a and b are on different devices,
    # move b to a's device before multiplying
    pass
```

### Challenge 4: Efficient Indexing
```python
# Given: A tensor of shape (1000, 100)
# Task: Extract every 10th row efficiently
# (without using a Python loop)
```

### Challenge 5: In-Place Operations
```python
# Research: What's the difference between:
# x = x + 1
# x += 1
# x.add_(1)
#
# Which one creates a new tensor?
# Which one modifies in-place?
# When should you use each?
```

## Troubleshooting

### Common Issues

**Issue: CUDA out of memory**
```python
# Solution: Use smaller tensors or CPU
x = torch.randn(100, 100)  # Instead of (10000, 10000)
```

**Issue: Runtime error: Expected all tensors to be on the same device**
```python
# Solution: Move tensors to the same device
x = x.to('cuda')
y = y.to('cuda')
result = x + y
```

**Issue: Matrices cannot be multiplied (shape mismatch)**
```python
# Check inner dimensions match
A: (m, n)
B: (n, p)  # Inner dimension n must match
result = A @ B  # Shape: (m, p)
```

## Next Steps

After completing all Module 1 exercises:

1. **Review** the key concepts from each documentation section
2. **Build something small**: Create a simple tensor manipulation script
3. **Move to Module 2**: [PyTorch Workflow Fundamentals](../module-02/README.md)

## Additional Practice

Looking for more practice?

- **Kaggle:** Try the [Titanic](https://www.kaggle.com/c/titanic) or [MNIST](https://www.kaggle.com/c/digit-recognizer) datasets
- **PyTorch Tutorials:** Work through [official PyTorch tutorials](https://pytorch.org/tutorials/)
- **Create your own exercises:** Pick a real-world problem and solve it with tensors

---

**Need Help?**

- Check the [PyTorch Documentation](https://pytorch.org/docs/stable/)
- Ask questions in the [PyTorch Forums](https://discuss.pytorch.org/)
- Review the documentation sections linked above

**Last Updated:** January 2026
