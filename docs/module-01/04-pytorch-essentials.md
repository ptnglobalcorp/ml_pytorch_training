# PyTorch Technical Essentials

## Learning Objectives

By the end of this lesson, you will be able to:
- Understand what PyTorch is and why it's popular
- Recognize key features: GPU acceleration, dynamic computation graphs
- Set up and verify your PyTorch environment
- Understand PyTorch 2.0 compatibility

## What is PyTorch?

**PyTorch** is an open-source machine learning framework developed by Meta AI (formerly Facebook). It's widely used in research and industry for building and training neural networks.

### Definition

PyTorch provides:
- **Tensor computation** with strong GPU acceleration
- **Automatic differentiation** for training neural networks
- **Deep neural networks** built on a tape-based autograd system

Think of PyTorch as NumPy with GPU acceleration and automatic differentiation, plus tools for building neural networks.

### Key Features

| Feature | Description |
|---------|-------------|
| **GPU Acceleration** | First-class CUDA support for NVIDIA GPUs, Apple Silicon (MPS), and other accelerators |
| **Dynamic Computation Graphs** | Define-by-run approach—graphs are built on-the-fly as you execute code |
| **Python-First Design** | Deep integration with Python; feels like native Python code |
| **Research-Favored** | Rapid prototyping and experimentation; preferred by most researchers |
| **Large Ecosystem** | Specialized libraries: torchvision, torchaudio, torchtext, and more |
| **Strong Community** | Active development, extensive documentation, and community support |

### PyTorch vs. Other Frameworks

| Framework | Strengths | Typical Use Cases |
|-----------|-----------|-------------------|
| **PyTorch** | Intuitive, debuggable, research-friendly | Research, rapid prototyping, education |
| **TensorFlow** | Production deployment, TensorFlow Serving | Large-scale production, mobile/embedded |
| **JAX** | Functional programming, JIT compilation | Research requiring maximum performance |

**Why PyTorch for learning?**
- **Intuitive API:** Code reads like Python, not a framework-specific language
- **Easy debugging:** Use Python's debugger and print statements naturally
- **Clear error messages:** Understand what went wrong quickly
- **Active research:** Most new papers publish PyTorch code

## PyTorch 2.0 Highlights

PyTorch 2.0 was released in March 2023 with significant improvements while maintaining **full backward compatibility**.

### What's New in 2.0?

| Feature | Description |
|---------|-------------|
| **torch.compile()** | Just-in-time compilation for 30-200% speedup |
| **Improved Windows Support** | Better performance and compatibility on Windows |
| **Better Device Management** | `torch.set_default_device()` for cleaner code |
| **Backward Compatible** | Your existing PyTorch 1.x code works without changes |

### Should You Care About 2.0?

**For Module 1 (this module):** No special considerations needed. The tensor fundamentals you'll learn are identical between PyTorch 1.x and 2.0.

**For later modules:** You'll learn about `torch.compile()` and other 2.0 features when they become relevant to your work.

> **Important:** All code in this curriculum works with PyTorch 2.0. If you have PyTorch 1.x installed, everything will still work—upgrade when convenient.

## Installation & Setup

### Installing PyTorch

The installation command depends on your system and whether you need GPU support. Visit [pytorch.org](https://pytorch.org/get-started/locally/) for the exact command for your setup.

**Common installation patterns:**

```bash
# CPU-only (simplest, good for learning)
pip install torch torchvision torchaudio

# With CUDA (NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# With ROCm (AMD GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

### Verify Installation

After installing, verify everything works:

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
```

**Expected output:**
```
PyTorch version: 2.1.0+cu118
CUDA available: True
CUDA version: 11.8
GPU device: NVIDIA GeForce RTX 3080
```

### Recommended Setup

**Python Version:** Python 3.10 or later required (PyTorch 2.9+)

**Installation Method:**
- Use **conda** for environment management (recommended for data science)
- Use **pip** if you prefer lighter-weight package management

**Example conda setup:**
```bash
# Create a dedicated environment
conda create -n pytorch python=3.10
conda activate pytorch

# Install PyTorch (command varies by system)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Jupyter Notebook (Optional)

For interactive experimentation, Jupyter notebooks are excellent:

```bash
pip install jupyter
jupyter notebook
```

Then create a new notebook and start with:
```python
import torch
print(torch.__version__)
```

## Why PyTorch for Learning?

PyTorch is particularly well-suited for learning deep learning for several reasons:

### 1. Pythonic and Intuitive

PyTorch code feels like natural Python:

```python
# This looks like regular Python, right?
x = torch.tensor([1, 2, 3, 4])
y = torch.tensor([5, 6, 7, 8])
z = x + y  # Element-wise addition
```

Compared to other frameworks that introduce complex abstractions, PyTorch stays close to Python.

### 2. Debuggable

You can use Python's standard debugging tools:

```python
import pdb; pdb.set_trace()  # Set breakpoint
# Or use your IDE's debugger
```

Print statements work naturally—you can inspect tensors at any point:

```python
x = torch.randn(2, 3)
print(f"Shape: {x.shape}")  # torch.Size([2, 3])
print(f"Values:\n{x}")
```

### 3. Clear Mental Model

PyTorch's dynamic graphs mean what you write is what runs. There's no separate "definition" and "execution" phase:

```python
# This code runs immediately
x = torch.tensor(5.0, requires_grad=True)
y = x ** 2
y.backward()  # Compute gradients
print(x.grad)  # tensor(10.)
```

### 4. Research and Industry Standard

- **Research:** Most papers on arXiv provide PyTorch code
- **Industry:** Companies like Tesla, Meta, Uber use PyTorch at scale
- **Community:** Large ecosystem of tutorials, forums, and open-source projects

## The PyTorch Ecosystem

PyTorch includes several specialized libraries:

| Library | Purpose |
|---------|---------|
| **torch** | Core tensor operations and neural network primitives |
| **torchvision** | Computer vision: datasets, models, image transformations |
| **torchaudio** | Audio processing: datasets, models, transformations |
| **torchtext** (deprecated) | Natural language processing (use Hugging Face instead) |
| **PyTorch Lightning** | High-level training framework (optional) |
| **Hugging Face Transformers** | State-of-the-art NLP models built on PyTorch |

In this curriculum, you'll use `torch` (core) and `torchvision` (for image datasets and models).

## Quick Start Example

Here's a minimal PyTorch program to give you a taste:

```python
import torch

# Create tensors
x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
y = torch.tensor([5, 6, 7, 8], dtype=torch.float32)

# Basic operations
print(f"Addition: {x + y}")
print(f"Multiplication: {x * y}")
print(f"Matrix multiplication: {x @ y.T}")  # .T transposes y

# GPU support (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_gpu = x.to(device)
print(f"x is on: {x_gpu.device}")

# Gradients (for neural network training)
x_grad = torch.tensor([2.0], requires_grad=True)
y = x_grad ** 2
y.backward()
print(f"Gradient of x^2 at x=2: {x_grad.grad}")  # Should be 4
```

**Output:**
```
Addition: tensor([ 6.,  8., 10., 12.])
Multiplication: tensor([ 5., 12., 21., 32.])
Matrix multiplication: 70.0
x is on: cuda:0
Gradient of x^2 at x=2: 4.0
```

## Key Takeaways

| Concept | Summary |
|---------|---------|
| **PyTorch** is an open-source ML framework favored by researchers |
| **Key features** include GPU acceleration, dynamic graphs, and Pythonic design |
| **PyTorch 2.0** adds compilation and improved performance while remaining backward compatible |
| **For learning**, PyTorch's intuitive API and debuggability make it ideal |
| **Installation** varies by system—use pytorch.org for the correct command |

## Next Steps

Now that PyTorch is installed and you understand what it is, let's dive into its fundamental building block:

- [Tensor Foundations](05-tensor-foundations.md) - The multi-dimensional arrays that power all deep learning

## Additional Resources

- [PyTorch Official Documentation](https://pytorch.org/docs/stable/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch GitHub Repository](https://github.com/pytorch/pytorch)

---

**Last Updated**: January 2026
