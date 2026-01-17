# Module 1: Deep Learning Foundations with PyTorch

**Build conceptual understanding and master PyTorch tensor fundamentals**

## Quick Start

1. **Read the conceptual guides** in Part 1 (start with Introduction to Deep Learning)
2. **Practice with hands-on exercises** in [`../../module-01/pytorch-fundamentals/`](../../module-01/pytorch-fundamentals/)

```
Learn:  docs/module-01/                  →  Theory and concepts
Do:     module-01/pytorch-fundamentals/   →  Hands-on exercises
```

## Module Overview

This module provides the foundation for deep learning with PyTorch, starting with conceptual understanding before diving into technical implementation. You'll learn when to use deep learning, how neural networks work, and master PyTorch tensors—the building blocks of all deep learning systems.

## Learning Objectives

By the end of this module, you will be able to:

### Conceptual Foundation
- Understand when and why to use deep learning
- Identify key components of neural networks (input, hidden, output layers)
- Apply effective learning methodologies for deep learning study

### Technical Essentials
- Create and manipulate PyTorch tensors
- Perform tensor operations (arithmetic, linear algebra, aggregations)
- Manage tensors across CPU and GPU devices

## Study Path

### Part 1: Conceptual Foundation (Start Here)

Build your understanding before writing code.

| # | Topic | Description |
|---|-------|-------------|
| 1 | [Introduction to Deep Learning](01-introduction-to-deep-learning.md) | The paradigm shift, when to use DL, when to avoid it |
| 2 | [Neural Network Anatomy](02-neural-network-anatomy.md) | Key components and the foundational workflow |
| 3 | [Learning Methodology](03-learning-methodology.md) | Strategies for effective deep learning study |

### Part 2: PyTorch Technical Essentials

Hands-on work with PyTorch tensors.

| # | Topic | Description | Practice |
|---|-------|-------------|----------|
| 4 | [PyTorch Essentials](04-pytorch-essentials.md) | What is PyTorch, setup, 2.0 features | [`01_tensor_creation.py`](../../module-01/pytorch-fundamentals/01_tensor_creation.py) |
| 5 | [Tensor Foundations](05-tensor-foundations.md) | Creation, critical attributes (shape, rank, device) | [`01_tensor_creation.py`](../../module-01/pytorch-fundamentals/01_tensor_creation.py) |
| 6 | [Tensor Operations](06-tensor-operations.md) | Math operations, matrix multiplication, aggregation | [`02_tensor_operations.py`](../../module-01/pytorch-fundamentals/02_tensor_operations.py) |
| 7 | [Tensor Manipulation](07-tensor-manipulation.md) | Reshaping, indexing, device management | [`03_tensor_manipulation.py`](../../module-01/pytorch-fundamentals/03_tensor_manipulation.py) |

### Exercises

- [Exercises Quick Reference](exercises.md) - Overview of all hands-on exercises

## Prerequisites

- **Python 3.8+** familiarity with basic syntax
- **PyTorch installed** ([Install Guide](https://pytorch.org/get-started/locally/))

> **PyTorch 2.0:** All code in this module is compatible with PyTorch 2.0. If you have PyTorch 1.x, everything will still work. See [PyTorch Essentials](04-pytorch-essentials.md) for 2.0 highlights.

## Running the Exercises

```bash
cd module-01/pytorch-fundamentals
python 01_tensor_creation.py
python 02_tensor_operations.py
python 03_tensor_manipulation.py
```

## Key Concepts

### When to Use Deep Learning

| Use Deep Learning When... | Avoid Deep Learning When... |
|---------------------------|-----------------------------|
| Rules are too complex to define manually | Simple rule-based systems work |
| Working with unstructured data (images, text, audio) | Explainability is required |
| Need adaptability to changing environments | Data is scarce |
| Can tolerate some error | Zero error tolerance |

### The Three Tensor Attributes

Every tensor has three critical attributes you must understand:

| Attribute | Description | Example |
|-----------|-------------|---------|
| **Shape** | Size of each dimension | `[32, 3, 224, 224]` = batch of 32 RGB images |
| **Rank (ndim)** | Number of dimensions | `4` dimensions for the batch above |
| **Device** | Where the tensor lives (CPU/GPU) | `cuda:0` or `cpu` |

### The Learning Mottos

1. **If in doubt, run the code!** - Experimentation builds intuition
2. **Experiment, experiment, experiment!** - Active learning creates understanding
3. **Visualize, visualize, visualize!** - Seeing patterns reveals insights

## Additional Resources

### External References
- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch 2.0 Release Notes](https://pytorch.org/blog/PyTorch-2.0-release/)

### Internal Documentation
- [Complete Study Guide](../README.md) - Overall training navigation
- [Module 2: PyTorch Workflow](../module-02/README.md) - Data preparation and model building

## Module Summary

After completing this module, you should:

- **Understand** when deep learning is the right tool for a problem
- **Know** the basic anatomy of a neural network
- **Be comfortable** creating and manipulating PyTorch tensors
- **Be ready** to build your first neural network in Module 2

## Next Steps

1. **Complete all exercises** in the `pytorch-fundamentals/` directory
2. **Review the key concepts** summary above
3. **Move to Module 2:** [PyTorch Workflow Fundamentals](../module-02/README.md)

---

**Module Overview:** [../../module-01/](../../module-01/)

**Last Updated:** January 2026
