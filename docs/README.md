# PyTorch Training Documentation

**Complete study guide for PyTorch deep learning fundamentals**

## Study Path Overview

This documentation follows a **hybrid structure**:
- **`docs/`** (you are here) - Conceptual learning and theory
- **`module-X/`** - Hands-on labs and practice code

## Quick Start

1. **Choose your module** below
2. **Read the conceptual documentation** in `docs/module-X/`
3. **Practice with labs** in `module-X/` folder

---

## Module 1: PyTorch Fundamentals

**Goal**: Master PyTorch tensors, operations, and fundamental tensor manipulation

### Study Path

| Order | Topic | Description | Lab Location |
|-------|-------|-------------|--------------|
| 1 | Introduction to Tensors | What are tensors, tensor creation, and basic properties | [`module-01/pytorch-fundamentals/`](../module-01/pytorch-fundamentals/) |
| 2 | Tensor Operations | Mathematical operations, indexing, and slicing | [`module-01/pytorch-fundamentals/`](../module-01/pytorch-fundamentals/) |
| 3 | Tensor Manipulation | Reshaping, broadcasting, and device handling | [`module-01/pytorch-fundamentals/`](../module-01/pytorch-fundamentals/) |

### Module 1 Documentation

- [Tensor Basics](module-01/tensor-basics.md)
- [Tensor Operations Guide](module-01/tensor-operations.md)
- [Tensor Manipulation](module-01/tensor-manipulation.md)

---

## Module 2: PyTorch Workflow Fundamentals

**Goal**: Learn the complete PyTorch workflow for building and training models

### Study Path

| Order | Topic | Description | Lab Location |
|-------|-------|-------------|--------------|
| 1 | Data Preparation | Loading and preprocessing data with DataLoader | [`module-02/pytorch-workflow/`](../module-02/pytorch-workflow/) |
| 2 | Building Models | Creating neural networks with nn.Module | [`module-02/pytorch-workflow/`](../module-02/pytorch-workflow/) |
| 3 | Training Loop | Loss functions, optimizers, and training loops | [`module-02/pytorch-workflow/`](../module-02/pytorch-workflow/) |
| 4 | Saving & Loading | Model checkpointing and inference | [`module-02/pytorch-workflow/`](../module-02/pytorch-workflow/) |

### Module 2 Documentation

- [Data Preparation Guide](module-02/data-preparation.md)
- [Building Neural Networks](module-02/building-models.md)
- [Training Loop Fundamentals](module-02/training-loop.md)
- [Model Persistence](module-02/model-persistence.md)

---

## Module 3: Neural Network Classification

**Goal**: Build and evaluate classification models with PyTorch

### Study Path

| Order | Topic | Description | Lab Location |
|-------|-------|-------------|--------------|
| 1 | Classification Basics | Binary and multi-class classification concepts | [`module-03/neural-network-classification/`](../module-03/neural-network-classification/) |
| 2 | Architecture Design | Designing neural network architectures for classification | [`module-03/neural-network-classification/`](../module-03/neural-network-classification/) |
| 3 | Training & Evaluation | Training classifiers and evaluating performance | [`module-03/neural-network-classification/`](../module-03/neural-network-classification/) |
| 4 | Model Deployment | Saving and using trained models for inference | [`module-03/neural-network-classification/`](../module-03/neural-network-classification/) |

### Module 3 Documentation

- [Classification Basics](module-03/classification-basics.md)
- [Architecture Design](module-03/architecture-design.md)
- [Training & Evaluation](module-03/training-evaluation.md)
- [Model Deployment](module-03/model-deployment.md)

---

## Study Tips

### For Each Module

1. **Read first** - Start with the conceptual guide in `docs/`
2. **Practice second** - Run the lab exercises in `module-X/`
3. **Experiment** - Modify code and observe changes
4. **Review** - Re-read documentation with practical context

### For Hands-on Skills

1. **Complete all lab exercises** - Don't skip!
2. **Break things intentionally** - Learn to troubleshoot
3. **Build variations** - Modify exercises to solve new problems
4. **Document your learnings** - Keep notes

### Example Study Workflow

```bash
# 1. Read the conceptual guide
cat docs/module-01/tensor-basics.md

# 2. Navigate to the lab
cd module-01/pytorch-fundamentals

# 3. Run the exercises
python 01_tensor_creation.py

# 4. Experiment and learn
python 02_tensor_operations.py

# 5. Build your own variation
# Try creating your own tensor operations
```

---

## Additional Resources

### External References

**PyTorch Official Resources:**
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch GitHub Repository](https://github.com/pytorch/pytorch)

**Deep Learning Fundamentals:**
- [Deep Learning with PyTorch Book](https://pytorch.org/deep-learning-with-pytorch)
- [PyTorch YouTube Channel](https://www.youtube.com/pytorch)

**Python & ML Prerequisites:**
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [Python Machine Learning Book](https://github.com/rasbt/python-machine-learning-book-3rd-edition)

### Internal Tools

- [`module-01/pytorch-fundamentals/`](../module-01/pytorch-fundamentals/) - Tensor operations practice
- [`module-02/pytorch-workflow/`](../module-02/pytorch-workflow/) - Complete workflow labs
- [`module-03/neural-network-classification/`](../module-03/neural-network-classification/) - Classification model labs

---

## Progress Tracking

Track your progress by checking off completed modules:

### Module 1: PyTorch Fundamentals
- [ ] Introduction to Tensors
- [ ] Tensor Operations
- [ ] Tensor Manipulation

### Module 2: PyTorch Workflow Fundamentals
- [ ] Data Preparation
- [ ] Building Models
- [ ] Training Loop
- [ ] Saving & Loading Models

### Module 3: Neural Network Classification
- [ ] Classification Basics
- [ ] Architecture Design
- [ ] Training & Evaluation
- [ ] Model Deployment

---

**Last Updated**: January 2026
