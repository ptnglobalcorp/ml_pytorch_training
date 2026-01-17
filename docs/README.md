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

## Module 1: Deep Learning Foundations with PyTorch

**Goal**: Build conceptual understanding of deep learning and master PyTorch tensor fundamentals

### Study Path

**Part 1: Conceptual Foundation**

| Order | Topic | Description | Lab Location |
|-------|-------|-------------|--------------|
| 1 | Introduction to Deep Learning | The paradigm shift, when to use DL, when to avoid it | N/A (Conceptual) |
| 2 | Neural Network Anatomy | Key components and the foundational workflow | N/A (Conceptual) |
| 3 | Learning Methodology | Strategies for effective deep learning study | N/A (Conceptual) |

**Part 2: PyTorch Technical Essentials**

| Order | Topic | Description | Lab Location |
|-------|-------|-------------|--------------|
| 4 | PyTorch Essentials | What is PyTorch, setup, 2.0 features | [`module-01/pytorch-fundamentals/`](../module-01/pytorch-fundamentals/) |
| 5 | Tensor Foundations | Creation, critical attributes (shape, rank, device) | [`module-01/pytorch-fundamentals/`](../module-01/pytorch-fundamentals/) |
| 6 | Tensor Operations | Math operations, matrix multiplication, aggregation | [`module-01/pytorch-fundamentals/`](../module-01/pytorch-fundamentals/) |
| 7 | Tensor Manipulation | Reshaping, indexing, device management | [`module-01/pytorch-fundamentals/`](../module-01/pytorch-fundamentals/) |

### Module 1 Documentation

**Part 1: Conceptual Foundation**
- [Introduction to Deep Learning](module-01/01-introduction-to-deep-learning.md)
- [Neural Network Anatomy](module-01/02-neural-network-anatomy.md)
- [Learning Methodology](module-01/03-learning-methodology.md)

**Part 2: PyTorch Technical Essentials**
- [PyTorch Essentials](module-01/04-pytorch-essentials.md)
- [Tensor Foundations](module-01/05-tensor-foundations.md)
- [Tensor Operations](module-01/06-tensor-operations.md)
- [Tensor Manipulation](module-01/07-tensor-manipulation.md)
- [Exercises Quick Reference](module-01/exercises.md)

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

### Module 1: Deep Learning Foundations
- [ ] Introduction to Deep Learning
- [ ] Neural Network Anatomy
- [ ] Learning Methodology
- [ ] PyTorch Essentials
- [ ] Tensor Foundations
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
