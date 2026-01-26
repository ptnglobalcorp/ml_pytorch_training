# Module 3: PyTorch Neural Network Classification

**Learn to build, train, and evaluate neural network classifiers**

## Quick Start

1. **Read the conceptual guides** in Part 1 (start with Classification Introduction)
2. **Practice with hands-on exercises** in [`../../module-03/neural-network-classification/`](../../module-03/neural-network-classification/)

```
Learn:  docs/module-03/                            →  Theory and concepts
Do:     module-03/neural-network-classification/   →  Hands-on exercises
```

## Module Overview

This module teaches you how to build, train, and evaluate neural network classifiers for a variety of classification tasks. You'll progress from simple binary classification to multi-class problems, mastering the complete workflow from data preparation through model evaluation.

**What we'll build:**
- Binary classifiers (e.g., separating circles)
- Multi-class classifiers (e.g., classifying blobs)
- Non-linear models with decision boundary visualization

## Learning Objectives

By the end of this module, you will be able to:

### Classification Fundamentals
- Understand the difference between binary, multi-class, and multi-label classification
- Choose appropriate loss functions (BCEWithLogitsLoss vs CrossEntropyLoss)
- Design neural network architectures for classification tasks

### Data & Visualization
- Create synthetic classification datasets (make_circles, make_blobs)
- Visualize data and decision boundaries
- Split data into train/test sets

### Model Building
- Build models by subclassing `nn.Module`
- Implement the 5-step training loop
- Use activation functions (ReLU, Sigmoid, Softmax)

### Evaluation
- Convert logits → probabilities → labels
- Implement proper inference patterns
- Calculate and interpret metrics (accuracy, precision, recall, F1)

## Study Path

### Part 1: Foundations

Master the essential concepts of classification.

| # | Topic | Description | Practice |
|---|-------|-------------|----------|
| 1 | [Classification Introduction](01-classification-intro.md) | Types of classification, device-agnostic code | [`01_binary_classification_intro.py`](../../module-03/neural-network-classification/01_binary_classification_intro.py) |
| 2 | [Architecture Components](02-architecture-components.md) | Input/output shapes, activations, loss functions | - |
| 3 | [Data Preparation](03-data-preparation.md) | make_circles, make_blobs, visualization | [`01_binary_classification_intro.py`](../../module-03/neural-network-classification/01_binary_classification_intro.py) (Parts 1-4) |

### Part 2: Building & Training

Learn to build and train classification models.

| # | Topic | Description | Practice |
|---|-------|-------------|----------|
| 4 | [Building Models](04-building-models.md) | nn.Module, nn.Sequential, CircleModelV0 | [`01_binary_classification_intro.py`](../../module-03/neural-network-classification/01_binary_classification_intro.py) (Parts 5-6) |
| 5 | [Training & Evaluation](05-training-evaluation.md) | Training loop, logits→labels, inference | [`02_training_and_predictions.py`](../../module-03/neural-network-classification/02_training_and_predictions.py) |
| 6 | [Improving Models](06-improving-models.md) | Non-linearity, ReLU, decision boundaries | [`03_non_linear_classification.py`](../../module-03/neural-network-classification/03_non_linear_classification.py) |

### Part 3: Multi-Class & Metrics

Complete your classification toolkit.

| # | Topic | Description | Practice |
|---|-------|-------------|----------|
| 7 | [Multi-Class Classification](04-building-models.md#multi-class-classification) | make_blobs, CrossEntropyLoss, argmax | [`04_multi_class_classification.py`](../../module-03/neural-network-classification/04_multi_class_classification.py) |
| 8 | [Evaluation Metrics](07-evaluation-metrics.md) | torchmetrics, precision/recall/F1, confusion matrix | [`05_evaluation_metrics.py`](../../module-03/neural-network-classification/05_evaluation_metrics.py) |

### Exercises

- [Exercises Quick Reference](exercises.md) - Overview of all hands-on exercises

## The Learning Mottos

Throughout this module, apply these three core principles:

### 1. If in doubt, run the code!

Don't just read about classification—execute it. Seeing the decision boundary develop builds intuition faster than studying theory.

**What this means in practice:**
- Run code examples even if you don't fully understand them yet
- Modify hyperparameters and see how the decision boundary changes
- Print intermediate values to understand what's happening
- Visualize the model's predictions at every stage

### 2. Experiment, experiment, experiment!

Active learning creates deeper understanding than passive reading. Don't be afraid to break things—that's how you learn the boundaries.

**What this means in practice:**
- Try different loss functions and observe the differences
- Change the number of hidden layers and units
- Experiment with different activation functions
- Intentionally create bugs to understand error messages

### 3. Visualize, visualize, visualize!

Classification problems are inherently visual. Decision boundaries reveal what your model has learned.

**What this means in practice:**
- Plot your data before training
- Visualize decision boundaries after training
- Plot training loss curves
- Use different colors for different classes
- Always label your axes and add titles

## Prerequisites

- **Module 1 completed**: PyTorch Fundamentals
- **Module 2 completed**: PyTorch Workflow Fundamentals
- **PyTorch installed**: [Install Guide](https://pytorch.org/get-started/locally/)
- **scikit-learn installed**: `pip install scikit-learn` (for datasets)
- **matplotlib installed**: `pip install matplotlib` (for visualizations)
- **torchmetrics installed**: `pip install torchmetrics` (for evaluation)

## Running the Exercises

```bash
cd module-03/neural-network-classification
python 01_binary_classification_intro.py
python 02_training_and_predictions.py
python 03_non_linear_classification.py
python 04_multi_class_classification.py
python 05_evaluation_metrics.py
python 06_complete_classification_workflow.py
```

## Key Concepts

### Types of Classification

| Type | Classes | Example | Loss Function | Output Shape |
|------|---------|---------|---------------|--------------|
| **Binary** | 2 | Spam vs Not Spam | BCEWithLogitsLoss | 1 |
| **Multi-class** | 3+ | Digit 0-9 | CrossEntropyLoss | num_classes |
| **Multi-label** | N | Tagging articles | BCEWithLogitsLoss | num_labels |

### Activation Functions

| Layer | Function | Range | Purpose |
|-------|----------|-------|---------|
| **Hidden** | ReLU | [0, ∞) | Add non-linearity |
| **Output (binary)** | Sigmoid | [0, 1] | Convert to probability |
| **Output (multi-class)** | Softmax | [0, 1], sum=1 | Class probabilities |

### The Logits → Labels Pipeline

```
Raw Logits → Activation → Probabilities → Decision → Labels
    Model    Sigmoid/Softmax   [0,1] values   Threshold/Argmax  0,1,2...
```

**Binary Classification:**
```python
logits = model(X)           # Raw output
probs = torch.sigmoid(logits)  # Convert to [0,1]
labels = (probs > 0.5).long()  # Threshold at 0.5
```

**Multi-class Classification:**
```python
logits = model(X)           # Raw output
probs = torch.softmax(logits, dim=1)  # Convert to probabilities
labels = torch.argmax(probs, dim=1)    # Pick highest probability
```

### The 5-Step Training Loop

```python
# 1. Forward pass
y_pred = model(X_train)

# 2. Calculate loss
loss = criterion(y_pred, y_train)

# 3. Zero gradients
optimizer.zero_grad()

# 4. Backward pass
loss.backward()

# 5. Update parameters
optimizer.step()
```

## Module Summary

After completing this module, you should:

- **Understand** the difference between binary and multi-class classification
- **Be able to** build neural network classifiers from scratch
- **Know how to** visualize decision boundaries
- **Be ready** for Module 4: Computer Vision with CNNs

## Additional Resources

### External References
- [PyTorch Classification Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [scikit-learn Datasets](https://scikit-learn.org/stable/datasets.html)
- [torchmetrics Documentation](https://torchmetrics.readthedocs.io/)

### Internal Documentation
- [Complete Study Guide](../README.md) - Overall training navigation
- [Module 1: PyTorch Fundamentals](../module-01/README.md) - Prerequisite concepts
- [Module 2: PyTorch Workflow](../module-02/README.md) - Prerequisite concepts

## Next Steps

1. **Start with** [Classification Introduction](01-classification-intro.md) to understand the types of classification
2. **Complete all exercises** in the `neural-network-classification/` directory
3. **Review the key concepts** summary above
4. **Move to Module 4:** Computer Vision with CNNs (coming soon)

---

**Module Overview:** [../../module-03/](../../module-03/)

**Last Updated:** January 2026
