# Data Preparation

## Learning Objectives

By the end of this lesson, you will be able to:
- Create synthetic classification datasets using `make_circles` and `make_blobs`
- Visualize classification data with matplotlib
- Split data into training and test sets
- Convert NumPy arrays to PyTorch tensors
- Inspect and understand your data before training

---

## Why Synthetic Datasets?

Synthetic datasets are perfect for learning because:
- **Controlled:** You know the true underlying pattern
- **Visualizable:** 2D data can be plotted and understood
- **Reproducible:** Same data every time with random seed
- **Quick:** No downloading or preprocessing required

**Remember the motto: Visualize, visualize, visualize!**

---

## Binary Classification: make_circles

`make_circles` creates a binary classification dataset with two concentric circles. This is a classic **non-linear** problem that requires non-linear models to solve.

### Creating make_circles Dataset

```python
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

# Make 1000 samples
n_samples = 1000

# Create circles dataset
X, y = make_circles(
    n_samples=n_samples,
    noise=0.03,          # Add random noise
    factor=0.5,          # Ratio of inner to outer circle
    random_state=42      # For reproducibility
)

print(f"X shape: {X.shape}")  # (1000, 2) - 2 features (x, y coordinates)
print(f"y shape: {y.shape}")  # (1000,) - binary labels (0 or 1)
print(f"Class 0: {sum(y == 0)} samples")
print(f"Class 1: {sum(y == 1)} samples")
```

### Visualizing make_circles

```python
plt.figure(figsize=(10, 6))

# Scatter plot with different colors for each class
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', s=10, alpha=0.6)

plt.title('make_circles Binary Classification Dataset', fontsize=14)
plt.xlabel('Feature 1 (x coordinate)', fontsize=12)
plt.ylabel('Feature 2 (y coordinate)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.colorbar(label='Class (0=Outer, 1=Inner)')
plt.tight_layout()
plt.show()
```

**Output:** You'll see two concentric circles - blue points (outer circle, class 0) and red points (inner circle, class 1).

### Understanding the Problem

**Why is this challenging?**

A linear model (straight line) cannot separate these two classes. You need a **non-linear model** with activation functions like ReLU.

```python
# This won't work well:
# Linear model: w1*x + w2*y + b = 0 (straight line decision boundary)

# This will work:
# Non-linear model can learn: x² + y² > r² (circular decision boundary)
```

### make_circles Parameters

| Parameter | Description | Default | Effect |
|-----------|-------------|---------|--------|
| `n_samples` | Number of samples | 100 | More samples = better training |
| `noise` | Standard deviation of Gaussian noise | None | More noise = harder problem |
| `factor` | Scale factor between circles | 0.8 | Smaller = smaller inner circle |
| `random_state` | Random seed | None | Set for reproducibility |

---

## Multi-class Classification: make_blobs

`make_blobs` creates a multi-class dataset with Gaussian clusters. This can be linearly or non-linearly separable depending on the `cluster_std` parameter.

### Creating make_blobs Dataset

```python
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Make 1000 samples with 4 classes
n_samples = 1000
n_classes = 4

# Create blobs dataset
X, y = make_blobs(
    n_samples=n_samples,
    n_features=2,         # 2D for visualization
    centers=n_classes,    # One center per class
    cluster_std=1.5,      # Standard deviation of clusters
    random_state=42
)

print(f"X shape: {X.shape}")  # (1000, 2) - 2 features
print(f"y shape: {y.shape}")  # (1000,) - class labels (0, 1, 2, 3)
print(f"Number of classes: {len(set(y))}")

for class_id in range(n_classes):
    print(f"Class {class_id}: {sum(y == class_id)} samples")
```

### Visualizing make_blobs

```python
plt.figure(figsize=(10, 6))

# Scatter plot with different colors for each class
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', s=10, alpha=0.6)

plt.title('make_blobs Multi-class Classification Dataset', fontsize=14)
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.grid(True, alpha=0.3)
plt.colorbar(label='Class')
plt.tight_layout()
plt.show()
```

**Output:** You'll see 4 distinct clusters, each a different color.

### make_blobs Parameters

| Parameter | Description | Default | Effect |
|-----------|-------------|---------|--------|
| `n_samples` | Number of samples | 100 | Total samples (divided among classes) |
| `n_features` | Number of features | 2 | Must be ≥ centers |
| `centers` | Number of centers (classes) | 3 | Number of classes |
| `cluster_std` | Standard deviation of clusters | 1.0 | Higher = more overlap (harder) |
| `random_state` | Random seed | None | Set for reproducibility |

---

## Data Inspection

Before training, always inspect your data!

### Checking Data Properties

```python
import numpy as np

print("=" * 60)
print("Data Inspection")
print("=" * 60)

# Shape
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Data types
print(f"X dtype: {X.dtype}")
print(f"y dtype: {y.dtype}")

# Value ranges
print(f"\nX range: [{X.min():.3f}, {X.max():.3f}]")
print(f"y values: {np.unique(y)}")

# Class distribution
print("\nClass distribution:")
for class_id in np.unique(y):
    count = sum(y == class_id)
    percentage = count / len(y) * 100
    print(f"  Class {class_id}: {count} samples ({percentage:.1f}%)")

# Feature statistics
print("\nFeature statistics:")
print(f"  Feature 1: mean={X[:, 0].mean():.3f}, std={X[:, 0].std():.3f}")
print(f"  Feature 2: mean={X[:, 1].mean():.3f}, std={X[:, 1].std():.3f}")
```

---

## Train/Test Split

Always split your data into training and test sets to evaluate generalization.

### Splitting the Data

```python
from sklearn.model_selection import train_test_split

# Split into 80% training, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,           # 20% for testing
    random_state=42,         # For reproducibility
    shuffle=True,            # Shuffle before splitting
    stratify=y               # Keep class distribution (for multi-class)
)

print(f"Training size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
```

### Visualizing the Split

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Training data
axes[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu', s=10, alpha=0.6)
axes[0].set_title(f'Training Data ({len(X_train)} samples)')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].grid(True, alpha=0.3)

# Test data
axes[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdYlBu', s=10, alpha=0.6)
axes[1].set_title(f'Test Data ({len(X_test)} samples)')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Converting to PyTorch Tensors

Before using the data in PyTorch, convert NumPy arrays to PyTorch tensors.

### Conversion for Binary Classification

```python
import torch

# For binary classification
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1)  # Shape: (n, 1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).unsqueeze(1)    # Shape: (n, 1)

print(f"X_train shape: {X_train.shape}")  # torch.Size([800, 2])
print(f"y_train shape: {y_train.shape}")  # torch.Size([800, 1])
print(f"X_train dtype: {X_train.dtype}")  # torch.float32
print(f"y_train dtype: {y_train.dtype}")  # torch.float32
```

### Conversion for Multi-class Classification

```python
import torch

# For multi-class classification
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)       # Use LongTensor for class indices
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)         # Use LongTensor for class indices

print(f"X_train shape: {X_train.shape}")  # torch.Size([800, 2])
print(f"y_train shape: {y_train.shape}")  # torch.Size([800])
print(f"X_train dtype: {X_train.dtype}")  # torch.float32
print(f"y_train dtype: {y_train.dtype}")  # torch.int64
```

### Tensor Type Reference

| Classification Type | X Tensor | y Tensor | y Shape |
|---------------------|----------|----------|---------|
| **Binary** | FloatTensor | FloatTensor | `[batch_size, 1]` |
| **Multi-class** | FloatTensor | LongTensor | `[batch_size]` |

---

## Complete Data Preparation Example

### Binary Classification Pipeline

```python
import torch
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

# 1. Create dataset
X, y = make_circles(n_samples=1000, noise=0.03, factor=0.5, random_state=42)

# 2. Visualize
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', s=10, alpha=0.6)
plt.title('Binary Classification Dataset (make_circles)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True, alpha=0.3)
plt.show()

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).unsqueeze(1)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Input features: {X_train.shape[1]}")
print(f"Output shape: {y_train.shape[1]}")
```

### Multi-class Classification Pipeline

```python
import torch
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

# 1. Create dataset
n_classes = 4
X, y = make_blobs(
    n_samples=1000,
    n_features=2,
    centers=n_classes,
    cluster_std=1.5,
    random_state=42
)

# 2. Visualize
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', s=10, alpha=0.6)
plt.title(f'Multi-class Classification Dataset (make_blobs, {n_classes} classes)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True, alpha=0.3)
plt.show()

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Input features: {X_train.shape[1]}")
print(f"Number of classes: {len(torch.unique(y_train))}")
```

---

## Creating DataLoaders

For efficient training, use PyTorch's DataLoader:

```python
from torch.utils.data import TensorDataset, DataLoader

# Create datasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Check dataloader
for X_batch, y_batch in train_loader:
    print(f"Batch shape: X={X_batch.shape}, y={y_batch.shape}")
    break
```

---

## Key Takeaways

| Concept | Description |
|---------|-------------|
| **make_circles** | Binary classification, non-linear (concentric circles) |
| **make_blobs** | Multi-class classification, Gaussian clusters |
| **Visualize first** | Always plot your data before training |
| **Train/test split** | Use 80/20 split, stratify for multi-class |
| **Binary tensors** | X: FloatTensor, y: FloatTensor with shape `[n, 1]` |
| **Multi-class tensors** | X: FloatTensor, y: LongTensor with shape `[n]` |
| **DataLoader** | Use for efficient batch processing during training |

---

## Discussion Questions

1. **Why do we use `stratify=y` for multi-class but not binary?** What happens if we don't stratify?

2. **What happens if you set `noise=0` in make_circles?** Would a linear model work then?

3. **Why do binary targets need `unsqueeze(1)` but multi-class don't?** Think about the expected shapes.

---

## Practice Exercises

1. **Experiment with make_circles:**
   - Try different noise values: 0.0, 0.03, 0.1, 0.3
   - How does noise affect the difficulty?

2. **Experiment with make_blobs:**
   - Try different `cluster_std` values: 0.5, 1.0, 2.0, 5.0
   - How does cluster overlap affect classification difficulty?

3. **Create your own dataset:**
   - Use make_circles with 2000 samples
   - Split 70/15/15 (train/val/test)
   - Visualize all three splits

---

## Next Steps

- [Building Models](04-building-models.md) - Implementing neural network models
- [Training & Evaluation](05-training-evaluation.md) - Training your classifiers
- [Practice Exercise](../../module-03/neural-network-classification/01_binary_classification_intro.py) - Hands-on with make_circles

---

**Last Updated:** January 2026
