# Classification Introduction

## Learning Objectives

By the end of this lesson, you will be able to:
- Understand what classification is and why it's important
- Distinguish between binary, multi-class, and multi-label classification
- Write device-agnostic code that works on CPU and GPU
- Choose the appropriate loss function for your classification task

---

## What is Classification?

Classification is the task of predicting discrete class labels from input features. Unlike regression (which predicts continuous values), classification predicts which category an input belongs to.

### Classification vs Regression

| Aspect | Regression | Classification |
|--------|------------|----------------|
| **Output** | Continuous value (e.g., 3.7, -2.1) | Discrete label (e.g., "spam", "not spam") |
| **Example** | Predicting house price | Classifying email as spam/not spam |
| **Loss** | MSELoss | BCEWithLogitsLoss / CrossEntropyLoss |
| **Output Layer** | 1 neuron (linear) | 1 or more neurons (with activation) |

---

## Types of Classification

There are three main types of classification problems:

### 1. Binary Classification

Predicting between **two mutually exclusive classes**.

| Property | Description |
|----------|-------------|
| **Number of classes** | 2 |
| **Examples** | Spam detection, disease diagnosis, fraud detection |
| **Labels** | 0 and 1, or "negative" and "positive" |
| **Loss Function** | `BCEWithLogitsLoss` |
| **Output Shape** | 1 (single value) |
| **Output Activation** | Sigmoid (converts to probability [0,1]) |

**Example:** Is this email spam or not spam?

```python
# Model output: single logit
output = model(X)  # Shape: (batch_size, 1)
# Convert to probability
prob = torch.sigmoid(output)  # Shape: (batch_size, 1)
# Convert to label
label = (prob > 0.5).long()  # Shape: (batch_size, 1)
```

### 2. Multi-class Classification

Predicting **one class from three or more mutually exclusive classes**.

| Property | Description |
|----------|-------------|
| **Number of classes** | 3 or more |
| **Examples** | Digit recognition (0-9), image classification (cat/dog/bird) |
| **Labels** | 0, 1, 2, ..., N-1 (class indices) |
| **Loss Function** | `CrossEntropyLoss` |
| **Output Shape** | num_classes (one value per class) |
| **Output Activation** | Softmax (converts to probabilities that sum to 1) |

**Example:** What digit is this (0-9)?

```python
# Model output: one logit per class
output = model(X)  # Shape: (batch_size, num_classes)
# Convert to probabilities
prob = torch.softmax(output, dim=1)  # Shape: (batch_size, num_classes)
# Convert to label
label = torch.argmax(prob, dim=1)  # Shape: (batch_size,)
```

### 3. Multi-label Classification

Predicting **multiple labels simultaneously**. Each label is independent.

| Property | Description |
|----------|-------------|
| **Number of labels** | N (each independent) |
| **Examples** | Movie genres (action, comedy, drama), article tags |
| **Labels** | Multi-hot encoded [0, 1, 0, 1, ...] |
| **Loss Function** | `BCEWithLogitsLoss` |
| **Output Shape** | num_labels (one value per label) |
| **Output Activation** | Sigmoid (each label independently) |

**Example:** What genres apply to this movie?

```python
# Model output: one logit per label
output = model(X)  # Shape: (batch_size, num_labels)
# Convert to probabilities (each independently)
prob = torch.sigmoid(output)  # Shape: (batch_size, num_labels)
# Convert to labels
labels = (prob > 0.5).long()  # Shape: (batch_size, num_labels)
```

### Classification Type Summary

| Type | Classes | Mutually Exclusive | Loss Function | Output Activation |
|------|---------|-------------------|---------------|-------------------|
| **Binary** | 2 | Yes | BCEWithLogitsLoss | Sigmoid |
| **Multi-class** | 3+ | Yes | CrossEntropyLoss | Softmax |
| **Multi-label** | N | No | BCEWithLogitsLoss | Sigmoid |

---

## Device-Agnostic Code

In PyTorch, it's important to write code that works on both CPU and GPU (CUDA) devices. This is called **device-agnostic code**.

### Setting Up Device-Agnostic Code

```python
import torch

# Check if CUDA (GPU) is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

**Output (with GPU):**
```
Using device: cuda
GPU Name: NVIDIA GeForce RTX 3080
GPU Memory: 10.0 GB
```

**Output (without GPU):**
```
Using device: cpu
```

### Moving Tensors and Models to Device

```python
# Move model to device
model = model.to(device)

# Move tensors to device
X = X.to(device)
y = y.to(device)

# Create tensors directly on device
X_gpu = torch.randn(100, 2).to(device)
```

### Complete Device-Agnostic Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
).to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Data (move to device)
X_train = torch.randn(100, 2).to(device)
y_train = torch.randint(0, 2, (100, 1)).float().to(device)

# Training loop
model.train()
for epoch in range(100):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')
```

### PyTorch 2.0+ Default Device

In PyTorch 2.0+, you can set a default device for all tensors:

```python
# PyTorch 2.0+ only
if torch.cuda.is_available():
    torch.set_default_device('cuda')

# All subsequent tensors are created on the default device
x = torch.randn(2, 3)  # Automatically on CUDA if available
```

---

## Choosing the Right Loss Function

### Binary Classification: BCEWithLogitsLoss

```python
criterion = nn.BCEWithLogitsLoss()

# Model outputs raw logits (no sigmoid in model)
outputs = model(X)  # Shape: (batch_size, 1)
loss = criterion(outputs, targets)

# Targets should be float tensor of 0s and 1s
targets = y.float()  # Shape: (batch_size, 1)
```

**Why BCEWithLogitsLoss instead of BCELoss?**

`BCEWithLogitsLoss` is more numerically stable because it combines the sigmoid and binary cross-entropy in a single operation.

### Multi-class Classification: CrossEntropyLoss

```python
criterion = nn.CrossEntropyLoss()

# Model outputs raw logits (no softmax in model)
outputs = model(X)  # Shape: (batch_size, num_classes)
loss = criterion(outputs, targets)

# Targets should be long tensor of class indices
targets = y.long()  # Shape: (batch_size,)
```

**Important Notes:**
- Use `CrossEntropyLoss` (not `NLLLoss`) - it combines LogSoftmax and NLLLoss
- Targets are class indices (not one-hot encoded)
- Model outputs raw logits (no softmax in forward pass)

### Loss Function Decision Tree

```
Is it binary classification (2 classes)?
├─ Yes → Use BCEWithLogitsLoss
└─ No
    └─ Is it multi-class (3+ mutually exclusive classes)?
        ├─ Yes → Use CrossEntropyLoss
        └─ No (multi-label) → Use BCEWithLogitsLoss
```

---

## Key Takeaways

| Concept | Description |
|---------|-------------|
| **Classification** predicts discrete labels | Unlike regression which predicts continuous values |
| **Binary classification** has 2 classes | Use BCEWithLogitsLoss |
| **Multi-class classification** has 3+ mutually exclusive classes | Use CrossEntropyLoss |
| **Multi-label classification** has N independent labels | Use BCEWithLogitsLoss |
| **Device-agnostic code** works on CPU and GPU | Use `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` |
| **BCEWithLogitsLoss** is more stable than BCELoss | Combines sigmoid and BCE in one operation |
| **CrossEntropyLoss** combines LogSoftmax and NLLLoss | Use class indices as targets (not one-hot) |

---

## Discussion Questions

1. **Why don't we use one-hot encoding with CrossEntropyLoss?** What would happen if we did?

2. **What's the difference between mutually exclusive and independent classes?** Give examples of each.

3. **When would you prefer a CPU over a GPU for training?** Think about data size, model complexity, and available hardware.

---

## Practice Exercises

1. **Identify the classification type:**
   - Predicting if an image contains a cat (yes/no)
   - Predicting if an image contains a cat, dog, or bird
   - Tagging a photo with "beach", "sunset", "family"

2. **Write device-agnostic code:**
   - Create a simple model
   - Set up device-agnostic training
   - Print which device is being used

3. **Choose the right loss function:**
   - Binary classification problem
   - 5-class classification problem
   - Multi-label problem with 10 labels

---

## Next Steps

- [Architecture Components](02-architecture-components.md) - Designing neural network architectures
- [Data Preparation](03-data-preparation.md) - Creating and visualizing classification datasets
- [Practice Exercise](../../module-03/neural-network-classification/01_binary_classification_intro.py) - Binary classification hands-on

---

**Last Updated:** January 2026
