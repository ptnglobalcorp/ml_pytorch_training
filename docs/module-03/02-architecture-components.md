# Architecture Components

## Learning Objectives

By the end of this lesson, you will be able to:
- Design the input layer to match your data
- Choose appropriate hidden layer configurations
- Design the output layer for your classification type
- Select the right activation functions for each layer
- Choose the correct loss function for your task

---

## Neural Network Architecture Overview

A classification neural network consists of three types of layers:

```
Input Layer → Hidden Layers → Output Layer
    ↓              ↓              ↓
 Match data    Learn patterns   Produce predictions
```

### Architecture Decision Flow

```
1. Input Layer Shape = Number of features in data
2. Hidden Layers = Your design choice (depth & width)
3. Output Layer Shape = 1 (binary) or num_classes (multi-class)
4. Hidden Activations = ReLU (most common)
5. Output Activation = None (logits) or Sigmoid/Softmax
6. Loss Function = BCEWithLogitsLoss or CrossEntropyLoss
```

---

## Input Layer

The input layer shape must match the number of features in your data.

### Determining Input Shape

| Data Type | Example | Input Features | Input Shape |
|-----------|---------|----------------|-------------|
| **Tabular** | Patient records | Age, weight, blood pressure, etc. | num_features |
| **Flattened Image** | 28×28 MNIST | 784 pixels | 784 |
| **Text (bag of words)** | Document vocabulary | 5000 words | 5000 |

### Example: Input Layer Setup

```python
import torch.nn as nn

# If your data has 2 features (e.g., x, y coordinates)
input_size = 2

# For flattened 28x28 images
input_size = 28 * 28  # 784

# For tabular data with 10 features
input_size = 10

# Create input layer
nn.Linear(input_size, hidden_size)
```

### Inspecting Your Data

Always verify your input shape by inspecting your data:

```python
import torch

# Check your data shape
X = torch.randn(100, 2)  # 100 samples, 2 features
print(f"Data shape: {X.shape}")  # torch.Size([100, 2])
print(f"Number of features: {X.shape[1]}")  # 2

# Set input size
input_size = X.shape[1]  # 2
```

---

## Hidden Layers

Hidden layers learn patterns in your data. You design both the **depth** (number of layers) and **width** (number of neurons per layer).

### Hidden Layer Guidelines

| Property | Guideline | Common Values |
|----------|-----------|---------------|
| **Depth** | More layers = more complex patterns | 1-5 layers for simple problems |
| **Width** | More neurons = more capacity | 10-512 neurons per layer |
| **Pattern** | Often decreasing widths | 128 → 64 → 32 |

### Activation Functions for Hidden Layers

| Activation | Formula | Range | When to Use |
|------------|---------|-------|-------------|
| **ReLU** | max(0, x) | [0, ∞) | Default choice |
| **LeakyReLU** | max(0.01x, x) | (-∞, ∞) | When you have dead neurons |
| **Tanh** | (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ) | [-1, 1] | When you need negative outputs |
| **Sigmoid** | 1/(1 + e⁻ˣ) | [0, 1] | Rarely used in hidden layers |

**Why ReLU is the default:**
- Computationally efficient
- Doesn't suffer from vanishing gradients
- Sparse activation (many neurons output 0)

### Hidden Layer Patterns

**Pattern 1: Single Hidden Layer**
```python
nn.Sequential(
    nn.Linear(input_size, 64),
    nn.ReLU(),
    nn.Linear(64, output_size)
)
```

**Pattern 2: Multiple Hidden Layers (Decreasing Width)**
```python
nn.Sequential(
    nn.Linear(input_size, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, output_size)
)
```

**Pattern 3: With Batch Normalization and Dropout**
```python
nn.Sequential(
    nn.Linear(input_size, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, output_size)
)
```

### Choosing Hidden Layer Configuration

**Rule of thumb:**
- Start simple: 1-2 hidden layers
- If underfitting: Add more layers/neurons
- If overfitting: Add dropout, reduce neurons

**Experiment with:**
```python
# Try different configurations
configs = [
    [64],           # Single layer, 64 neurons
    [128, 64],      # Two layers, decreasing
    [256, 128, 64], # Three layers, decreasing
    [32, 32, 32],   # Three layers, constant width
]
```

---

## Output Layer

The output layer design depends on your classification type.

### Binary Classification Output

**Shape:** 1 (single value)

**Model outputs:** Raw logits (no activation in forward pass)

```python
class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU()
        )
        self.output = nn.Linear(64, 1)  # Single output

    def forward(self, x):
        x = self.hidden(x)
        return self.output(x)  # Raw logits
```

**Converting to predictions:**
```python
logits = model(X)           # Raw output
probs = torch.sigmoid(logits)  # [0, 1]
labels = (probs > 0.5).long()  # 0 or 1
```

### Multi-class Classification Output

**Shape:** num_classes (one value per class)

**Model outputs:** Raw logits (no activation in forward pass)

```python
class MultiClassClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU()
        )
        self.output = nn.Linear(64, num_classes)  # One per class

    def forward(self, x):
        x = self.hidden(x)
        return self.output(x)  # Raw logits
```

**Converting to predictions:**
```python
logits = model(X)                # Raw output
probs = torch.softmax(logits, dim=1)  # Sum to 1
labels = torch.argmax(probs, dim=1)    # Class index
```

### Output Layer Comparison

| Classification Type | Output Shape | Activation in Model | Activation for Prediction |
|---------------------|--------------|---------------------|---------------------------|
| **Binary** | 1 | None (logits) | Sigmoid |
| **Multi-class** | num_classes | None (logits) | Softmax |
| **Multi-label** | num_labels | None (logits) | Sigmoid |

---

## Activation Functions

Activation functions introduce non-linearity, allowing neural networks to learn complex patterns.

### Hidden Layer Activations

| Activation | PyTorch | Best For |
|------------|---------|----------|
| **ReLU** | `nn.ReLU()` | Most cases (default) |
| **LeakyReLU** | `nn.LeakyReLU(0.01)` | When ReLU causes dead neurons |
| **Tanh** | `nn.Tanh()` | When you need negative outputs |

**ReLU is the default choice:**
```python
# Recommended
nn.ReLU()

# Equivalent to:
torch.nn.functional.relu(x)
```

### Output Layer Activations

**Important:** Don't use activation functions in the model's forward pass. Apply them when converting logits to predictions.

| Classification | PyTorch Activation | Applied When |
|----------------|-------------------|--------------|
| **Binary** | `torch.sigmoid()` | During inference |
| **Multi-class** | `torch.softmax(dim=1)` | During inference |
| **Multi-label** | `torch.sigmoid()` | During inference |

**Why not use activation in the model?**

PyTorch's loss functions (`BCEWithLogitsLoss`, `CrossEntropyLoss`) expect raw logits. They apply the appropriate activation internally for numerical stability.

---

## Loss Functions

The loss function measures how well your model is performing.

### Binary Classification: BCEWithLogitsLoss

```python
import torch.nn as nn

criterion = nn.BCEWithLogitsLoss()

# Model outputs raw logits (shape: batch_size, 1)
outputs = model(X)

# Targets are float tensor of 0s and 1s (shape: batch_size, 1)
targets = y.float()

loss = criterion(outputs, targets)
```

**Requirements:**
- Model outputs: Raw logits (shape: `[batch_size, 1]`)
- Targets: Float tensor with values 0 or 1 (shape: `[batch_size, 1]`)

### Multi-class Classification: CrossEntropyLoss

```python
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

# Model outputs raw logits (shape: batch_size, num_classes)
outputs = model(X)

# Targets are long tensor of class indices (shape: batch_size,)
targets = y.long()

loss = criterion(outputs, targets)
```

**Requirements:**
- Model outputs: Raw logits (shape: `[batch_size, num_classes]`)
- Targets: Long tensor with class indices (shape: `[batch_size]`)

**Important:** Targets are class indices (0, 1, 2, ...), NOT one-hot encoded.

### Loss Function Quick Reference

| Classification Type | Loss Function | Output Shape | Target Type | Target Values |
|---------------------|---------------|--------------|-------------|---------------|
| **Binary** | `BCEWithLogitsLoss()` | `[batch, 1]` | Float | 0.0 or 1.0 |
| **Multi-class** | `CrossEntropyLoss()` | `[batch, num_classes]` | Long | 0, 1, 2, ... |
| **Multi-label** | `BCEWithLogitsLoss()` | `[batch, num_labels]` | Float | 0.0 or 1.0 |

---

## Complete Architecture Examples

### Binary Classifier

```python
import torch.nn as nn

class BinaryClassifier(nn.Module):
    def __init__(self, input_size=2, hidden_size=64):
        super().__init__()
        self.network = nn.Sequential(
            # Input layer
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),

            # Hidden layers
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),

            # Output layer (1 for binary)
            nn.Linear(hidden_size // 2, 1)
            # No activation here (output is logits)
        )

    def forward(self, x):
        return self.network(x)

# Usage
model = BinaryClassifier(input_size=2, hidden_size=64)
criterion = nn.BCEWithLogitsLoss()
```

### Multi-class Classifier

```python
import torch.nn as nn

class MultiClassClassifier(nn.Module):
    def __init__(self, input_size=2, num_classes=4, hidden_size=64):
        super().__init__()
        self.network = nn.Sequential(
            # Input layer
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),

            # Hidden layers
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),

            # Output layer (num_classes for multi-class)
            nn.Linear(hidden_size // 2, num_classes)
            # No activation here (output is logits)
        )

    def forward(self, x):
        return self.network(x)

# Usage
model = MultiClassClassifier(input_size=2, num_classes=4, hidden_size=64)
criterion = nn.CrossEntropyLoss()
```

---

## Architecture Decision Tree

Use this flowchart to design your classifier:

```
1. What's your input shape?
   └─ Count the number of features in your data
   └─ Set: input_size = number_of_features

2. What type of classification?
   ├─ Binary (2 classes)
   │  ├─ Output shape: 1
   │  ├─ Loss: BCEWithLogitsLoss
   │  └─ Prediction: torch.sigmoid then threshold
   │
   └─ Multi-class (3+ classes)
      ├─ Output shape: num_classes
      ├─ Loss: CrossEntropyLoss
      └─ Prediction: torch.softmax then argmax

3. Hidden layers?
   ├─ Start simple: 1-2 layers
   ├─ Width: 32-128 neurons per layer
   ├─ Activation: ReLU
   └─ Add dropout if overfitting

4. Regularization?
   ├─ Overfitting → Add dropout (0.2-0.5)
   ├─ Slow training → Add batch normalization
   └─ High variance → Reduce model capacity
```

---

## Key Takeaways

| Component | Design Rule |
|-----------|-------------|
| **Input layer** | Shape = number of features in data |
| **Hidden layers** | Start simple: 1-2 layers, 32-128 neurons |
| **Hidden activation** | Use ReLU (default choice) |
| **Output layer (binary)** | Shape = 1 |
| **Output layer (multi-class)** | Shape = num_classes |
| **Output activation** | None (use logits) |
| **Binary loss** | BCEWithLogitsLoss |
| **Multi-class loss** | CrossEntropyLoss |
| **Targets (binary)** | Float tensor with 0.0 or 1.0 |
| **Targets (multi-class)** | Long tensor with class indices |

---

## Discussion Questions

1. **Why don't we use softmax in the model for multi-class classification?** Hint: Think about what CrossEntropyLoss does internally.

2. **When would you use more hidden layers vs. wider layers?** What's the trade-off?

3. **What happens if your input shape doesn't match your data?** What error would you see?

---

## Practice Exercises

1. **Design a binary classifier:**
   - Input: 10 features
   - Hidden: 2 layers (64, 32 neurons)
   - Output: 1
   - Loss: ?

2. **Design a multi-class classifier:**
   - Input: 784 features (flattened 28x28 image)
   - Hidden: 3 layers (256, 128, 64 neurons)
   - Output: 10 (for digits 0-9)
   - Loss: ?

3. **Compare architectures:**
   - Try deep (many layers) vs. wide (many neurons)
   - Which trains faster? Which achieves better accuracy?

---

## Next Steps

- [Data Preparation](03-data-preparation.md) - Creating classification datasets
- [Building Models](04-building-models.md) - Implementing nn.Module models
- [Practice Exercise](../../module-03/neural-network-classification/01_binary_classification_intro.py) - Build your first classifier

---

**Last Updated:** January 2026
