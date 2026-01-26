# Building Models

## Learning Objectives

By the end of this lesson, you will be able to:
- Build classification models by subclassing `nn.Module`
- Use `nn.Sequential` for rapid prototyping
- Implement binary and multi-class classifiers
- Inspect model parameters and architecture
- Test your models with dummy inputs

---

## Two Ways to Build Models

PyTorch provides two main approaches for building neural networks:

| Approach | When to Use | Pros | Cons |
|----------|-------------|------|------|
| **`nn.Module` subclass** | Complex models, custom logic | Full control, readable | More boilerplate |
| **`nn.Sequential`** | Simple models, quick prototypes | Concise, fast to write | Less flexible |

Both approaches produce the same type of model. Choose based on your needs.

---

## Method 1: Subclassing nn.Module

This is the recommended approach for most classification tasks.

### Basic Structure

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyModel, self).__init__()
        # Define layers here
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Define forward pass
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.layer2(x)
        return x
```

### Binary Classifier: CircleModelV0 (Linear)

This model uses only linear layers (no non-linearity). It will struggle with the circles dataset.

```python
import torch.nn as nn

class CircleModelV0(nn.Module):
    """Linear model for binary classification"""

    def __init__(self, input_size=2, hidden_size=8, output_size=1):
        super(CircleModelV0, self).__init__()
        # Define layers
        self.layer_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            # Note: No activation here (linear model)
            nn.Linear(hidden_size, output_size)
            # Note: No activation here (output is logits)
        )

    def forward(self, x):
        return self.layer_stack(x)

# Create the model
model_v0 = CircleModelV0(input_size=2, hidden_size=8, output_size=1)
print(model_v0)
```

**Output:**
```
CircleModelV0(
  (layer_stack): Sequential(
    (0): Linear(in_features=2, out_features=8, bias=True)
    (1): Linear(in_features=8, out_features=1, bias=True)
  )
)
```

### Multi-class Classifier

```python
import torch.nn as nn

class BlobModel(nn.Module):
    """Multi-class classifier for make_blobs dataset"""

    def __init__(self, input_size=2, hidden_size=16, num_classes=4):
        super(BlobModel, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
            # Note: No softmax here (output is logits)
        )

    def forward(self, x):
        return self.layer_stack(x)

# Create the model
model = BlobModel(input_size=2, hidden_size=16, num_classes=4)
print(model)
```

**Output:**
```
BlobModel(
  (layer_stack): Sequential(
    (0): Linear(in_features=2, out_features=16, bias=True)
    (1): ReLU()
    (2): Linear(in_features=16, out_features=4, bias=True)
  )
)
```

---

## Method 2: Using nn.Sequential

For simple models, `nn.Sequential` provides a more concise syntax.

### Binary Classifier

```python
import torch.nn as nn

# Create model using nn.Sequential
model = nn.Sequential(
    nn.Linear(in_features=2, out_features=8),
    nn.ReLU(),
    nn.Linear(in_features=8, out_features=1)
)

print(model)
```

### Multi-class Classifier

```python
import torch.nn as nn

# Create model using nn.Sequential
model = nn.Sequential(
    nn.Linear(in_features=2, out_features=16),
    nn.ReLU(),
    nn.Linear(in_features=16, out_features=4)
)

print(model)
```

### When to Use nn.Sequential

Use `nn.Sequential` when:
- Your model is a simple sequence of layers
- You don't need custom forward pass logic
- You want to quickly prototype

Use `nn.Module` subclass when:
- You need complex forward pass logic
- You want to share layers
- You need multiple inputs/outputs

---

## Inspecting Model Architecture

Always inspect your model before training.

### Model Summary

```python
print("=" * 60)
print("Model Architecture")
print("=" * 60)
print(model)

print("\n" + "=" * 60)
print("Model Details")
print("=" * 60)
```

### Counting Parameters

```python
def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

total, trainable = count_parameters(model)
print(f"Total parameters: {total:,}")
print(f"Trainable parameters: {trainable:,}")
```

### Layer-wise Information

```python
print("\nLayer-wise Information:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape} = {param.numel()} parameters")
```

### Output Shapes

```python
print("\nOutput shapes after each layer:")
x = torch.randn(1, 2)  # Dummy input
for i, layer in enumerate(model.layer_stack):
    x = layer(x)
    print(f"  Layer {i} ({layer.__class__.__name__}): {x.shape}")
```

---

## Testing with Dummy Input

Before training, verify your model works with dummy data.

```python
import torch

# Create dummy input
dummy_input = torch.randn(1, 2)  # Batch of 1, 2 features

# Forward pass
with torch.no_grad():
    output = model(dummy_input)

print(f"Input shape: {dummy_input.shape}")  # torch.Size([1, 2])
print(f"Output shape: {output.shape}")      # Depends on model
print(f"Output (logits): {output}")
```

### Binary Classifier Output

```python
# For binary classification
output = model(dummy_input)
print(f"Output shape: {output.shape}")  # torch.Size([1, 1])
print(f"Raw logits: {output.item():.4f}")

# Convert to probability
prob = torch.sigmoid(output)
print(f"Probability: {prob.item():.4f}")

# Convert to label
label = (prob > 0.5).long()
print(f"Predicted label: {label.item()}")
```

### Multi-class Classifier Output

```python
# For multi-class classification
output = model(dummy_input)
print(f"Output shape: {output.shape}")  # torch.Size([1, 4])

# Convert to probabilities
probs = torch.softmax(output, dim=1)
print(f"Probabilities: {probs[0]}")

# Convert to label
label = torch.argmax(probs, dim=1)
print(f"Predicted label: {label.item()}")
```

---

## Complete Model Building Examples

### Binary Classifier with make_circles

```python
import torch
import torch.nn as nn

class CircleModelV0(nn.Module):
    """Linear model for binary classification (circles dataset)"""

    def __init__(self, input_size=2, hidden_size=8, output_size=1):
        super(CircleModelV0, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layer_stack(x)

# Create model
model = CircleModelV0(input_size=2, hidden_size=8, output_size=1)

# Inspect model
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# Test with dummy input
dummy_input = torch.randn(1, 2)
with torch.no_grad():
    output = model(dummy_input)
    prob = torch.sigmoid(output)
    label = (prob > 0.5).long()

print(f"\nInput shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")
print(f"Logits: {output.item():.4f}")
print(f"Probability: {prob.item():.4f}")
print(f"Predicted label: {label.item()}")
```

### Multi-class Classifier with make_blobs

```python
import torch
import torch.nn as nn

class BlobModel(nn.Module):
    """Multi-class classifier for blobs dataset"""

    def __init__(self, input_size=2, hidden_size=16, num_classes=4):
        super(BlobModel, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.layer_stack(x)

# Create model
num_classes = 4
model = BlobModel(input_size=2, hidden_size=16, num_classes=num_classes)

# Inspect model
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# Test with dummy input
dummy_input = torch.randn(1, 2)
with torch.no_grad():
    output = model(dummy_input)
    probs = torch.softmax(output, dim=1)
    label = torch.argmax(probs, dim=1)

print(f"\nInput shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")
print(f"Logits: {output[0]}")
print(f"Probabilities: {probs[0]}")
print(f"Predicted label: {label.item()}")
```

---

## Model Architecture Patterns

### Pattern 1: Minimal (No Hidden Layers)

```python
# Direct input to output
model = nn.Sequential(
    nn.Linear(2, 1)
)
# Only 3 parameters (2 weights + 1 bias)
```

### Pattern 2: Single Hidden Layer

```python
# One hidden layer with activation
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)
# Parameters: (2×8 + 8) + (8×1 + 1) = 33 parameters
```

### Pattern 3: Multiple Hidden Layers

```python
# Deep network with multiple hidden layers
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)
# Parameters: (2×16 + 16) + (16×8 + 8) + (8×1 + 1) = 265 parameters
```

### Pattern 4: With Batch Normalization and Dropout

```python
# Regularized network
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.BatchNorm1d(16),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(16, 8),
    nn.BatchNorm1d(8),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(8, 1)
)
```

---

## Common Mistakes

### Mistake 1: Using Sigmoid in the Model

```python
# DON'T DO THIS
class BadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # Don't put activation here!
        )

# DO THIS INSTEAD
class GoodModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
            # No activation - output is logits
        )
```

**Why?** PyTorch's `BCEWithLogitsLoss` expects raw logits (no sigmoid) for numerical stability.

### Mistake 2: Wrong Output Shape

```python
# DON'T DO THIS
class BadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 4)  # Wrong! Binary needs 1 output
        )

# DO THIS INSTEAD
class GoodModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1)  # Correct! Binary needs 1 output
        )
```

### Mistake 3: Wrong Target Type for Multi-class

```python
# DON'T DO THIS
# Targets are float (wrong for CrossEntropyLoss)
targets = torch.FloatTensor([0, 1, 2, 3])

# DO THIS INSTEAD
# Targets are long integers (correct for CrossEntropyLoss)
targets = torch.LongTensor([0, 1, 2, 3])
```

---

## Key Takeaways

| Concept | Description |
|---------|-------------|
| **nn.Module** | Recommended for most models (full control) |
| **nn.Sequential** | Good for simple models (concise) |
| **Binary output** | Shape: `[batch_size, 1]` |
| **Multi-class output** | Shape: `[batch_size, num_classes]` |
| **No output activation** | Let loss function handle it (BCEWithLogitsLoss/CrossEntropyLoss) |
| **Test with dummy input** | Always verify model before training |
| **Count parameters** | More parameters = more capacity (but risk overfitting) |

---

## Discussion Questions

1. **Why do we put activation functions in hidden layers but not the output layer?** What would happen if we did?

2. **What's the trade-off between more parameters (wider layers) and more layers (deeper network)?**

3. **Why does `nn.Sequential` exist when `nn.Module` is more flexible?** When would you prefer one over the other?

---

## Practice Exercises

1. **Build a binary classifier:**
   - Input: 2 features
   - Hidden: 16 neurons with ReLU
   - Output: 1
   - Count the parameters

2. **Build a multi-class classifier:**
   - Input: 2 features
   - Hidden: 2 layers (32, 16 neurons) with ReLU
   - Output: 4 classes
   - Test with dummy input

3. **Compare architectures:**
   - Build a minimal model (no hidden layers)
   - Build a deep model (3 hidden layers)
   - Count parameters in each

---

## Next Steps

- [Training & Evaluation](05-training-evaluation.md) - Training your models
- [Improving Models](06-improving-models.md) - Adding non-linearity
- [Practice Exercise](../../module-03/neural-network-classification/01_binary_classification_intro.py) - Build CircleModelV0

---

**Last Updated:** January 2026
