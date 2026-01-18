# Building PyTorch Models

## Learning Objectives

By the end of this lesson, you will be able to:
- Create models by subclassing `nn.Module`
- Understand `nn.Parameter` for learnable weights
- Implement the `forward()` method
- Inspect model parameters and make predictions

---

## Step 2: Building a PyTorch Model

Now that we have our data, we need a model to learn from it. In PyTorch, all neural network models are built by subclassing `nn.Module`—the base class for all neural network modules.

### The Linear Regression Model

We'll build a simple linear regression model:

$$y = w \times X + b$$

Where:
- $w$ (weight) and $b$ (bias) are learnable parameters
- Our goal is for the model to learn $w \approx 0.7$ and $b \approx 0.3$

```python
import torch
import torch.nn as nn

class LinearRegressionModel(nn.Module):
    """
    Simple linear regression model: y = weight * X + bias
    """

    def __init__(self):
        super().__init__()
        # Create learnable parameters
        self.weight = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute predictions from input
        """
        return self.weight * x + self.bias
```

### Understanding nn.Module

`nn.Module` is the foundation of all PyTorch models. It provides:

| Feature | What It Does |
|---------|--------------|
| **Parameter tracking** | Automatically registers `nn.Parameter` objects |
| **GPU support** | Easy movement between CPU and GPU with `.to(device)` |
| **Mode switching** | `train()` and `eval()` for different behaviors |
| **Nested modules** | Can contain other `nn.Module` instances |

**Key methods:**
- `__init__()`: Define layers and parameters
- `forward()`: Define how data flows through the model

### Understanding nn.Parameter

`nn.Parameter` wraps a tensor so PyTorch tracks it for optimization:

```python
# Regular tensor - NOT tracked for gradients
regular_tensor = torch.tensor([1.0, 2.0])
print(f"Regular tensor requires_grad: {regular_tensor.requires_grad}")  # False

# Parameter - TRACKED for gradients
parameter = nn.Parameter(torch.tensor([1.0, 2.0]))
print(f"Parameter requires_grad: {parameter.requires_grad}")  # True
```

**Why use `nn.Parameter`?**
- Automatically tracked by the optimizer
- Included in `model.parameters()` iterator
- Gradient computation is automatic during backward pass

---

## Creating and Inspecting the Model

```python
# Set random seed for reproducibility
torch.manual_seed(42)

# Create model instance
model = LinearRegressionModel()

print("Model created successfully!")
print(f"\nModel architecture:\n{model}")
```

**Output:**
```
Model created successfully!

Model architecture:
LinearRegressionModel(
  (weight): Parameter containing: torch.Tensor([0.3367])
  (bias): Parameter containing: torch.Tensor([0.1288])
)
```

### Inspecting Model Parameters

```python
# Print all parameters
print("\nModel parameters:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.item():.4f} (requires_grad={param.requires_grad})")

# Access individual parameters
print(f"\nWeight details:")
print(f"  Value: {model.weight.item():.4f}")
print(f"  Shape: {model.weight.shape}")
print(f"  Gradient enabled: {model.weight.requires_grad}")

print(f"\nBias details:")
print(f"  Value: {model.bias.item():.4f}")
print(f"  Shape: {model.bias.shape}")
print(f"  Gradient enabled: {model.bias.requires_grad}")
```

**Output:**
```
Model parameters:
  weight: 0.3367 (requires_grad=True)
  bias: 0.1288 (requires_grad=True)

Weight details:
  Value: 0.3367
  Shape: torch.Size([1])
  Gradient enabled: True
Bias details:
  Value: 0.1288
  Shape: torch.Size([1])
  Gradient enabled: True
```

**Notice:** The parameters are randomly initialized, not yet close to the true values (weight=0.7, bias=0.3). Training will adjust them.

### Counting Parameters

```python
# Count total and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nTotal parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

# Show model state dict
print(f"\nModel state_dict keys:")
for key in model.state_dict().keys():
    print(f"  {key}")
```

**Output:**
```
Total parameters: 2
Trainable parameters: 2

Model state_dict keys:
  weight
  bias
```

---

## Making Predictions

The `forward()` method is called automatically when you pass data to the model:

```python
# Create some test data
X_test = torch.tensor([[0.0], [0.5], [1.0]])

# Make predictions (forward pass is automatic)
with torch.no_grad():
    predictions = model(X_test)

print(f"Input values:\n{X_test.flatten()}")
print(f"\nPredictions:\n{predictions.flatten()}")
print(f"\nExpected (if weight=0.7, bias=0.3):\n{0.7 * X_test.flatten() + 0.3}")
```

**Output:**
```
Input values:
tensor([0., 0.5000, 1.0000])

Predictions:
tensor([0.1288, 0.2962, 0.4655])

Expected (if weight=0.7, bias=0.3):
tensor([0.3000, 0.6500, 1.0000])
```

**Observation:** The predictions don't match because the model hasn't been trained yet! The parameters are random.

### Visualizing Initial Predictions

```python
import matplotlib.pyplot as plt

# Create data for visualization
X = torch.arange(0, 1, 0.02).unsqueeze(dim=1)
y_true = 0.7 * X + 0.3  # True relationship

# Get model predictions
with torch.no_grad():
    y_pred = model(X)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X, y_true, c='b', s=50, alpha=0.6, label='True data (y=0.7X+0.3)')
plt.scatter(X, y_pred, c='r', s=50, alpha=0.6, label='Model predictions (untrained)')
plt.plot(X, y_pred, 'r--', alpha=0.3)
plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend(fontsize=10)
plt.title(f'Initial Predictions (Before Training)\nweight={model.weight.item():.3f}, bias={model.bias.item():.3f}',
          fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**What you see:** The model's predictions (red) don't match the true data (blue) because we haven't trained yet. The red line shows the model's current understanding—a poor fit!

---

## The forward() Method

The `forward()` method defines how data flows through your model:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.weight * x + self.bias
```

**Key points:**
- Called automatically when you do `model(x)`
- Must return a tensor (the model's output)
- Can use any PyTorch operations
- Can be as simple or complex as needed

**Example: Adding a non-linearity**
```python
class NonLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        # Apply ReLU activation
        return torch.relu(self.weight * x + self.bias)
```

---

## Model Building Essentials

| Component | Purpose | Example |
|-----------|---------|---------|
| `__init__()` | Define layers and parameters | `self.weight = nn.Parameter(...)` |
| `forward()` | Define computation flow | `return self.weight * x + self.bias` |
| `nn.Parameter` | Create learnable parameters | Wraps tensor for gradient tracking |
| `nn.Module` | Base class for models | Provides training utilities |
| `model.parameters()` | Iterator over parameters | Used by optimizer |

---

## Key Takeaways

| Concept | Description |
|---------|-------------|
| **nn.Module** | Base class for all PyTorch models |
| **nn.Parameter** | Wraps tensors so they're tracked for optimization |
| **forward()** | Defines how data flows through the model |
| **Random initialization** | Parameters start random; training adjusts them |
| **model.train() / model.eval()** | Switch between training and inference modes |
| **state_dict()** | Dictionary of all parameters |

---

## Practice Exercises

1. **Create a model with different initialization**: Modify the model to initialize with `weight=0.0` and `bias=0.0`. What do predictions look like?

2. **Multiple input features**: Extend the model to handle 2 input features:
   ```python
   # y = w1*x1 + w2*x2 + b
   ```

3. **Add a non-linearity**: Create a model that uses `torch.relu()` in the forward pass. How does this change predictions?

4. **Print model summary**: Write a function that prints a summary of the model including:
   - Number of layers
   - Parameters per layer
   - Total parameters

---

## Discussion Questions

1. **Why do we need `requires_grad=True`?** What would happen if parameters didn't track gradients?

2. **What happens if we don't call `super().__init__()`?** Why is this necessary?

3. **How does PyTorch know which parameters to update?** Think about how `nn.Parameter` and `model.parameters()` work together.

4. **Why visualize predictions before training?** What does this tell us about the model's initial state?

---

## Next Steps

Our model is defined but the predictions are wrong. Let's train it!

[Continue to: Training Loop →](04-training-loop.md)

---

**Practice this lesson:** [Exercise 2: Building Models](../../module-02/pytorch-workflow/02_building_models.py)

**Last Updated:** January 2026
