# Improving Models

## Learning Objectives

By the end of this lesson, you will be able to:
- Understand why linear models fail on non-linear data
- Add non-linearity with ReLU activation
- Build and train non-linear models
- Compare linear vs non-linear decision boundaries
- Tune hyperparameters to improve performance

---

## Why Linear Models Fail

The make_circles dataset is a **non-linear** problem. Let's see why a linear model struggles.

### Visualizing the Problem

```
    Outer Circle (Class 0)    Inner Circle (Class 1)
            ┌──────┐                    ┌──┐
        ┌───┘      └───┐                │  │
        │     ●        │              ┌─┘  └─┐
        │   ○   ○      │              │  ●  │
        │     ○        │              └─┐  ┌─┘
        │              │                └──┘
        └───┐      ┌───┘
            └──────┘
```

A **linear decision boundary** is a straight line. No matter where you draw it, you can't perfectly separate the two circles.

### Linear Model Limitations

```python
# Linear model: y = w1*x1 + w2*x2 + b
# This can only learn: y > 0 (one side of a line)

# But the true pattern is: x1² + x2² > r² (circle)
# This requires non-linear combinations!
```

---

## Adding Non-Linearity

The solution: add **non-linear activation functions** to create non-linear decision boundaries.

### What is Non-Linearity?

A non-linear function cannot be written as a straight line. Common non-linear activations:

| Activation | Formula | Output Range | Key Property |
|------------|---------|--------------|--------------|
| **ReLU** | max(0, x) | [0, ∞) | Allows model to learn non-linear patterns |
| **Sigmoid** | 1/(1 + e⁻ˣ) | [0, 1] | Squashes output to probability |
| **Tanh** | (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ) | [-1, 1] | Zero-centered output |

**ReLU is the default choice** for hidden layers because:
- Simple and fast to compute
- Doesn't suffer from vanishing gradients
- Works well in practice

---

## CircleModelV1: Non-Linear Model

Let's upgrade CircleModelV0 by adding ReLU activation.

### Linear vs Non-Linear Comparison

```python
import torch.nn as nn

# CircleModelV0: Linear (no activation)
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 8),
            # No activation (linear)
            nn.Linear(8, 1)
        )

# CircleModelV1: Non-linear (with ReLU)
class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),  # Add non-linearity!
            nn.Linear(8, 1)
        )
```

### More Complex Non-Linear Model

```python
class CircleModelV2(nn.Module):
    """Deeper non-linear model"""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
```

---

## Training Comparison

Let's train both models and compare their performance.

### Experiment Setup

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

# Set random seed
torch.manual_seed(42)

# Prepare data
X, y = make_circles(n_samples=1000, noise=0.03, factor=0.5, random_state=42)
X = torch.FloatTensor(X)
y = torch.FloatTensor(y).unsqueeze(1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Training function
def train_model(model, model_name, epochs=100):
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        y_logits = model(X_train)
        loss = criterion(y_logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Testing
        model.eval()
        with torch.inference_mode():
            test_logits = model(X_test)
            test_loss = criterion(test_logits, y_test)

        train_losses.append(loss.item())
        test_losses.append(test_loss.item())

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'  Train Loss: {loss.item():.4f}')
            print(f'  Test Loss: {test_loss.item():.4f}')

    # Calculate final accuracy
    model.eval()
    with torch.inference_mode():
        test_probs = torch.sigmoid(test_logits)
        test_preds = (test_probs > 0.5).long()
        accuracy = (test_preds == y_test).float().mean()

    print(f'\nFinal Test Accuracy: {accuracy.item()*100:.2f}%')

    return train_losses, test_losses
```

### Train Both Models

```python
# Train linear model
model_v0 = CircleModelV0()
losses_v0_train, losses_v0_test = train_model(model_v0, "CircleModelV0 (Linear)")

# Train non-linear model
model_v1 = CircleModelV1()
losses_v1_train, losses_v1_test = train_model(model_v1, "CircleModelV1 (Non-linear)")
```

**Expected Output:**
```
============================================================
Training CircleModelV0 (Linear)
============================================================
Epoch [20/100]
  Train Loss: 0.6928
  Test Loss: 0.6931
...
Final Test Accuracy: 50.00%  # No better than random!

============================================================
Training CircleModelV1 (Non-linear)
============================================================
Epoch [20/100]
  Train Loss: 0.6543
  Test Loss: 0.6589
...
Final Test Accuracy: 99.50%  # Much better!
```

---

## Decision Boundary Comparison

Visualizing the difference between linear and non-linear models.

### Plotting Function

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(model, X, y, title):
    """Plot decision boundary"""
    model.eval()

    # Create meshgrid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )

    # Make predictions
    mesh = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    with torch.inference_mode():
        Z = torch.sigmoid(model(mesh)).reshape(xx.shape)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z.numpy(), alpha=0.3, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y.numpy().squeeze(),
                cmap='RdYlBu', s=40, edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.colorbar(label='Class')
```

### Compare Models

```python
# Plot linear model
plot_decision_boundary(model_v0, X_test, y_test,
                      'CircleModelV0 (Linear) - Accuracy: ~50%')

# Plot non-linear model
plot_decision_boundary(model_v1, X_test, y_test,
                      'CircleModelV1 (Non-linear) - Accuracy: ~99%')
```

**Visualization:**
- **Linear model**: Straight line decision boundary (can't separate circles)
- **Non-linear model**: Circular decision boundary (correctly separates circles)

---

## Hyperparameter Tuning

Beyond adding non-linearity, you can tune hyperparameters to improve performance.

### Key Hyperparameters

| Hyperparameter | Description | Common Values | Effect |
|----------------|-------------|---------------|--------|
| **Learning Rate** | Step size for updates | 0.001, 0.01, 0.1 | Too high: unstable; Too low: slow |
| **Hidden Units** | Neurons per layer | 8, 16, 32, 64, 128 | More = more capacity |
| **Hidden Layers** | Depth of network | 1, 2, 3, 4 | More = can learn more complex patterns |
| **Epochs** | Training iterations | 50, 100, 500, 1000 | More = better (but risk overfitting) |
| **Batch Size** | Samples per update | 16, 32, 64, 128 | Larger = more stable gradients |

### Experimenting with Hyperparameters

```python
# Try different configurations
configs = [
    {'hidden': 8, 'lr': 0.01, 'epochs': 100},
    {'hidden': 16, 'lr': 0.01, 'epochs': 100},
    {'hidden': 32, 'lr': 0.001, 'epochs': 200},
    {'hidden': 64, 'lr': 0.001, 'epochs': 200},
]

results = []

for config in configs:
    # Create model
    model = nn.Sequential(
        nn.Linear(2, config['hidden']),
        nn.ReLU(),
        nn.Linear(config['hidden'], 1)
    )

    # Train
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    for epoch in range(config['epochs']):
        model.train()
        y_logits = model(X_train)
        loss = criterion(y_logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test)
        test_preds = (torch.sigmoid(test_logits) > 0.5).long()
        accuracy = (test_preds == y_test).float().mean()

    results.append({**config, 'accuracy': accuracy.item()})
    print(f"Hidden: {config['hidden']}, LR: {config['lr']}, "
          f"Epochs: {config['epochs']}, Accuracy: {accuracy.item()*100:.2f}%")
```

---

## When to Add More Complexity

### Signs of Underfitting

Your model is too simple if:
- Training loss is high
- Training accuracy is low
- Decision boundary is too simple

**Solutions:**
- Add more hidden layers
- Increase hidden units
- Add non-linearity (ReLU)
- Train for more epochs

### Signs of Overfitting

Your model is too complex if:
- Training loss is low
- Test loss is high
- Training accuracy >> Test accuracy

**Solutions:**
- Add dropout (0.2-0.5)
- Reduce hidden units
- Add weight decay
- Use more training data

---

## Architecture Evolution

See how models can evolve from simple to complex:

```
Stage 1: Linear (Underfits)
├─ 2 → 8 → 1
└─ Accuracy: ~50%

Stage 2: Add ReLU (Better)
├─ 2 → 8 (ReLU) → 1
└─ Accuracy: ~80%

Stage 3: Deeper (Good)
├─ 2 → 16 (ReLU) → 8 (ReLU) → 1
└─ Accuracy: ~95%

Stage 4: Wider (Best)
├─ 2 → 64 (ReLU) → 32 (ReLU) → 1
└─ Accuracy: ~99%
```

---

## Key Takeaways

| Concept | Description |
|---------|-------------|
| **Linear models** | Can only learn straight-line decision boundaries |
| **Non-linear models** | Can learn complex patterns with activation functions |
| **ReLU** | Default activation for hidden layers (adds non-linearity) |
| **make_circles** | Requires non-linear model to solve |
| **Decision boundary** | Visual representation of what model learned |
| **Underfitting** | Model too simple (add layers/units) |
| **Overfitting** | Model too complex (add dropout/reduce capacity) |
| **Hyperparameter tuning** | Systematically search for best configuration |

---

## Discussion Questions

1. **Why does ReLU enable non-linear decision boundaries?** What would happen without it?

2. **When would you prefer a deeper network vs. a wider network?** Think about parameter count and learning capacity.

3. **Why does the linear model achieve ~50% accuracy on make_circles?** Is this better or worse than random?

---

## Practice Exercises

1. **Compare models:**
   - Train linear vs non-linear models on make_circles
   - Plot decision boundaries for both
   - Compare final accuracy

2. **Tune hyperparameters:**
   - Try different hidden layer sizes: 4, 8, 16, 32, 64
   - Try different learning rates: 0.001, 0.01, 0.1
   - Track accuracy for each configuration

3. **Build deeper models:**
   - Create models with 1, 2, 3, 4 hidden layers
   - Compare their decision boundaries
   - Which performs best?

---

## Next Steps

- [Evaluation Metrics](07-evaluation-metrics.md) - Measuring model performance
- [Practice Exercise](../../module-03/neural-network-classification/03_non_linear_classification.py) - Compare linear vs non-linear

---

**Last Updated:** January 2026
