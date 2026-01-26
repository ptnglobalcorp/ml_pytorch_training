# Training & Evaluation

## Learning Objectives

By the end of this lesson, you will be able to:
- Implement the 5-step training loop
- Convert logits → probabilities → labels
- Perform inference with `model.eval()` and `torch.inference_mode()`
- Visualize decision boundaries
- Track training progress

---

## The 5-Step Training Loop

Every PyTorch training loop follows the same 5 steps:

```python
# 1. Forward pass: Compute predictions
y_pred = model(X_train)

# 2. Calculate loss: Compare predictions to targets
loss = loss_fn(y_pred, y_train)

# 3. Zero gradients: Clear previous gradients
optimizer.zero_grad()

# 4. Backward pass: Compute gradients
loss.backward()

# 5. Optimizer step: Update parameters
optimizer.step()
```

### Complete Training Loop Template

```python
import torch
import torch.nn as nn

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Loss function (binary)
criterion = nn.BCEWithLogitsLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training settings
epochs = 100

# Training loop
model.train()  # Set to training mode
for epoch in range(epochs):
    # 1. Forward pass
    y_logits = model(X_train)
    y_pred = torch.sigmoid(y_logits)

    # 2. Calculate loss
    loss = criterion(y_logits, y_train)

    # 3. Zero gradients
    optimizer.zero_grad()

    # 4. Backward pass
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```

---

## Training vs Evaluation Modes

PyTorch models behave differently during training and evaluation.

### Training Mode

```python
model.train()  # Set model to training mode

# Enables:
# - Dropout (randomly drops neurons)
# - BatchNorm (uses batch statistics)

# Use during: Training loop
```

### Evaluation Mode

```python
model.eval()   # Set model to evaluation mode

# Disables:
# - Dropout (uses all neurons)
# - BatchNorm (uses running statistics)

# Use during: Testing, validation, inference
```

### Inference Mode

```python
with torch.inference_mode():
    # Disables gradient calculation
    # Faster, uses less memory
    # Use for: Making predictions

    predictions = model(X_test)
```

**Important:** Always use `torch.inference_mode()` when making predictions. It's faster and uses less memory than `torch.no_grad()`.

---

## Logits → Probabilities → Labels Pipeline

Understanding how to convert raw model outputs to predictions.

### What are Logits?

Logits are the **raw, unnormalized outputs** of a neural network. They can be any real number (negative, zero, positive).

```
Logits → Raw model output (any real number)
           ↓
Activation → Sigmoid or Softmax
           ↓
Probabilities → Values in [0, 1] that sum to 1
           ↓
Decision → Threshold or argmax
           ↓
Labels → Final class prediction
```

### Binary Classification Pipeline

```python
# Model outputs raw logits
logits = model(X)  # Shape: [batch_size, 1], values like [-2.3, 0.5, 3.1]

# Step 1: Convert to probabilities using sigmoid
probs = torch.sigmoid(logits)  # Shape: [batch_size, 1], values in [0, 1]
# Example: [0.09, 0.62, 0.96]

# Step 2: Convert to labels using threshold
labels = (probs > 0.5).long()  # Shape: [batch_size, 1], values 0 or 1
# Example: [[0], [1], [1]]

print(f"Logits: {logits[0].item():.4f}")
print(f"Probability: {probs[0].item():.4f}")
print(f"Label: {labels[0].item()}")
```

### Multi-class Classification Pipeline

```python
# Model outputs raw logits
logits = model(X)  # Shape: [batch_size, num_classes]
# Example: [[-1.2, 2.3, 0.5, -0.8],
#           [3.1, -0.5, 1.2, 0.3]]

# Step 1: Convert to probabilities using softmax
probs = torch.softmax(logits, dim=1)  # Shape: [batch_size, num_classes]
# Example: [[0.04, 0.71, 0.18, 0.07],
#           [0.78, 0.03, 0.14, 0.06]]

# Step 2: Convert to labels using argmax
labels = torch.argmax(probs, dim=1)  # Shape: [batch_size]
# Example: [1, 0]

print(f"Logits: {logits[0]}")
print(f"Probabilities: {probs[0]}")
print(f"Predicted class: {labels[0].item()}")
```

---

## Complete Training Example

### Binary Classification Training

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

# Set random seed
torch.manual_seed(42)

# 1. Prepare data
X, y = make_circles(n_samples=1000, noise=0.03, factor=0.5, random_state=42)
X = torch.FloatTensor(X)
y = torch.FloatTensor(y).unsqueeze(1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Define model
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 8),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.layers(x)

model = CircleModelV0()

# 3. Define loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. Training loop
epochs = 100
model.train()

for epoch in range(epochs):
    # Forward pass
    y_logits = model(X_train)
    loss = criterion(y_logits, y_train)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```

### Multi-class Classification Training

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Set random seed
torch.manual_seed(42)

# 1. Prepare data
X, y = make_blobs(n_samples=1000, n_features=2, centers=4, random_state=42)
X = torch.FloatTensor(X)
y = torch.LongTensor(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Define model
class BlobModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )

    def forward(self, x):
        return self.layers(x)

model = BlobModel()

# 3. Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. Training loop
epochs = 100
model.train()

for epoch in range(epochs):
    # Forward pass
    y_logits = model(X_train)
    loss = criterion(y_logits, y_train)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```

---

## Inference (Making Predictions)

After training, use `model.eval()` and `torch.inference_mode()` for predictions.

### Binary Classification Inference

```python
# Set model to evaluation mode
model.eval()

# Make predictions
with torch.inference_mode():
    # Forward pass (raw logits)
    test_logits = model(X_test)

    # Convert to probabilities
    test_probs = torch.sigmoid(test_logits)

    # Convert to labels
    test_preds = (test_probs > 0.5).long()

# Calculate accuracy
accuracy = (test_preds == y_test).float().mean()
print(f'Test Accuracy: {accuracy.item()*100:.2f}%')
```

### Multi-class Classification Inference

```python
# Set model to evaluation mode
model.eval()

# Make predictions
with torch.inference_mode():
    # Forward pass (raw logits)
    test_logits = model(X_test)

    # Convert to probabilities
    test_probs = torch.softmax(test_logits, dim=1)

    # Convert to labels
    test_preds = torch.argmax(test_probs, dim=1)

# Calculate accuracy
accuracy = (test_preds == y_test).float().mean()
print(f'Test Accuracy: {accuracy.item()*100:.2f}%')
```

---

## Decision Boundary Visualization

Visualizing what your model has learned.

### Decision Boundary Function

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(model, X, y, device='cpu'):
    """Plot decision boundary for classification model"""

    # Move model to device and set to eval mode
    model.to(device)
    model.eval()

    # Set min and max values with padding
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    # Create meshgrid
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )

    # Flatten and stack meshgrid
    mesh = np.c_[xx.ravel(), yy.ravel()]
    mesh_tensor = torch.FloatTensor(mesh).to(device)

    # Make predictions on meshgrid
    with torch.inference_mode():
        Z = model(mesh_tensor)

    # Handle binary vs multi-class
    if Z.shape[1] == 1:
        # Binary classification
        Z = torch.sigmoid(Z).reshape(xx.shape)
    else:
        # Multi-class classification
        Z = torch.argmax(Z, dim=1).reshape(xx.shape)

    # Plot decision boundary
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z.cpu(), alpha=0.3, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='RdYlBu', edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.grid(True, alpha=0.3)
    plt.colorbar(label='Class')
    plt.tight_layout()
    plt.show()
```

### Using the Decision Boundary Function

```python
# For binary classification
plot_decision_boundary(model, X_train.numpy(), y_train.numpy().squeeze())
```

---

## Tracking Training Progress

### Tracking Loss and Accuracy

```python
def train_model(model, X_train, y_train, X_test, y_test, epochs=100):
    """Train model and track progress"""

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Track metrics
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    # Training loop
    for epoch in range(epochs):
        # Training
        model.train()
        y_logits = model(X_train.to(device))
        loss = criterion(y_logits, y_train.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate train accuracy
        with torch.inference_mode():
            train_preds = (torch.sigmoid(y_logits) > 0.5).long()
            train_acc = (train_preds == y_train.to(device)).float().mean()

        # Testing
        model.eval()
        with torch.inference_mode():
            test_logits = model(X_test.to(device))
            test_loss = criterion(test_logits, y_test.to(device))
            test_preds = (torch.sigmoid(test_logits) > 0.5).long()
            test_acc = (test_preds == y_test.to(device)).float().mean()

        # Store metrics
        train_losses.append(loss.item())
        test_losses.append(test_loss.item())
        train_accuracies.append(train_acc.item())
        test_accuracies.append(test_acc.item())

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'  Train Loss: {loss.item():.4f}, Train Acc: {train_acc.item()*100:.2f}%')
            print(f'  Test Loss: {test_loss.item():.4f}, Test Acc: {test_acc.item()*100:.2f}%')
            print()

    return train_losses, test_losses, train_accuracies, test_accuracies
```

---

## Key Takeaways

| Concept | Description |
|---------|-------------|
| **5-step training loop** | Forward → Loss → Zero grad → Backward → Optimizer |
| **model.train()** | Enable dropout and batch norm training behavior |
| **model.eval()** | Disable dropout, use running statistics |
| **torch.inference_mode()** | Faster prediction, no gradient calculation |
| **Logits** | Raw model outputs (any real number) |
| **Sigmoid** | Binary: converts logits to [0, 1] probabilities |
| **Softmax** | Multi-class: converts logits to probabilities summing to 1 |
| **Threshold** | Binary: probability > 0.5 → class 1 |
| **Argmax** | Multi-class: index of max probability → class |
| **Decision boundary** | Visual representation of what model learned |

---

## Discussion Questions

1. **Why do we use `torch.inference_mode()` instead of `torch.no_grad()`?** What's the difference?

2. **What happens if you forget to call `model.eval()` before testing?** How would this affect your results?

3. **Why do we track both training and test accuracy?** What does it mean if training accuracy is high but test accuracy is low?

---

## Practice Exercises

1. **Implement training loop:**
   - Create a binary classifier
   - Implement 5-step training loop
   - Track loss over epochs

2. **Make predictions:**
   - Train a model
   - Use `model.eval()` and `torch.inference_mode()`
   - Convert logits → probs → labels

3. **Visualize decision boundary:**
   - Train a model on make_circles
   - Plot the decision boundary
   - Try with linear vs non-linear models

---

## Next Steps

- [Improving Models](06-improving-models.md) - Adding non-linearity for better performance
- [Evaluation Metrics](07-evaluation-metrics.md) - Calculating precision, recall, F1
- [Practice Exercise](../../module-03/neural-network-classification/02_training_and_predictions.py) - Hands-on training

---

**Last Updated:** January 2026
