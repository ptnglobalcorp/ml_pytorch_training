# Saving and Loading Models

## Learning Objectives

By the end of this lesson, you will be able to:
- Save trained models using `state_dict`
- Load saved models for inference
- Save and load complete checkpoints
- Write device-agnostic code that works on CPU and GPU
- Put together the complete end-to-end workflow

---

## Step 6: Saving and Loading Models

After training a model, you want to save it so you can use it later without retraining. PyTorch provides two main approaches:

### Saving Model State (Recommended)

Save only the learned parameters (weights and biases):

```python
# Save model state_dict
torch.save(model.state_dict(), 'linear_model.pth')

print(f"Model saved to 'linear_model.pth'")
```

**Why `state_dict`?**
- Contains only the learned parameters (small file size)
- More flexible (can load into different model architectures)
- Safer (avoids potential issues with pickling entire model)

### Loading Model State

```python
# Create a new model instance
loaded_model = LinearRegressionModel()

# Load the saved state
loaded_model.load_state_dict(torch.load('linear_model.pth'))
loaded_model.eval()  # Set to evaluation mode

print("Model loaded successfully!")
print(f"Weight: {loaded_model.weight.item():.4f}")
print(f"Bias: {loaded_model.bias.item():.4f}")
```

### Saving Complete Checkpoints

For resuming training, save a complete checkpoint including optimizer state:

```python
import datetime

# Create checkpoint dictionary
checkpoint = {
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_losses[-1],
    'val_loss': val_losses[-1],
    'hyperparameters': {
        'learning_rate': 0.01,
        'weight': weight,
        'bias': bias
    },
    'timestamp': datetime.datetime.now().isoformat(),
    'pytorch_version': torch.__version__,
}

# Save checkpoint
# Note: .tar is the PyTorch convention for checkpoints (contains more than just state_dict)
torch.save(checkpoint, 'checkpoint.tar')
print(f"Checkpoint saved to 'checkpoint.tar'")
```

> **💡 Why `.tar` for checkpoints?**
> > - `.pth`/`.pt` → Typically for model `state_dict` only (learned parameters)
> > - `.tar` → For full checkpoints containing multiple items (model + optimizer + metadata)
> >
> > This convention helps you quickly identify what's in each file at a glance.

### Loading from Checkpoint

```python
# Load checkpoint
loaded_checkpoint = torch.load('checkpoint.tar')

print(f"Checkpoint from epoch {loaded_checkpoint['epoch']}")
print(f"Train loss: {loaded_checkpoint['train_loss']:.4f}")
print(f"Val loss: {loaded_checkpoint['val_loss']:.4f}")
print(f"Saved: {loaded_checkpoint['timestamp']}")
print(f"PyTorch version: {loaded_checkpoint['pytorch_version']}")

# Restore model and optimizer
restored_model = LinearRegressionModel()
restored_model.load_state_dict(loaded_checkpoint['model_state_dict'])
restored_model.eval()

restored_optimizer = optim.SGD(restored_model.parameters(), lr=0.01)
restored_optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])

print("Model and optimizer restored!")
```

---

## Step 7: Making Predictions with Loaded Models

Once a model is loaded, use it to make predictions:

```python
# Load and use the model
loaded_model.eval()
with torch.no_grad():
    new_data = torch.tensor([[0.25], [0.5], [0.75]])
    predictions = loaded_model(new_data)

print("Predictions for new data:")
for i, x in enumerate(new_data):
    print(f"  X = {x.item():.2f} → y = {predictions[i].item():.4f}")
```

**Output:**
```
Predictions for new data:
  X = 0.25 → y = 0.4769
  X = 0.50 → y = 0.6517
  X = 0.75 → y = 0.8264
```

---

## Step 8: Device-Agnostic Code

Your code should work seamlessly on both CPU and GPU. PyTorch makes this easy:

### Setting the Device

```python
# Set device automatically
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Expected output:
# Using device: cuda  (if GPU available)
# Using device: cpu   (if no GPU)
```

> **💡 Why `.to(device)` instead of `set_default_device()`?**
>
> You saw `torch.set_default_device()` in the Introduction—it's a cleaner PyTorch 2.0+ feature. So why do we use `.to(device)` everywhere?
>
> **Two reasons:**
> 1. **Compatibility** — Works with both PyTorch 1.x and 2.0+, so your code runs anywhere
> 2. **Learning value** — Explicit device placement helps you understand exactly where your data lives (important for debugging!)
>
> Once you're comfortable, feel free to use `set_default_device()` in your own PyTorch 2.0+ projects for cleaner code.

### Moving Data and Models to Device

```python
# Move model to device
model = model.to(device)

# Move data to device
X_train = X_train.to(device)
y_train = y_train.to(device)
X_val = X_val.to(device)
y_val = y_val.to(device)

# All computations now happen on the selected device
y_pred = model(X_train)  # Automatically uses the device
```

### Complete Device-Agnostic Training Loop

```python
# Everything works on CPU or GPU automatically
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LinearRegressionModel().to(device)
X_train = X_train.to(device)
y_train = y_train.to(device)

for epoch in range(epochs):
    model.train()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## Device Management: Choosing Your Approach

PyTorch offers two ways to handle devices. Here's how to choose:

### Approach 1: Traditional (What this course uses)

```python
# Works with PyTorch 1.x AND 2.0+
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)
```

**Pros:**
- ✅ Compatible with all PyTorch versions
- ✅ Explicit — you see exactly where each tensor lives
- ✅ Flexible — different tensors can go to different devices
- ✅ Common in existing codebases and tutorials

**Cons:**
- ❌ Verbose — need `.to(device)` on every tensor
- ❌ Easy to forget moving a tensor

**Use when:** Learning, working with older code, or need explicit control

---

### Approach 2: PyTorch 2.0+ Default Device

```python
# PyTorch 2.0+ only
torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')

# All tensors automatically created on default device
model = LinearRegressionModel()
x = torch.randn(2, 3)  # Already on CUDA if available
```

**Pros:**
- ✅ Cleaner — no repetitive `.to(device)` calls
- ✅ Less error-prone — can't forget to move a tensor
- ✅ Modern — PyTorch's recommended direction

**Cons:**
- ❌ PyTorch 2.0+ only
- ❌ Implicit — harder to trace device placement
- ❌ Global — affects your entire program

**Use when:** PyTorch 2.0+ environment, new projects, prefer cleaner code

---

### Quick Reference

| Scenario | Recommended Approach |
|----------|---------------------|
| **Learning PyTorch** | Traditional (`.to(device)`) |
| **Sharing code widely** | Traditional (max compatibility) |
| **Personal projects (2.0+)** | `set_default_device()` |
| **Multi-GPU or complex setups** | Traditional (explicit control) |

---

### Bottom Line

> **This course uses `.to(device)`** so you learn the fundamentals and build compatible code. Once you understand device management, `set_default_device()` is a great cleaner option for your PyTorch 2.0+ projects.

---

## Putting It All Together: Complete Workflow

Here's the complete end-to-end script combining everything:

```python
"""
Complete PyTorch Workflow for Linear Regression
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ============================================
# 1. Setup
# ============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

torch.manual_seed(42)

# ============================================
# 2. Prepare Data
# ============================================
weight = 0.7
bias = 0.3

X = torch.arange(0, 1, 0.02).unsqueeze(dim=1).to(device)
y = weight * X + bias

# Split data
train_split = int(0.7 * len(X))
val_split = int(0.85 * len(X))

X_train, y_train = X[:train_split], y[:train_split]
X_val, y_val = X[train_split:val_split], y[train_split:val_split]
X_test, y_test = X[val_split:], y[val_split:]

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# ============================================
# 3. Build Model
# ============================================
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return self.weight * x + self.bias

model = LinearRegressionModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ============================================
# 4. Train Model
# ============================================
epochs = 100
train_losses = []
val_losses = []

print(f"\nTraining for {epochs} epochs...")

for epoch in range(epochs):
    # Training
    model.train()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val)
        val_losses.append(val_loss.item())

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")

# ============================================
# 5. Save Model
# ============================================
torch.save(model.state_dict(), 'linear_model.pth')
print(f"\nModel saved!")

# ============================================
# 6. Load and Evaluate
# ============================================
loaded_model = LinearRegressionModel().to(device)
loaded_model.load_state_dict(torch.load('linear_model.pth'))
loaded_model.eval()

with torch.no_grad():
    final_weight = loaded_model.weight.item()
    final_bias = loaded_model.bias.item()

print(f"\nFinal Results:")
print(f"  Learned weight: {final_weight:.4f} (true: {weight})")
print(f"  Learned bias: {final_bias:.4f} (true: {bias})")

# Test loss
model.eval()
with torch.no_grad():
    test_pred = model(X_test)
    test_loss = criterion(test_pred, y_test).item()

print(f"  Test loss: {test_loss:.6f}")

# ============================================
# 7. Visualize Results
# ============================================
plt.figure(figsize=(12, 5))

# Training curves
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train', linewidth=2)
plt.plot(val_losses, label='Validation', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.title('Training Progress')
plt.grid(True, alpha=0.3)

# Final predictions
plt.subplot(1, 2, 2)
X_plot = torch.arange(0, 1, 0.02).unsqueeze(dim=1).to(device)
y_true = weight * X_plot + bias

with torch.no_grad():
    y_pred = loaded_model(X_plot)

plt.scatter(X_train.cpu(), y_train.cpu(), c='b', alpha=0.6, label='Train data')
plt.scatter(X_val.cpu(), y_val.cpu(), c='g', alpha=0.6, label='Val data')
plt.scatter(X_test.cpu(), y_test.cpu(), c='r', alpha=0.6, label='Test data')
plt.plot(X_plot.cpu(), y_pred.cpu(), 'r-', linewidth=2,
         label=f"Learned: y={final_weight:.2f}X+{final_bias:.2f}")
plt.plot(X_plot.cpu(), y_true.cpu(), 'g--', linewidth=2,
         label=f"True: y={weight}X+{bias}")
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Final Predictions')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✓ Complete workflow finished!")
```

---

## Best Practices

| Practice | Description | Why |
|----------|-------------|-----|
| **Save state_dict** | Save only parameters, not entire model | Smaller files, more flexible |
| **Use .tar for checkpoints** | `.tar` extension for full checkpoints, `.pth`/`.pt` for state_dict only | Follows PyTorch convention |
| **Add checkpoint metadata** | Include `timestamp` and `pytorch_version` in checkpoints | Reproducibility and debugging |
| **Set device explicitly** | Use `torch.device()` for CPU/GPU detection | Code works anywhere |
| **Use model.eval()** | Set mode before inference | Disables dropout/batch norm |
| **Disable gradients** | Use `torch.no_grad()` for predictions | Faster, less memory |
| **Track hyperparameters** | Save with checkpoints | Reproducibility |
| **Validate before testing** | Use validation set during training | Prevent overfitting |

---

## Key Takeaways

| Concept | Description |
|---------|-------------|
| **state_dict()** | Dictionary of model parameters (recommended for saving) |
| **torch.save()** | Save model or checkpoint to file |
| **torch.load()** | Load saved model or checkpoint |
| **model.eval()** | Set model to evaluation mode |
| **torch.no_grad()** | Disable gradient computation for efficiency |
| **Device-agnostic code** | Works on CPU and GPU automatically |

---

## Practice Exercises

1. **Save multiple models**: Train models with different learning rates, save each, and compare their predictions.

2. **Resume training**: Save a checkpoint at epoch 50, load it, and continue training to epoch 100.

3. **Compare optimizers**: Train with SGD and Adam, save both, and visualize their predictions side-by-side.

4. **GPU vs CPU**: Run the complete workflow on both CPU and GPU. Compare training time.

5. **Batch training**: Modify the code to use mini-batch training instead of full-batch.

---

## Discussion Questions

1. **Why save state_dict instead of the entire model?** What problems can occur when saving the full model?

2. **What information should be in a checkpoint?** Think about what you'd need to resume training exactly where you left off. Why include `timestamp` and `pytorch_version`?

3. **Why use `.tar` for checkpoints instead of `.pth`?** How does this convention help with file organization?

4. **How does device-agnostic code work?** Why is this important for sharing code?

5. **When should you use model.eval() vs model.train()?** What happens if you forget to switch modes?

---

## Module Summary

Congratulations! You've completed the PyTorch Workflow Fundamentals module. You now know:

1. **Data Preparation**: Creating and splitting data into train/val/test sets
2. **Building Models**: Subclassing `nn.Module` and defining parameters
3. **Training**: Implementing the 5-step training loop with loss and optimizer
4. **Evaluation**: Making predictions and computing test loss
5. **Saving/Loading**: Persisting models with `state_dict`
6. **Device-Agnostic Code**: Writing code that works on CPU and GPU

### What's Next?

**Module 3: Neural Network Classification**
- Apply your workflow knowledge to classification problems
- Learn about activation functions (ReLU, sigmoid, softmax)
- Build multi-class classifiers
- Handle real-world datasets

---

**Practice this lesson:**
- [Exercise 4: Inference and Saving](../../module-02/pytorch-workflow/04_inference_and_saving.py)
- [Exercise 5: Complete Workflow](../../module-02/pytorch-workflow/05_complete_workflow.py)

**Last Updated:** January 2026
