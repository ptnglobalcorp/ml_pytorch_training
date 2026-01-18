# Training the Model

## Learning Objectives

By the end of this lesson, you will be able to:
- Choose appropriate loss functions for different tasks
- Set up optimizers to update model parameters
- Implement the 5-step PyTorch training loop
- Track training progress with loss curves
- Evaluate models on test data

---

## Step 3: Training the Model

Training is the process of adjusting model parameters to minimize the difference between predictions and actual values. This is where the learning happens!

### Loss Functions

A loss function measures how "wrong" our predictions are. During training, we try to minimize this loss.

```python
import torch.nn as nn

# For regression tasks (predicting continuous values)
criterion = nn.MSELoss()  # Mean Squared Error

# For classification tasks (predicting categories)
criterion = nn.CrossEntropyLoss()

# Other common loss functions
criterion = nn.L1Loss()        # Mean Absolute Error
criterion = nn.BCELoss()       # Binary Cross Entropy
```

**Mean Squared Error (MSE)** is appropriate for our linear regression task:

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_{pred} - y_{true})^2$$

### Optimizers

Optimizers update model parameters based on the computed gradients. They implement various optimization algorithms.

```python
import torch.optim as optim

# Stochastic Gradient Descent (SGD)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Adam (adaptive learning rate)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Other optimizers
optimizer = optim.RMSprop(model.parameters(), lr=0.01)
optimizer = optim.Adagrad(model.parameters(), lr=0.01)
```

| Optimizer | Best For | Notes |
|-----------|----------|-------|
| **SGD** | Simple problems, large datasets | Requires tuning learning rate |
| **Adam** | Most problems | Adaptive, works well out-of-box |
| **RMSprop** | RNNs, non-stationary problems | Good for recurrent networks |

**Learning rate (`lr`)**: Controls how big of a step the optimizer takes
- Too small: Learning is very slow
- Too large: Learning is unstable or diverges
- Just right: Fast, stable convergence

---

## The 5-Step Training Loop

Every PyTorch training loop follows the same 5 steps:

```python
# Step 1: Forward pass - make predictions
y_pred = model(X_train)

# Step 2: Calculate loss - measure error
loss = criterion(y_pred, y_train)

# Step 3: Zero gradients - clear previous gradients
optimizer.zero_grad()

# Step 4: Backward pass - compute gradients
loss.backward()

# Step 5: Update parameters - take optimization step
optimizer.step()
```

### What Each Step Does

| Step | Action | What Happens |
|------|--------|--------------|
| 1 | **Forward pass** | Input data flows through model → predictions |
| 2 | **Calculate loss** | Compare predictions to actual values → loss value |
| 3 | **Zero gradients** | Clear accumulated gradients from previous step |
| 4 | **Backward pass** | Compute gradients of loss w.r.t. parameters |
| 5 | **Optimizer step** | Update parameters using computed gradients |

---

## Complete Training Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

# 1. Prepare data (from previous lesson)
weight = 0.7
bias = 0.3
X = torch.arange(0, 1, 0.02).unsqueeze(dim=1)
y = weight * X + bias

# Split data
train_split = int(0.7 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
val_split = int(0.85 * len(X))
X_val, y_val = X[train_split:val_split], y[train_split:val_split]

# 2. Create model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return self.weight * x + self.bias

model = LinearRegressionModel()

# 3. Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. Training loop
epochs = 100
train_losses = []
val_losses = []

print(f"Training for {epochs} epochs...")
print("-" * 60)

for epoch in range(epochs):
    ### Training phase
    model.train()  # Set model to training mode
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    ### Validation phase
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val)
        val_losses.append(val_loss.item())

    ### Print progress
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")

print("\nTraining complete!")
print(f"Final parameters:")
print(f"  Weight: {model.weight.item():.4f} (true: {weight})")
print(f"  Bias: {model.bias.item():.4f} (true: {bias})")
```

**Output:**
```
Training for 100 epochs...
------------------------------------------------------------
Epoch   0: Train Loss = 0.0785, Val Loss = 0.5167
Epoch  20: Train Loss = 0.0115, Val Loss = 0.0207
Epoch  40: Train Loss = 0.0025, Val Loss = 0.0037
Epoch  60: Train Loss = 0.0009, Val Loss = 0.0012
Epoch  80: Train Loss = 0.0004, Val Loss = 0.0006

Training complete!
Final parameters:
  Weight: 0.6987 (true: 0.7)
  Bias: 0.3023 (true: 0.3)
```

---

## Visualizing Training Progress

```python
# Plot training curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss', linewidth=2)
plt.plot(val_losses, label='Validation Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.legend(fontsize=10)
plt.title('Training Progress', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**What you see:**
- Both losses decrease over time (model is learning!)
- Training loss is typically lower than validation loss
- Curves flatten as the model converges

### Visualizing Predictions After Training

```python
# Plot predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, c='b', s=50, alpha=0.6, label='Training data')
plt.scatter(X_val, y_val, c='g', s=50, alpha=0.6, label='Validation data')

# Model's learned line
X_all = torch.cat([X_train, X_val])
with torch.no_grad():
    y_pred = model(X_all)
    plt.plot(X_all, y_pred, 'r-', linewidth=2,
             label=f"Learned: y={model.weight.item():.2f}X+{model.bias.item():.2f}")
    plt.plot(X_all, weight * X_all + bias, 'g--', linewidth=2,
             label=f"True: y={weight}X+{bias}")

plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend(fontsize=10)
plt.title('Model Predictions After Training', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Step 4: Making Predictions (Inference)

Once trained, we use the model to make predictions on new, unseen data.

### Inference Mode

When making predictions, we should:
1. Set model to evaluation mode: `model.eval()`
2. Disable gradient computation: `torch.no_grad()`
3. Move data to the same device as the model

```python
# Prepare test data (unseen during training)
X_test = torch.tensor([[0.25], [0.5], [0.75]])
y_test_true = weight * X_test + bias

# Make predictions
model.eval()
with torch.no_grad():
    test_predictions = model(X_test)

print("Test predictions:")
for i, x in enumerate(X_test):
    print(f"  X={x.item():.2f} → Predicted: {test_predictions[i].item():.4f}, True: {y_test_true[i].item():.4f}")
```

**Output:**
```
Test predictions:
  X=0.25 → Predicted: 0.4769, True: 0.4750
  X=0.50 → Predicted: 0.6517, True: 0.6500
  X=0.75 → Predicted: 0.8264, True: 0.8250
```

### The Testing Loop

For comprehensive evaluation, we use a testing loop:

```python
def evaluate_model(model, X, y, criterion):
    """Evaluate model on given data"""
    model.eval()
    with torch.no_grad():
        predictions = model(X)
        loss = criterion(predictions, y)
    return loss.item(), predictions

# Evaluate on test set
test_loss, test_preds = evaluate_model(model, X_test, y_test_true, criterion)
print(f"\nTest Loss (MSE): {test_loss:.6f}")
```

---

## Step 5: Improving Through Experimentation

Training is an experimental process. Here are key hyperparameters to tune:

### Learning Rate Experiments

```python
learning_rates = [0.001, 0.01, 0.1]

for lr in learning_rates:
    print(f"\n{'='*60}")
    print(f"Experiment: learning_rate={lr}")
    print(f"{'='*60}")

    # Create fresh model
    model = LinearRegressionModel()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Train (simplified)
    for epoch in range(100):
        model.train()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Final weight: {model.weight.item():.4f} (true: {weight})")
    print(f"Final bias: {model.bias.item():.4f} (true: {bias})")
```

**Expected observations:**
- `lr=0.001`: Slow learning, may not converge
- `lr=0.01`: Good balance, converges well
- `lr=0.1`: Unstable, may overshoot or diverge

### Tracking Experiments

```python
results = {
    'learning_rate': [],
    'final_train_loss': [],
    'final_val_loss': [],
    'final_weight': [],
    'final_bias': []
}

# Log results after each experiment
results['learning_rate'].append(lr)
results['final_train_loss'].append(train_losses[-1])
results['final_val_loss'].append(val_losses[-1])
results['final_weight'].append(model.weight.item())
results['final_bias'].append(model.bias.item())
```

---

## Key Takeaways

| Concept | Description |
|---------|-------------|
| **Loss function** | Measures prediction error; MSE for regression |
| **Optimizer** | Updates parameters to minimize loss |
| **5-step loop** | Forward → Loss → Zero grad → Backward → Step |
| **model.train()** | Enable dropout, batch norm during training |
| **model.eval()** | Disable training-specific layers for inference |
| **torch.no_grad()** | Disable gradient computation for efficiency |
| **Hyperparameter tuning** | Experiment with learning rate, epochs, etc. |

---

## Practice Exercises

1. **Learning rate experiments**: Train with `lr=0.001, 0.01, 0.1`. Plot all training curves on one graph.

2. **Optimizer comparison**: Compare SGD vs Adam. Which converges faster?

3. **Early stopping**: Implement early stopping that halts training when validation loss doesn't improve for 10 epochs.

4. **More metrics**: Add Mean Absolute Error (MAE) and R² score to your evaluation.

5. **Learning rate scheduling**: Decrease the learning rate when loss plateaus.

---

## Discussion Questions

1. **Why validation and training losses?** What does each tell us? What if they diverge?

2. **When to stop training?** How do you know the model has converged?

3. **What if validation loss increases?** What does this indicate about the model?

4. **Why use model.eval()?** What happens if we forget this step?

---

## Next Steps

Our model is trained and making predictions! Now let's learn how to save and load it.

[Continue to: Saving & Loading →](05-saving-loading.md)

---

**Practice this lesson:** [Exercise 3: Training Models](../../module-02/pytorch-workflow/03_training_models.py)

**Last Updated:** January 2026
