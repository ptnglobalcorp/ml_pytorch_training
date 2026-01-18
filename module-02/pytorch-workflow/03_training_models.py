"""
Exercise 3: Training Models
PyTorch Workflow Fundamentals - Module 2

This exercise covers:
- Setting up loss functions
- Using optimizers
- Implementing the 5-step training loop
- Tracking training progress
- Visualizing training curves

Learning Mottos:
- If in doubt, run the code!
- Experiment, experiment, experiment!
- Visualize, visualize, visualize!
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

# ============================================
# Part 1: Setup - Data and Model
# ============================================

print("=" * 60)
print("Part 1: Setup - Data and Model")
print("=" * 60)

# TODO: Create synthetic data
weight = 0.7
bias = 0.3
X = torch.arange(0, 1, 0.02).unsqueeze(dim=1)
y = weight * X + bias

# TODO: Split data
train_split = int(0.7 * len(X))
val_split = int(0.85 * len(X))

X_train, y_train = X[:train_split], y[:train_split]
X_val, y_val = X[train_split:val_split], y[train_split:val_split]

print(f"Train size: {len(X_train)}")
print(f"Validation size: {len(X_val)}")

# TODO: Create model


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return self.weight * x + self.bias


model = LinearRegressionModel()

print(f"\nInitial parameters:")
print(f"  Weight: {model.weight.item():.4f} (true: {weight})")
print(f"  Bias: {model.bias.item():.4f} (true: {bias})")

# ============================================
# Part 2: Loss Function and Optimizer
# ============================================

print("\n" + "=" * 60)
print("Part 2: Loss Function and Optimizer")
print("=" * 60)

# TODO: Define loss function
criterion = nn.MSELoss()
print(f"Loss function: MSELoss (Mean Squared Error)")

# TODO: Define optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)
print(f"Optimizer: SGD (learning rate=0.01)")

# TODO: Calculate initial loss
with torch.no_grad():
    initial_pred = model(X_train)
    initial_loss = criterion(initial_pred, y_train)

print(f"\nInitial training loss: {initial_loss.item():.4f}")

# ============================================
# Part 3: The 5-Step Training Loop
# ============================================

print("\n" + "=" * 60)
print("Part 3: The 5-Step Training Loop")
print("=" * 60)

# TODO: Implement one training step
print("\nOne training step:")

# Step 1: Forward pass
y_pred = model(X_train)
print(f"1. Forward pass - predictions shape: {y_pred.shape}")

# Step 2: Calculate loss
loss = criterion(y_pred, y_train)
print(f"2. Calculate loss - loss value: {loss.item():.4f}")

# Step 3: Zero gradients
optimizer.zero_grad()
print(f"3. Zero gradients - cleared previous gradients")

# Step 4: Backward pass
loss.backward()
print(f"4. Backward pass - computed gradients")
print(f"   Weight gradient: {model.weight.grad.item():.4f}")
print(f"   Bias gradient: {model.bias.grad.item():.4f}")

# Step 5: Update parameters
optimizer.step()
print(f"5. Update parameters - optimizer stepped")

print(f"\nParameters after one step:")
print(f"  Weight: {model.weight.item():.4f}")
print(f"  Bias: {model.bias.item():.4f}")

# ============================================
# Part 4: Complete Training Loop
# ============================================

print("\n" + "=" * 60)
print("Part 4: Complete Training Loop")
print("=" * 60)

# TODO: Reinitialize model
model = LinearRegressionModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 100
train_losses = []
val_losses = []

print(f"\nTraining for {epochs} epochs...")
print("-" * 60)

for epoch in range(epochs):
    ### Training
    model.train()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    ### Validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val)
        val_losses.append(val_loss.item())

    ### Print progress
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")

print(f"\nFinal parameters:")
print(f"  Weight: {model.weight.item():.4f} (true: {weight})")
print(f"  Bias: {model.bias.item():.4f} (true: {bias})")

# ============================================
# Part 5: Visualizing Training Progress
# ============================================

print("\n" + "=" * 60)
print("Part 5: Visualizing Training Progress")
print("=" * 60)

# TODO: Plot training curves
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

print("Training curve visualized!")

# TODO: Visualize predictions
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, c='b', s=50, alpha=0.6, label='Training data')
plt.scatter(X_val, y_val, c='g', s=50, alpha=0.6, label='Validation data')

with torch.no_grad():
    X_all = torch.cat([X_train, X_val])
    y_pred_all = model(X_all)
    plt.plot(X_all, y_pred_all, 'r-', linewidth=2,
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

print("Predictions visualized!")

# ============================================
# Exercises
# ============================================

print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

# Exercise 1: Learning rate experiments
print("\nExercise 1: Learning rate experiments")
# TODO: Train with lr=0.001, 0.01, 0.1
# TODO: Compare training curves
# TODO: Visualize all on same plot
print("Tip: Create a loop over different learning rates and store results")

# Exercise 2: Optimizer comparison
print("\nExercise 2: Optimizer comparison")
# TODO: Compare SGD vs Adam
# TODO: Compare convergence speed
# TODO: Compare final results
print("Tip: optim.Adam(model.parameters(), lr=0.01)")

# Exercise 3: Early stopping
print("\nExercise 3: Early stopping")
# TODO: Implement early stopping
# TODO: Stop when val loss doesn't improve for N epochs
# TODO: Save best model
print("Tip: Track best_val_loss and use a patience counter")

# Exercise 4: Loss functions
print("\nExercise 4: Different loss functions")
# TODO: Try MAE (L1Loss)
# TODO: Compare with MSE
# TODO: Discuss differences
print("Tip: criterion = nn.L1Loss()")

# Exercise 5: Training analysis
print("\nExercise 5: Training analysis")
# TODO: Plot gradient magnitudes during training
# TODO: Plot parameter updates
# TODO: Analyze convergence
print("Tip: Store param.grad.item() each epoch to plot gradients")

print("\n" + "=" * 60)
print("Exercise 3 Complete!")
print("Remember: Experiment, experiment, experiment!")
print("=" * 60)
