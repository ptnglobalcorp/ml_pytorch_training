"""
Exercise 4: Inference and Model Saving
PyTorch Workflow Fundamentals - Module 2

This exercise covers:
- Making predictions in inference mode
- Understanding model.eval() and torch.no_grad()
- Saving model state
- Loading saved models
- Evaluating on test data

Learning Mottos:
- If in doubt, run the code!
- Experiment, experiment, experiment!
- Visualize, visualize, visualize!
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import datetime

# Set random seed for reproducibility
torch.manual_seed(42)

# ============================================
# Part 1: Train a Model (Quick Setup)
# ============================================

print("=" * 60)
print("Part 1: Training a Model")
print("=" * 60)

# TODO: Setup data
weight = 0.7
bias = 0.3
X = torch.arange(0, 1, 0.02).unsqueeze(dim=1)
y = weight * X + bias

train_split = int(0.7 * len(X))
val_split = int(0.85 * len(X))

X_train, y_train = X[:train_split], y[:train_split]
X_val, y_val = X[train_split:val_split], y[train_split:val_split]
X_test, y_test = X[val_split:], y[val_split:]

# TODO: Create and train model


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return self.weight * x + self.bias


model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Quick training
epochs = 100
for epoch in range(epochs):
    model.train()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"Model trained!")
print(f"Learned weight: {model.weight.item():.4f} (true: {weight})")
print(f"Learned bias: {model.bias.item():.4f} (true: {bias})")

# ============================================
# Part 2: Making Predictions in Inference Mode
# ============================================

print("\n" + "=" * 60)
print("Part 2: Making Predictions")
print("=" * 60)

# TODO: Set model to evaluation mode
model.eval()
print("Model set to evaluation mode")

# TODO: Make predictions on test data
with torch.no_grad():
    test_predictions = model(X_test)

print(f"\nTest predictions (first 5):")
print(f"  X: {X_test[:5].flatten()}")
print(f"  Predicted: {test_predictions[:5].flatten()}")
print(f"  Actual: {y_test[:5].flatten()}")

# TODO: Calculate test loss
test_loss = criterion(test_predictions, y_test)
print(f"\nTest Loss (MSE): {test_loss.item():.4f}")

# ============================================
# Part 3: Visualizing Predictions
# ============================================

print("\n" + "=" * 60)
print("Part 3: Visualizing Predictions")
print("=" * 60)

# TODO: Plot test predictions
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, c='b', s=50, alpha=0.6, label='Actual data')
plt.scatter(X_test, test_predictions, c='r', s=50, alpha=0.6, label='Predictions')
plt.plot(X_test, test_predictions, 'r-', linewidth=2, alpha=0.3)
plt.plot(X_test, weight * X_test + bias, 'g--', linewidth=2, label=f'True: y={weight}X+{bias}')
plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend(fontsize=10)
plt.title('Test Set: Predictions vs Actual', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Predictions visualized!")

# ============================================
# Part 4: Saving the Model
# ============================================

print("\n" + "=" * 60)
print("Part 4: Saving the Model")
print("=" * 60)

# TODO: Create directory if it doesn't exist
os.makedirs('saved_models', exist_ok=True)

# TODO: Save model state dict
model_path = 'saved_models/linear_model.pth'
torch.save(model.state_dict(), model_path)

print(f"Model saved to: {model_path}")

# TODO: Verify file was created
if os.path.exists(model_path):
    file_size = os.path.getsize(model_path)
    print(f"File size: {file_size} bytes")
else:
    print("Error: File was not created!")

# ============================================
# Part 5: Loading the Model
# ============================================

print("\n" + "=" * 60)
print("Part 5: Loading the Model")
print("=" * 60)

# TODO: Create new model instance
loaded_model = LinearRegressionModel()
print("Created new model instance")
print(f"Parameters before loading:")
print(f"  Weight: {loaded_model.weight.item():.4f}")
print(f"  Bias: {loaded_model.bias.item():.4f}")

# TODO: Load saved state
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.eval()

print(f"\nParameters after loading:")
print(f"  Weight: {loaded_model.weight.item():.4f}")
print(f"  Bias: {loaded_model.bias.item():.4f}")

# TODO: Verify loaded model works
with torch.no_grad():
    loaded_predictions = loaded_model(X_test)
    verification_loss = criterion(loaded_predictions, y_test)

print(f"\nVerification:")
print(f"  Test Loss: {verification_loss.item():.4f}")
print(f"  Matches original: {torch.allclose(test_predictions, loaded_predictions)}")

# ============================================
# Part 6: Saving Complete Checkpoints
# ============================================

print("\n" + "=" * 60)
print("Part 6: Saving Complete Checkpoints")
print("=" * 60)

# TODO: Create checkpoint dictionary
checkpoint = {
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': criterion(model(X_train), y_train).item(),
    'val_loss': criterion(model(X_val), y_val).item(),
    'test_loss': test_loss.item(),
    'hyperparameters': {
        'learning_rate': 0.01,
        'weight': weight,
        'bias': bias
    },
    'timestamp': datetime.datetime.now().isoformat(),
    'pytorch_version': torch.__version__,
}

# TODO: Save checkpoint
# Note: .tar is the PyTorch convention for checkpoints (contains more than just state_dict)
checkpoint_path = 'saved_models/checkpoint.tar'
torch.save(checkpoint, checkpoint_path)

print(f"Checkpoint saved to: {checkpoint_path}")
print(f"Checkpoint contents:")
for key in checkpoint.keys():
    print(f"  {key}")

# ============================================
# Part 7: Loading from Checkpoint
# ============================================

print("\n" + "=" * 60)
print("Part 7: Loading from Checkpoint")
print("=" * 60)

# TODO: Load checkpoint
loaded_checkpoint = torch.load(checkpoint_path)

print(f"Loaded checkpoint from epoch {loaded_checkpoint['epoch']}")
print(f"Train loss: {loaded_checkpoint['train_loss']:.4f}")
print(f"Val loss: {loaded_checkpoint['val_loss']:.4f}")
print(f"Test loss: {loaded_checkpoint['test_loss']:.4f}")
print(f"Hyperparameters: {loaded_checkpoint['hyperparameters']}")

# TODO: Restore model and optimizer
restored_model = LinearRegressionModel()
restored_model.load_state_dict(loaded_checkpoint['model_state_dict'])
restored_model.eval()

restored_optimizer = optim.SGD(restored_model.parameters(), lr=0.01)
restored_optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])

print("\nModel and optimizer restored successfully!")

# ============================================
# Exercises
# ============================================

print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

# Exercise 1: Multiple model versions
print("\nExercise 1: Save multiple model versions")
# TODO: Train models with different learning rates
# TODO: Save each with a different name
# TODO: Load and compare them
print("Tip: Use f-strings for filenames: f'model_lr{lr}.pth'")

# Exercise 2: Inference on new data
print("\nExercise 2: Inference on new data")
# TODO: Create new data points outside training range
# TODO: Make predictions
# TODO: Discuss extrapolation
print("Tip: Try X values like -0.5, 1.5 (outside 0-1 range)")

# Exercise 3: Model comparison
print("\nExercise 3: Model comparison")
# TODO: Load multiple saved models
# TODO: Compare their predictions
# TODO: Visualize all on same plot
print("Tip: Plot multiple learned lines on one graph")

# Exercise 4: Save training history
print("\nExercise 4: Save training history")
# TODO: Save loss curves with checkpoint
# TODO: Load and plot training history
# TODO: Compare multiple training runs
print("Tip: Add 'train_losses': train_losses to checkpoint")

# Exercise 5: Resume training
print("\nExercise 5: Resume training")
# TODO: Load checkpoint
# TODO: Continue training from saved epoch
# TODO: Verify it continues correctly
print("Tip: Use checkpoint['epoch'] as starting point")

print("\n" + "=" * 60)
print("Exercise 4 Complete!")
print("Remember: Visualize, visualize, visualize!")
print("=" * 60)
