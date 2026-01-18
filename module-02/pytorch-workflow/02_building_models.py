"""
Exercise 2: Building PyTorch Models
PyTorch Workflow Fundamentals - Module 2

This exercise covers:
- Creating models with nn.Module
- Understanding nn.Parameter
- Implementing the forward() method
- Inspecting model parameters
- Making predictions

Learning Mottos:
- If in doubt, run the code!
- Experiment, experiment, experiment!
- Visualize, visualize, visualize!
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

# ============================================
# Part 1: Creating the Linear Regression Model
# ============================================

print("=" * 60)
print("Part 1: Creating the Linear Regression Model")
print("=" * 60)


class LinearRegressionModel(nn.Module):
    """
    Simple linear regression model: y = weight * X + bias
    """

    def __init__(self):
        super().__init__()
        # TODO: Create learnable parameters
        self.weight = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass
        return self.weight * x + self.bias


# TODO: Create model instance
model = LinearRegressionModel()

print("Model created successfully!")
print(f"\nModel architecture:\n{model}")

# ============================================
# Part 2: Understanding nn.Parameter
# ============================================

print("\n" + "=" * 60)
print("Part 2: Understanding nn.Parameter")
print("=" * 60)

# TODO: Inspect parameters
print("\nModel parameters:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.item():.4f} (requires_grad={param.requires_grad})")

# TODO: Show parameter details
print(f"\nWeight details:")
print(f"  Value: {model.weight.item():.4f}")
print(f"  Shape: {model.weight.shape}")
print(f"  Gradient enabled: {model.weight.requires_grad}")

print(f"\nBias details:")
print(f"  Value: {model.bias.item():.4f}")
print(f"  Shape: {model.bias.shape}")
print(f"  Gradient enabled: {model.bias.requires_grad}")

# ============================================
# Part 3: Making Predictions
# ============================================

print("\n" + "=" * 60)
print("Part 3: Making Predictions")
print("=" * 60)

# Create test data
X_test = torch.tensor([[0.0], [0.5], [1.0]])

# TODO: Make predictions
with torch.no_grad():
    predictions = model(X_test)

print(f"\nInput values:\n{X_test.flatten()}")
print(f"\nPredictions:\n{predictions.flatten()}")
print(f"\nExpected (if weight=0.7, bias=0.3):\n{0.7 * X_test.flatten() + 0.3}")

# ============================================
# Part 4: Visualizing Initial Predictions
# ============================================

print("\n" + "=" * 60)
print("Part 4: Visualizing Initial Predictions")
print("=" * 60)

# Create data for visualization
X = torch.arange(0, 1, 0.02).unsqueeze(dim=1)
y_true = 0.7 * X + 0.3

# TODO: Make predictions
with torch.no_grad():
    y_pred = model(X)

# TODO: Visualize
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

print("Visualization created!")
print("Note: Predictions don't match yet because model is untrained.")

# ============================================
# Part 5: Inspecting Model State
# ============================================

print("\n" + "=" * 60)
print("Part 5: Inspecting Model State")
print("=" * 60)

# TODO: Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nTotal parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

# TODO: Show model state dict
print(f"\nModel state_dict keys:")
for key in model.state_dict().keys():
    print(f"  {key}")

# ============================================
# Exercises
# ============================================

print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

# Exercise 1: Create a model with different initialization
print("\nExercise 1: Different initialization")
# TODO: Initialize weight=0.0, bias=0.0
# TODO: Visualize predictions
print("Tip: Modify the nn.Parameter initialization in __init__")

# Exercise 2: Multiple input features
print("\nExercise 2: Multiple input features")


class MultiFeatureLinearRegression(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # TODO: Implement this
        # Hint: Use nn.Linear instead of individual parameters
        pass

    def forward(self, x):
        # TODO: Implement this
        pass


print("Tip: Use nn.Linear(input_size, 1) for multiple features")

# Exercise 3: Add activation function
print("\nExercise 3: Add non-linearity")


class NonLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))
        # TODO: Add a non-linear activation
        pass

    def forward(self, x):
        # TODO: Apply activation
        return self.weight * x + self.bias


print("Tip: Use torch.relu(), torch.sigmoid(), or torch.tanh()")

# Exercise 4: Print model summary
print("\nExercise 4: Model summary")
# TODO: Print layer-by-layer summary
# TODO: Count parameters per layer
print("Tip: Iterate over model.named_parameters() and print details")

# Exercise 5: Model comparison
print("\nExercise 5: Model comparison")
# TODO: Create multiple models with different initializations
# TODO: Compare their predictions
# TODO: Visualize all predictions on same plot
print("Tip: Create 3 models with different random seeds")

print("\n" + "=" * 60)
print("Exercise 2 Complete!")
print("Remember: If in doubt, run the code!")
print("=" * 60)
