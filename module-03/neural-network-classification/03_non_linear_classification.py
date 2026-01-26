"""
Exercise 3: Non-Linear Classification
Neural Network Classification - Module 3

This exercise covers:
- Why linear models fail on non-linear data
- Adding non-linearity with ReLU activation
- Building CircleModelV1 (non-linear)
- Visualizing decision boundaries
- Comparing linear vs non-linear models

Learning Mottos:
- If in doubt, run the code!
- Experiment, experiment, experiment!
- Visualize, visualize, visualize!
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================
# Part 1: Setup and Data Preparation
# ============================================

print("=" * 60)
print("Part 1: Setup and Data Preparation")
print("=" * 60)

# Create dataset
X, y = make_circles(n_samples=1000, noise=0.03, factor=0.5, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).unsqueeze(1)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move to device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

print(f"Training size: {len(X_train)}")
print(f"Test size: {len(X_test)}")

# ============================================
# Part 2: Define Linear Model (CircleModelV0)
# ============================================

print("\n" + "=" * 60)
print("Part 2: Linear Model (CircleModelV0)")
print("=" * 60)

class CircleModelV0(nn.Module):
    """Linear model - no activation"""
    def __init__(self):
        super(CircleModelV0, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(2, 8),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.layer_stack(x)

model_v0 = CircleModelV0().to(device)
print("CircleModelV0 (Linear):")
print(model_v0)

# ============================================
# Part 3: Define Non-Linear Model (CircleModelV1)
# ============================================

print("\n" + "=" * 60)
print("Part 3: Non-Linear Model (CircleModelV1)")
print("=" * 60)

class CircleModelV1(nn.Module):
    """Non-linear model with ReLU activation"""
    def __init__(self):
        super(CircleModelV1, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),  # Add non-linearity!
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.layer_stack(x)

model_v1 = CircleModelV1().to(device)
print("CircleModelV1 (Non-linear):")
print(model_v1)

# ============================================
# Part 4: Train Both Models
# ============================================

print("\n" + "=" * 60)
print("Part 4: Train Both Models")
print("=" * 60)

def train_model(model, model_name, epochs=100):
    """Train a model and return metrics"""
    print(f"\nTraining {model_name}...")

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

    # Calculate final accuracy
    model.eval()
    with torch.inference_mode():
        test_probs = torch.sigmoid(test_logits)
        test_preds = (test_probs > 0.5).long()
        accuracy = (test_preds == y_test).float().mean()

    print(f"{model_name} - Test Accuracy: {accuracy.item()*100:.2f}%")

    return train_losses, test_losses, accuracy.item()

# Train both models
losses_v0_train, losses_v0_test, acc_v0 = train_model(model_v0, "CircleModelV0 (Linear)")
losses_v1_train, losses_v1_test, acc_v1 = train_model(model_v1, "CircleModelV1 (Non-linear)")

# ============================================
# Part 5: Compare Training Curves
# ============================================

print("\n" + "=" * 60)
print("Part 5: Compare Training Curves")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss comparison
axes[0].plot(losses_v0_train, label='Linear (Train)', linestyle='--')
axes[0].plot(losses_v0_test, label='Linear (Test)', linestyle='--')
axes[0].plot(losses_v1_train, label='Non-linear (Train)')
axes[0].plot(losses_v1_test, label='Non-linear (Test)')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss: Linear vs Non-linear')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy comparison
axes[1].bar(['Linear', 'Non-linear'], [acc_v0, acc_v1], color=['#ff7f0e', '#2ca02c'])
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Final Test Accuracy')
axes[1].set_ylim(0, 100)
for i, v in enumerate([acc_v0, acc_v1]):
    axes[1].text(i, v + 2, f'{v*100:.1f}%', ha='center')

plt.tight_layout()
plt.show()

print("\nComparison:")
print(f"Linear Model Accuracy:     {acc_v0*100:.2f}%")
print(f"Non-linear Model Accuracy: {acc_v1*100:.2f}%")

# ============================================
# Part 6: Decision Boundary Visualization
# ============================================

print("\n" + "=" * 60)
print("Part 6: Decision Boundary Visualization")
print("=" * 60)

def plot_decision_boundary(model, X, y, title):
    """Plot decision boundary for a model"""
    model.eval()

    # Create meshgrid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )

    # Make predictions on meshgrid
    mesh = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device)
    with torch.inference_mode():
        Z = torch.sigmoid(model(mesh)).reshape(xx.shape)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z.cpu().numpy(), alpha=0.3, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='RdYlBu', edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.colorbar(label='Class')
    plt.tight_layout()
    plt.show()

# Plot decision boundaries
X_test_np = X_test.cpu().numpy()
y_test_np = y_test.cpu().numpy().squeeze()

plot_decision_boundary(model_v0, X_test_np, y_test_np,
                      f'Linear Model - Accuracy: {acc_v0*100:.1f}%')
plot_decision_boundary(model_v1, X_test_np, y_test_np,
                      f'Non-linear Model - Accuracy: {acc_v1*100:.1f}%')

# ============================================
# Part 7: Understanding Why Non-Linearity Matters
# ============================================

print("\n" + "=" * 60)
print("Part 7: Why Non-Linearity Matters")
print("=" * 60)

print("The make_circles dataset is NON-LINEAR:")
print("  - The pattern is: x² + y² > r² (circular)")
print("  - A linear model can only learn: w1*x + w2*y + b (straight line)")
print("  - No straight line can separate concentric circles!")

print("\nReLU activation enables non-linear decision boundaries:")
print("  - ReLU(x) = max(0, x)")
print("  - Allows the model to learn complex, non-linear patterns")
print("  - Multiple ReLU layers can approximate any function")

print("\nKey insight:")
print("  - Linear model: ~50% accuracy (no better than random)")
print("  - Non-linear model: ~99% accuracy (learns the circular pattern)")

# ============================================
# Exercises
# ============================================

print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

# Exercise 1: Add more hidden layers
print("\nExercise 1: Add more hidden layers")
print("Try deeper models with more ReLU layers:")
print("  - 2 hidden layers: Linear -> ReLU -> Linear -> ReLU -> Linear")
print("  - 3 hidden layers: Linear -> ReLU -> Linear -> ReLU -> Linear -> ReLU -> Linear")
print("Tip: Modify CircleModelV1 to add more layers")

# Exercise 2: Change hidden units
print("\nExercise 2: Change number of hidden units")
print("Try different widths:")
print("  - 4 hidden units")
print("  - 8 hidden units (default)")
print("  - 16 hidden units")
print("  - 32 hidden units")
print("Tip: Modify the 'hidden_size' in the model")

# Exercise 3: Try different activation functions
print("\nExercise 3: Try different activation functions")
print("Experiment with:")
print("  - nn.ReLU() (default)")
print("  - nn.LeakyReLU(0.01)")
print("  - nn.Tanh()")
print("Tip: Replace nn.ReLU() with other activations")

# Exercise 4: Train for more epochs
print("\nExercise 4: Train for more epochs")
print("Try different numbers of epochs:")
print("  - epochs=50")
print("  - epochs=100 (default)")
print("  - epochs=200")
print("Tip: Modify the 'epochs' parameter in train_model()")

# Exercise 5: Compare decision boundaries
print("\nExercise 5: Compare decision boundaries")
print("Questions to answer:")
print("  - What does the linear model's boundary look like?")
print("  - What does the non-linear model's boundary look like?")
print("  - Why can't a straight line separate the circles?")
print("Tip: Visualize, visualize, visualize!")

print("\n" + "=" * 60)
print("Exercise 3 Complete!")
print("Remember: If in doubt, run the code!")
print("Remember: Experiment, experiment, experiment!")
print("Remember: Visualize, visualize, visualize!")
print("=" * 60)
