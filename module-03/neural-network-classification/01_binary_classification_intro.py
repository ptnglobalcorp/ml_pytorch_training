"""
Exercise 1: Binary Classification Introduction
Neural Network Classification - Module 3

This exercise covers:
- Creating synthetic data with make_circles
- Visualizing classification data with matplotlib
- Setting up device-agnostic code (CPU/GPU)
- Building a linear model (CircleModelV0)
- Making initial predictions

Learning Mottos:
- If in doubt, run the code!
- Experiment, experiment, experiment!
- Visualize, visualize, visualize!
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================
# Part 1: Creating make_circles Dataset
# ============================================

print("=" * 60)
print("Part 1: Creating make_circles Dataset")
print("=" * 60)

# TODO: Create make_circles dataset
n_samples = 1000
X, y = make_circles(
    n_samples=n_samples,
    noise=0.03,
    factor=0.5,
    random_state=42
)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Class 0 samples: {sum(y == 0)}")
print(f"Class 1 samples: {sum(y == 1)}")

# ============================================
# Part 2: Visualizing the Data
# ============================================

print("\n" + "=" * 60)
print("Part 2: Visualizing the Data")
print("=" * 60)

# TODO: Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', s=10, alpha=0.6)
plt.title('make_circles Binary Classification Dataset', fontsize=14)
plt.xlabel('Feature 1 (x coordinate)', fontsize=12)
plt.ylabel('Feature 2 (y coordinate)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.colorbar(label='Class (0=Outer, 1=Inner)')
plt.tight_layout()
plt.show()

print("Visualization created!")
print("Motto: Visualize, visualize, visualize!")

# ============================================
# Part 3: Train/Test Split
# ============================================

print("\n" + "=" * 60)
print("Part 3: Train/Test Split")
print("=" * 60)

# TODO: Split data into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# TODO: Visualize the split
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu', s=10, alpha=0.6)
axes[0].set_title(f'Training Data ({len(X_train)} samples)')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdYlBu', s=10, alpha=0.6)
axes[1].set_title(f'Test Data ({len(X_test)} samples)')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# Part 4: Device-Agnostic Setup
# ============================================

print("\n" + "=" * 60)
print("Part 4: Device-Agnostic Setup")
print("=" * 60)

# TODO: Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# TODO: Convert to PyTorch tensors and move to device
X_train = torch.FloatTensor(X_train).to(device)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(device)

print(f"\nTensor shapes:")
print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_test: {y_test.shape}")

print(f"\nTensor dtypes:")
print(f"X_train: {X_train.dtype}")
print(f"y_train: {y_train.dtype}")

# ============================================
# Part 5: Building CircleModelV0 (Linear)
# ============================================

print("\n" + "=" * 60)
print("Part 5: Building CircleModelV0 (Linear)")
print("=" * 60)

# TODO: Define linear model
class CircleModelV0(nn.Module):
    """Linear model for binary classification"""

    def __init__(self, input_size=2, hidden_size=8, output_size=1):
        super(CircleModelV0, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            # Note: No activation (linear model)
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layer_stack(x)

# TODO: Create the model
model = CircleModelV0(input_size=2, hidden_size=8, output_size=1)
model = model.to(device)

print(model)

# TODO: Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# ============================================
# Part 6: Initial Predictions
# ============================================

print("\n" + "=" * 60)
print("Part 6: Initial Predictions")
print("=" * 60)

# TODO: Make predictions on test set
model.eval()
with torch.inference_mode():
    # Forward pass (raw logits)
    test_logits = model(X_test)

    # Convert to probabilities
    test_probs = torch.sigmoid(test_logits)

    # Convert to labels
    test_preds = (test_probs > 0.5).long()

# TODO: Show some examples
print("\nFirst 5 predictions:")
for i in range(5):
    print(f"  Sample {i+1}:")
    print(f"    Logits: {test_logits[i].item():.4f}")
    print(f"    Probability: {test_probs[i].item():.4f}")
    print(f"    Predicted: {test_preds[i].item()}, Actual: {y_test[i].item()}")

# TODO: Calculate initial accuracy
accuracy = (test_preds == y_test).float().mean()
print(f"\nInitial Test Accuracy: {accuracy.item()*100:.2f}%")
print("(Note: This is before training, so accuracy will be around 50%)")

# ============================================
# Part 7: Understanding the Logits → Labels Pipeline
# ============================================

print("\n" + "=" * 60)
print("Part 7: Logits → Probabilities → Labels Pipeline")
print("=" * 60)

# TODO: Demonstrate the pipeline
sample_idx = 0
logit = test_logits[sample_idx].item()
prob = test_probs[sample_idx].item()
pred = test_preds[sample_idx].item()
actual = y_test[sample_idx].item()

print(f"\nSample {sample_idx + 1}:")
print(f"  Input: X={X_test[sample_idx].cpu().numpy()}")
print(f"  1. Logits (raw output): {logit:.4f}")
print(f"  2. Probability (sigmoid): {prob:.4f}")
print(f"  3. Label (threshold > 0.5): {pred}")
print(f"  Actual label: {actual}")

# ============================================
# Exercises
# ============================================

print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

# Exercise 1: Experiment with different noise levels
print("\nExercise 1: Experiment with noise levels")
print("Try different noise values in make_circles:")
print("  - noise=0.0 (perfect circles)")
print("  - noise=0.03 (default)")
print("  - noise=0.1 (very noisy)")
print("Tip: Modify the 'noise' parameter in make_circles()")

# Exercise 2: Try different random seeds
print("\nExercise 2: Try different random seeds")
print("Set random_state to different values (42, 123, 456)")
print("Tip: Modify torch.manual_seed() and make_circles random_state")

# Exercise 3: Change train/test split ratio
print("\nExercise 3: Change train/test split ratio")
print("Try different test_size values:")
print("  - test_size=0.2 (80/20 split)")
print("  - test_size=0.3 (70/30 split)")
print("  - test_size=0.5 (50/50 split)")
print("Tip: Modify the 'test_size' parameter in train_test_split()")

# Exercise 4: Explore different hidden sizes
print("\nExercise 4: Explore different hidden sizes")
print("Try different hidden_size values:")
print("  - hidden_size=4")
print("  - hidden_size=8 (default)")
print("  - hidden_size=16")
print("Tip: Modify the 'hidden_size' parameter in CircleModelV0()")

# Exercise 5: Data exploration
print("\nExercise 5: Data exploration")
print("Answer these questions:")
print("  - What happens if you increase the noise?")
print("  - Why does the initial model have ~50% accuracy?")
print("  - Can a linear model separate concentric circles?")
print("Tip: Visualize, experiment, and run the code!")

print("\n" + "=" * 60)
print("Exercise 1 Complete!")
print("Remember: If in doubt, run the code!")
print("Remember: Experiment, experiment, experiment!")
print("Remember: Visualize, visualize, visualize!")
print("=" * 60)
