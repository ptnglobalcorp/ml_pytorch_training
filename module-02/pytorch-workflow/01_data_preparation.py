"""
Exercise 1: Data Preparation
PyTorch Workflow Fundamentals - Module 2

This exercise covers:
- Creating synthetic data for linear regression
- Visualizing data distributions
- Splitting data into train/val/test sets
- Understanding the importance of data splits

Learning Mottos:
- If in doubt, run the code!
- Experiment, experiment, experiment!
- Visualize, visualize, visualize!
"""

import torch
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

# ============================================
# Part 1: Creating Synthetic Data
# ============================================

print("=" * 60)
print("Part 1: Creating Synthetic Data")
print("=" * 60)

# TODO: Define known parameters
weight = 0.7
bias = 0.3

# TODO: Create input data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)

# TODO: Create output data using linear formula
y = weight * X + bias

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Total samples: {len(X)}")
print(f"\nFirst 5 X values:\n{X[:5]}")
print(f"\nFirst 5 y values:\n{y[:5]}")

# ============================================
# Part 2: Visualizing the Data
# ============================================

print("\n" + "=" * 60)
print("Part 2: Visualizing the Data")
print("=" * 60)

# TODO: Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(X, y, c='b', s=50, alpha=0.6, label='Data points')
plt.plot(X, weight * X + bias, 'r-', linewidth=2,
         label=f'True line: y = {weight}X + {bias}')
plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend(fontsize=10)
plt.title('Linear Regression Data', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Visualization created!")
print("Motto: Visualize, visualize, visualize!")

# ============================================
# Part 3: Splitting the Data
# ============================================

print("\n" + "=" * 60)
print("Part 3: Splitting the Data")
print("=" * 60)

# TODO: Calculate split points
train_split = int(0.7 * len(X))
val_split = int(0.85 * len(X))

print(f"Total samples: {len(X)}")
print(f"Train split index: {train_split}")
print(f"Val split index: {val_split}")

# TODO: Split the data
X_train, y_train = X[:train_split], y[:train_split]
X_val, y_val = X[train_split:val_split], y[train_split:val_split]
X_test, y_test = X[val_split:], y[val_split:]

print(f"\nTrain size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation size: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# ============================================
# Part 4: Visualizing Data Splits
# ============================================

print("\n" + "=" * 60)
print("Part 4: Visualizing Data Splits")
print("=" * 60)

# TODO: Create visualization of splits
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, c='b', s=50, alpha=0.6, label='Train (70%)')
plt.scatter(X_val, y_val, c='g', s=50, alpha=0.6, label='Validation (15%)')
plt.scatter(X_test, y_test, c='r', s=50, alpha=0.6, label='Test (15%)')
plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend(fontsize=10)
plt.title('Data Splits: Train/Validation/Test', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Data splits visualized!")

# ============================================
# Part 5: Understanding the Splits
# ============================================

print("\n" + "=" * 60)
print("Part 5: Understanding the Splits")
print("=" * 60)

# TODO: Print statistics for each split
print("\nTraining set statistics:")
print(f"  X range: [{X_train.min():.3f}, {X_train.max():.3f}]")
print(f"  y range: [{y_train.min():.3f}, {y_train.max():.3f}]")

print("\nValidation set statistics:")
print(f"  X range: [{X_val.min():.3f}, {X_val.max():.3f}]")
print(f"  y range: [{y_val.min():.3f}, {y_val.max():.3f}]")

print("\nTest set statistics:")
print(f"  X range: [{X_test.min():.3f}, {X_test.max():.3f}]")
print(f"  y range: [{y_test.min():.3f}, {y_test.max():.3f}]")

# ============================================
# Exercises
# ============================================

print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

# Exercise 1: Change weight and bias
print("\nExercise 1: Change weight and bias")
print("Try different values and visualize:")
# TODO: Try weight=1.5, bias=0.5
# TODO: Try weight=-0.5, bias=0.8
print("Tip: Modify the weight and bias variables at the top of the script")

# Exercise 2: Experiment with split ratios
print("\nExercise 2: Experiment with split ratios")
print("Try different train/val/test ratios:")
# TODO: Try 80/10/10 split
# TODO: Try 60/20/20 split
print("Tip: Modify the train_split and val_split calculations")

# Exercise 3: Add noise to the data
print("\nExercise 3: Add noise to the data")
# TODO: Add Gaussian noise to y
# TODO: Visualize noisy data
print("Tip: y_noisy = weight * X + bias + torch.randn_like(y) * 0.05")

# Exercise 4: Create non-linear data
print("\nExercise 4: Create non-linear data")
# TODO: Create quadratic data: y = ax^2 + bx + c
# TODO: Visualize the relationship
print("Tip: y = 0.5 * X**2 + 0.3 * X + 0.1")

# Exercise 5: Data exploration
print("\nExercise 5: Data exploration")
print("Answer these questions:")
print("  - What happens if X is not sorted?")
print("  - What happens with very few data points (e.g., step=0.1)?")
print("  - What happens with many data points (e.g., step=0.01)?")
print("Tip: Modify the 'step' variable and observe the changes")

print("\n" + "=" * 60)
print("Exercise 1 Complete!")
print("Remember: Experiment, experiment, experiment!")
print("=" * 60)
