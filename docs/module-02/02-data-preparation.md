# Data Preparation

## Learning Objectives

By the end of this lesson, you will be able to:
- Create synthetic data for machine learning experiments
- Split data into training, validation, and test sets
- Understand the purpose of each data split
- Visualize data distributions and relationships

---

## Step 1: Data Preparation

The first step in any machine learning workflow is preparing your data. Real-world data comes from many sources—images, text files, databases, APIs—but before feeding it to a model, we need to convert it to numerical tensors and organize it properly.

In this lesson, we'll create **synthetic data**—data we generate ourselves with known parameters. This is incredibly valuable for learning because we know the "true" relationship, so we can verify that our model learns correctly.

### Creating Synthetic Data

We'll create a simple linear relationship:

$$y = 0.7 \times X + 0.3$$

```python
import torch
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

# Known parameters (what we want the model to learn)
weight = 0.7
bias = 0.3

# Create input data X
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)

# Create output data y using the linear formula
y = weight * X + bias

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"First 5 X values:\n{X[:5]}")
print(f"First 5 y values:\n{y[:5]}")
```

**Output:**
```
X shape: torch.Size([50, 1])
y shape: torch.Size([50, 1])
First 5 X values:
tensor([[0.0000],
        [0.0200],
        [0.0400],
        [0.0600],
        [0.0800]])
First 5 y values:
tensor([[0.3000],
        [0.3140],
        [0.3280],
        [0.3420],
        [0.3560]])
```

**Key points:**
- We use `torch.arange()` to create evenly spaced values from 0 to 1
- `.unsqueeze(dim=1)` converts the shape from `[50]` to `[50, 1]`—a column vector
- The output `y` is computed using the known weight and bias

### Visualizing the Data

Remember: **Visualize, visualize, visualize!**

```python
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
```

This visualization shows:
- **Blue dots**: The individual data points
- **Red line**: The true underlying relationship

---

## Splitting Data into Train/Validation/Test Sets

This is **the most important concept in machine learning**. Never train and test on the same data—this leads to overfitting and gives a false sense of performance.

### The Three Splits

| Split | Purpose | Usage | Typical Size |
|-------|---------|-------|--------------|
| **Training** | Model learns patterns here | Fit parameters | 70% of data |
| **Validation** | Tune hyperparameters | Prevent overfitting | 15% of data |
| **Test** | Final performance check | Unseen data | 15% of data |

### Why Three Splits?

Think of it like studying for an exam:

1. **Training set** = Your textbook and practice problems (you learn from these)
2. **Validation set** = Practice quizzes (you check your understanding, adjust study habits)
3. **Test set** = The final exam (you never saw these questions before—they measure true learning)

If you studied using the exact exam questions, you'd get 100% but wouldn't actually understand the material. Similarly, a model tested on training data performs perfectly but fails on new data.

### Implementing the Split

```python
# Calculate split indices
train_split = int(0.7 * len(X))   # 70% for training
val_split = int(0.85 * len(X))    # Additional 15% for validation (total 85%)

print(f"Total samples: {len(X)}")
print(f"Training samples: {train_split}")
print(f"Validation samples: {val_split - train_split}")
print(f"Test samples: {len(X) - val_split}")

# Split the data
X_train, y_train = X[:train_split], y[:train_split]
X_val, y_val = X[train_split:val_split], y[train_split:val_split]
X_test, y_test = X[val_split:], y[val_split:]

print(f"\nX_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")
```

**Output:**
```
Total samples: 50
Training samples: 35
Validation samples: 7
Test samples: 8

X_train shape: torch.Size([35, 1])
X_val shape: torch.Size([7, 1])
X_test shape: torch.Size([8, 1])
```

### Visualizing the Splits

```python
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
```

**What you see:**
- **Blue**: Training data (leftmost 70%)
- **Green**: Validation data (next 15%)
- **Red**: Test data (rightmost 15%)

---

## Understanding Data Splits

### Why Not Use All Data for Training?

Using all data for training is like giving students the exam questions to study. They'll ace the exam but won't actually understand the material.

**With proper splits:**
- Training set: Model learns patterns
- Validation set: We detect overfitting early
- Test set: We measure real-world performance

### What Each Split Teaches Us

```python
# Training set: Learn the relationship
print("Training set teaches the model:")
print(f"  X range: [{X_train.min():.2f}, {X_train.max():.2f}]")
print(f"  y range: [{y_train.min():.2f}, {y_train.max():.2f}]")

# Validation set: Tune hyperparameters
print("\nValidation set guides the learning:")
print(f"  X range: [{X_val.min():.2f}, {X_val.max():.2f}]")
print(f"  y range: [{y_val.min():.2f}, {y_val.max():.2f}]")

# Test set: Final evaluation
print("\nTest set measures final performance:")
print(f"  X range: [{X_test.min():.2f}, {X_test.max():.2f}]")
print(f"  y range: [{y_test.min():.2f}, {y_test.max():.2f}]")
```

### Common Split Ratios

| Scenario | Train/Val/Test | When to Use |
|----------|----------------|-------------|
| Large dataset (>100K samples) | 80/10/10 | Plenty of data for all splits |
| Medium dataset (~10K samples) | 70/15/15 | Default choice |
| Small dataset (<1K samples) | 60/20/20 | Need more data for testing |
| Very small dataset | Use cross-validation | When every sample matters |

---

## Key Takeaways

| Concept | Description |
|---------|-------------|
| **Synthetic data** | Generated from known parameters—great for learning |
| **Train/Val/Test** | Three-way split prevents overfitting |
| **Training set** | Model learns patterns here (70%) |
| **Validation set** | Tune hyperparameters, detect overfitting (15%) |
| **Test set** | Final evaluation on unseen data (15%) |
| **Visualize splits** | See how data is distributed across sets |

---

## Practice Exercises

1. **Change the parameters**: Try `weight = 1.5` and `bias = -0.2`. How does the visualization change?

2. **Experiment with split ratios**: Try 80/10/10 and 60/20/20. How does this affect the size of each split?

3. **Add noise to the data**:
   ```python
   noise = torch.randn_like(y) * 0.05  # Add random noise
   y_noisy = weight * X + bias + noise
   ```
   Visualize the noisy data. How does this differ from the clean data?

4. **Create non-linear data**: Generate quadratic data using `y = a*X^2 + b*X + c`. Can you visualize the relationship?

---

## Discussion Questions

1. **Why do we set a random seed?** What would happen if we didn't set `torch.manual_seed(42)` before creating data?

2. **Can splits be random instead of sequential?** What are the pros and cons of random shuffling before splitting?

3. **What if validation loss increases while training loss decreases?** What does this tell us about the model?

---

## Next Steps

Now that we have our data prepared and split, let's build a model to learn the relationship!

[Continue to: Building Models →](03-building-models.md)

---

**Practice this lesson:** [Exercise 1: Data Preparation](../../module-02/pytorch-workflow/01_data_preparation.py)

**Last Updated:** January 2026
