"""
Exercise 5: Complete PyTorch Workflow
PyTorch Workflow Fundamentals - Module 2

This exercise covers:
- Putting together the complete workflow
- Device-agnostic code
- Hyperparameter experimentation
- Training from scratch to deployment
- Comparing experiments

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
from datetime import datetime

# Set random seed for reproducibility
torch.manual_seed(42)

# ============================================
# Part 1: Complete Workflow Function
# ============================================

print("=" * 60)
print("Part 1: Complete Workflow Function")
print("=" * 60)


def train_linear_regression(
    weight=0.7,
    bias=0.3,
    train_ratio=0.7,
    val_ratio=0.15,
    learning_rate=0.01,
    epochs=100,
    device='cpu',
    save_model=False,
    model_name='linear_model'
):
    """
    Complete workflow for linear regression
    """
    print(f"\nTraining with lr={learning_rate}, epochs={epochs}")
    print("-" * 60)

    # 1. Prepare data
    X = torch.arange(0, 1, 0.02).unsqueeze(dim=1).to(device)
    y = weight * X + bias

    train_split = int(train_ratio * len(X))
    val_split = int((train_ratio + val_ratio) * len(X))

    X_train, y_train = X[:train_split], y[:train_split]
    X_val, y_val = X[train_split:val_split], y[train_split:val_split]
    X_test, y_test = X[val_split:], y[val_split:]

    # 2. Build model


    class LinearRegressionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(1))
            self.bias = nn.Parameter(torch.randn(1))

        def forward(self, x):
            return self.weight * x + self.bias

    model = LinearRegressionModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # 3. Train
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val)
            val_losses.append(val_loss.item())

    # 4. Evaluate
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test)
        test_loss = criterion(test_pred, y_test).item()

    print(f"Final - Train Loss: {train_losses[-1]:.4f}, "
          f"Val Loss: {val_losses[-1]:.4f}, Test Loss: {test_loss:.4f}")
    print(f"Learned - weight: {model.weight.item():.4f} (true: {weight}), "
          f"bias: {model.bias.item():.4f} (true: {bias})")

    # 5. Save
    results = {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_loss': test_loss,
        'final_weight': model.weight.item(),
        'final_bias': model.bias.item()
    }

    if save_model:
        os.makedirs('saved_models', exist_ok=True)
        save_path = f'saved_models/{model_name}_lr{learning_rate}_e{epochs}.pth'
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to: {save_path}")
        results['save_path'] = save_path

    return results


# TODO: Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================
# Part 2: Running a Single Experiment
# ============================================

print("\n" + "=" * 60)
print("Part 2: Running a Single Experiment")
print("=" * 60)

results = train_linear_regression(
    learning_rate=0.01,
    epochs=100,
    device=device,
    save_model=True,
    model_name='experiment_1'
)

# TODO: Visualize results
plt.figure(figsize=(12, 5))

# Training curve
plt.subplot(1, 2, 1)
plt.plot(results['train_losses'], label='Train', linewidth=2)
plt.plot(results['val_losses'], label='Validation', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.title('Training Progress')
plt.grid(True, alpha=0.3)

# Final predictions
plt.subplot(1, 2, 2)
X = torch.arange(0, 1, 0.02).unsqueeze(dim=1).to(device)
y_true = 0.7 * X + 0.3
with torch.no_grad():
    y_pred = results['model'](X)

plt.scatter(X.cpu(), y_true.cpu(), c='b', alpha=0.6, label='True data')
plt.plot(X.cpu(), y_pred.cpu(), 'r-', linewidth=2,
         label=f"Learned: y={results['final_weight']:.2f}X+{results['final_bias']:.2f}")
plt.plot(X.cpu(), 0.7 * X.cpu() + 0.3, 'g--', linewidth=2, label='True: y=0.7X+0.3')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Final Predictions')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# Part 3: Hyperparameter Experiments
# ============================================

print("\n" + "=" * 60)
print("Part 3: Hyperparameter Experiments")
print("=" * 60)

# TODO: Experiment with different learning rates
learning_rates = [0.001, 0.01, 0.1]
all_results = {}

for lr in learning_rates:
    print(f"\n{'='*60}")
    print(f"Experiment with learning_rate={lr}")
    print(f"{'='*60}")
    results = train_linear_regression(
        learning_rate=lr,
        epochs=100,
        device=device
    )
    all_results[lr] = results

# ============================================
# Part 4: Comparing Experiments
# ============================================

print("\n" + "=" * 60)
print("Part 4: Comparing Experiments")
print("=" * 60)

# TODO: Compare training curves
plt.figure(figsize=(10, 6))
for lr, results in all_results.items():
    plt.plot(results['train_losses'], label=f'lr={lr}', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Training Loss (MSE)')
plt.legend()
plt.title('Training Curves: Different Learning Rates')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# TODO: Compare final results
print(f"\nFinal Results Comparison:")
print(f"{'LR':<10} {'Test Loss':<12} {'Weight':<10} {'Bias':<10}")
print("-" * 45)
for lr, results in all_results.items():
    print(f"{lr:<10.3f} {results['test_loss']:<12.4f} "
          f"{results['final_weight']:<10.4f} {results['final_bias']:<10.4f}")

# ============================================
# Part 5: Making Predictions with Loaded Models
# ============================================

print("\n" + "=" * 60)
print("Part 5: Making Predictions with Loaded Models")
print("=" * 60)


# TODO: Load the best model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return self.weight * x + self.bias


# Load
loaded_model = LinearRegressionModel().to(device)
loaded_model.load_state_dict(torch.load('saved_models/experiment_1_lr0.01_e100.pth'))
loaded_model.eval()

print("Model loaded successfully!")

# TODO: Make predictions on new data
new_data = torch.tensor([[0.25], [0.5], [0.75]]).to(device)
with torch.no_grad():
    predictions = loaded_model(new_data)

print(f"\nPredictions for new data:")
for i, x in enumerate(new_data):
    print(f"  X={x.item():.2f} -> y={predictions[i].item():.4f}")

# ============================================
# Part 6: Complete Summary
# ============================================

print("\n" + "=" * 60)
print("Part 6: Complete Summary")
print("=" * 60)

print(f"\nWorkflow Complete!")
print(f"{'='*60}")
print(f"Summary:")
print(f"  - Trained {len(all_results)} models with different learning rates")
print(f"  - Best test loss: {min(r['test_loss'] for r in all_results.values()):.4f}")
print(f"  - Models saved to: saved_models/")
print(f"  - Device used: {device}")
print(f"\nKey Takeaways:")
print(f"  - Learning rate significantly affects convergence")
print(f"  - Too small: slow learning")
print(f"  - Too large: instability")
print(f"  - Just right: fast, stable convergence")

# ============================================
# Exercises
# ============================================

print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

# Exercise 1: Epoch experiments
print("\nExercise 1: Vary number of epochs")
# TODO: Train with 50, 100, 200, 500 epochs
# TODO: Compare results
# TODO: Identify when overfitting occurs
print("Tip: Look for when val loss starts increasing")

# Exercise 2: Optimizer comparison
print("\nExercise 2: Compare SGD vs Adam")
# TODO: Train with SGD and Adam
# TODO: Compare convergence speed
# TODO: Compare final results
print("Tip: optim.Adam(model.parameters(), lr=0.01)")

# Exercise 3: Noise robustness
print("\nExercise 3: Add noise to data")
# TODO: Add Gaussian noise to training data
# TODO: Train model on noisy data
# TODO: Compare with clean data results
print("Tip: y_noisy = y + torch.randn_like(y) * 0.1")

# Exercise 4: Different data ranges
print("\nExercise 4: Different data ranges")
# TODO: Try X in range [0, 2], [0, 10]
# TODO: Try negative values
# TODO: Analyze effect on training
print("Tip: Modify torch.arange() to change range")

# Exercise 5: Complete experiments
print("\nExercise 5: Design your own experiment")
# TODO: Come up with a hypothesis
# TODO: Design experiment to test it
# TODO: Run and analyze results
# TODO: Document findings
print("Tip: What happens if you change the weight and bias?")

print("\n" + "=" * 60)
print("Exercise 5 Complete!")
print("Remember: The three mottos apply to everything!")
print("  - If in doubt, run the code!")
print("  - Experiment, experiment, experiment!")
print("  - Visualize, visualize, visualize!")
print("=" * 60)
