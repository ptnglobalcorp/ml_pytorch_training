# Introduction & Setup

## Learning Objectives

By the end of this lesson, you will be able to:
- Understand the big picture of the PyTorch workflow
- Apply the three learning mottos to your deep learning practice
- Set up your environment with PyTorch and matplotlib
- Verify your installation is working correctly

---

## The Big Picture: The PyTorch Workflow

Deep learning is fundamentally a game of two parts:

1. **Turning data into numbers** - Converting images, text, audio, or other inputs into numerical representations (tensors)
2. **Building models to learn patterns** - Creating neural networks that discover relationships in those numerical representations

In this module, we'll walk through the complete PyTorch workflow that connects these two parts.

### What We'll Build

We'll create a complete linear regression model that learns a simple relationship:

$$ y = 0.7 \times X + 0.3 $$

While this is a simple example, it contains all the essential components of training any deep learning model. Once you understand this workflow, you can scale it to complex problems like image classification, language modeling, and more.

### The 6-Step Workflow

```mermaid
graph LR
    A[1. Data Preparation] --> B[2. Build Model]
    B --> C[3. Train]
    C --> D[4. Evaluate]
    D --> E[5. Save Model]
    E --> F[6. Load & Deploy]
```

| Step | Action | What You'll Learn |
|------|--------|-------------------|
| 1 | **Data Preparation** | Create synthetic data and split into train/val/test sets |
| 2 | **Build Model** | Define neural network architecture with `nn.Module` |
| 3 | **Train** | Implement the training loop with loss functions and optimizers |
| 4 | **Evaluate** | Test model performance on unseen data |
| 5 | **Save** | Persist trained models using `state_dict` |
| 6 | **Load** | Reload models for inference and deployment |

---

## The Learning Mottos

Throughout this module (and your entire deep learning journey), keep these three principles in mind:

### 1. If in doubt, run the code!

> "Theory guides, experiment decides." — Anonymous

Don't get stuck trying to understand everything intellectually before running code. Deep learning is experimental—seeing the actual output often clarifies concepts faster than reading explanations.

**What this means in practice:**
- Run code examples even if you don't fully understand them yet
- Modify values and see how the output changes
- Print intermediate values to understand what's happening
- Use print statements liberally to inspect your data and model state

### 2. Experiment, experiment, experiment!

> "The best way to learn is by doing." — Ancient proverb

Passive reading builds familiarity; active experimentation builds understanding. Don't be afraid to break things—that's how you learn the boundaries.

**What this means in practice:**
- Try different hyperparameter values (learning rate, epochs, etc.)
- Change model architectures and observe the effects
- Intentionally create bugs to understand error messages
- Combine concepts in new ways

### 3. Visualize, visualize, visualize!

> "A picture is worth a thousand parameters." — Adapted proverb

Deep learning models process high-dimensional data that's impossible to understand as raw numbers. Visualization reveals patterns, problems, and insights that tables of numbers cannot.

**What this means in practice:**
- Plot your data before training
- Visualize training progress (loss curves)
- Plot predictions vs actual values
- Use different colors for train/val/test splits
- Always label your axes and add titles

---

## Environment Setup

### Prerequisites

Before starting, ensure you have:

- **Python 3.10+** installed
- **PyTorch installed** — [Get it here](https://pytorch.org/get-started/locally/)
- **matplotlib installed** — For visualizations: `pip install matplotlib`

### Verification Code

Run the following code to verify your environment is set up correctly:

```python
import torch
import matplotlib.pyplot as plt

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# Test matplotlib
plt.figure(figsize=(6, 4))
plt.plot([1, 2, 3], [1, 4, 9], 'o-')
plt.title("matplotlib is working!")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()

print("✓ Environment setup complete!")
```

**Expected output:**
```
PyTorch version: 2.x.x
CUDA available: True/False
Device: cuda/cpu
✓ Environment setup complete!
```

A matplotlib window should appear showing a simple line plot.

### PyTorch Version Notes

This module is compatible with both PyTorch 1.x and PyTorch 2.0+. PyTorch 2.0 introduces performance improvements and a new device management API, but all code in this module works with both versions.

If you're using PyTorch 2.0+, you can use the new `torch.set_default_device()` feature for cleaner device management:

```python
# PyTorch 2.0+ only
torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')

# All subsequent tensors are created on the default device
x = torch.randn(2, 3)  # Automatically on CUDA if available
```

---

## Why Start with Linear Regression?

You might wonder: why start with such a simple example? Here's why:

| Aspect | Linear Regression | Complex Models |
|--------|-------------------|----------------|
| **Understandable** | Every component is visible | Components hidden in abstraction |
| **Debuggable** | Easy to trace what's happening | Harder to isolate issues |
| **Visualizable** | Can plot 2D relationships | High-dimensional, hard to visualize |
| **Fast training** | Instant feedback | Training can take hours/days |
| **Same workflow** | Uses identical PyTorch patterns | Same patterns, just more complex |

Linear regression contains the **exact same workflow** as training a massive neural network:
- Data preparation
- Model definition
- Training loop
- Evaluation
- Saving/loading

Once you master these fundamentals with a simple example, scaling to complex models is just adding more layers—not learning new concepts.

---

## Key Takeaways

| Concept | Description |
|---------|-------------|
| **Deep Learning = Data + Models** | Turn data into numbers, then learn patterns in those numbers |
| **The 6-Step Workflow** | Prepare → Build → Train → Evaluate → Save → Load |
| **If in doubt, run the code!** | Execution builds intuition faster than reading |
| **Experiment actively** | Modify, break, and combine to build true understanding |
| **Visualize everything** | Plots reveal patterns that numbers hide |
| **Start simple** | Linear regression has the same workflow as complex models |

---

## Discussion Questions

1. **Why do we split data into train/validation/test sets?** What would happen if we trained on all our data and then tested on the same data?

2. **How does visualization help in deep learning?** Think of examples where seeing a plot would reveal problems that raw numbers wouldn't.

3. **What does "experiment" mean in the context of deep learning?** How can you be more experimental in your learning approach?

---

## Practice Exercises

1. **Verify your setup**: Run the verification code above and confirm everything works.

2. **Test your understanding**: What do you think each of these commands does? Run them to find out:
   ```python
   import torch
   x = torch.tensor([1, 2, 3])
   print(x.shape)
   print(x.dtype)
   print(x.device)
   ```

3. **Visualize something**: Create a simple matplotlib plot showing `y = x^2` for x values from -5 to 5.

---

## Next Steps

Now that you understand the big picture and have your environment set up, let's create some data!

[Continue to: Data Preparation →](02-data-preparation.md)

---

**Last Updated:** January 2026
