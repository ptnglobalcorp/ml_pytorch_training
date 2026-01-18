# Module 2: PyTorch Workflow Fundamentals

**Learn the complete deep learning workflow from data to deployment**

## Quick Start

1. **Read the conceptual guides** in Part 1 (start with Introduction & Setup)
2. **Practice with hands-on exercises** in [`../../module-02/pytorch-workflow/`](../../module-02/pytorch-workflow/)

```
Learn:  docs/module-02/                  →  Theory and concepts
Do:     module-02/pytorch-workflow/      →  Hands-on exercises
```

## Module Overview

This module teaches the end-to-end PyTorch workflow for building, training, and deploying deep learning models. You'll work through a complete linear regression example, learning each step of the process: from data preparation through model saving.

**What we'll build:** A linear regression model that learns the relationship `y = 0.7*X + 0.3` from synthetic data.

## Learning Objectives

By the end of this module, you will be able to:

### Workflow Fundamentals
- Understand the 6-step PyTorch workflow (data → build → train → evaluate → save → deploy)
- Create and split datasets into train/validation/test sets
- Build models by subclassing `nn.Module`

### Training & Evaluation
- Implement the 5-step training loop (forward, loss, zero grad, backward, optimizer step)
- Use loss functions and optimizers effectively
- Evaluate models on test data

### Model Persistence
- Save trained models using `state_dict`
- Load models for inference
- Write device-agnostic code that works on CPU and GPU

## Study Path

### Part 1: The PyTorch Workflow

Master the foundational steps of the deep learning workflow.

| # | Topic | Description | Practice |
|---|-------|-------------|----------|
| 1 | [Introduction & Setup](01-introduction-setup.md) | The big picture, learning mottos, environment setup | - |
| 2 | [Data Preparation](02-data-preparation.md) | Creating synthetic data, train/val/test splits, visualization | [`01_data_preparation.py`](../../module-02/pytorch-workflow/01_data_preparation.py) |
| 3 | [Building Models](03-building-models.md) | nn.Module, nn.Parameter, forward method | [`02_building_models.py`](../../module-02/pytorch-workflow/02_building_models.py) |

### Part 2: Training & Deployment

Complete the workflow with training, evaluation, and model persistence.

| # | Topic | Description | Practice |
|---|-------|-------------|----------|
| 4 | [Training Loop](04-training-loop.md) | Loss functions, optimizers, 5-step training loop, inference | [`03_training_models.py`](../../module-02/pytorch-workflow/03_training_models.py) |
| 5 | [Saving & Loading](05-saving-loading.md) | Model persistence, checkpoints, device-agnostic code | [`04_inference_and_saving.py`](../../module-02/pytorch-workflow/04_inference_and_saving.py)<br>[`05_complete_workflow.py`](../../module-02/pytorch-workflow/05_complete_workflow.py) |

### Exercises

- [Exercises Quick Reference](exercises.md) - Overview of all hands-on exercises

## The Learning Mottos

Throughout this module, apply these three core principles:

### 1. If in doubt, run the code!
Don't just read about concepts—execute them. Seeing the output builds intuition faster than studying theory.

### 2. Experiment, experiment, experiment!
Modify parameters, break things intentionally, try different approaches. Active learning creates deeper understanding than passive reading.

### 3. Visualize, visualize, visualize!
Plot your data, your training progress, your predictions. Visual patterns reveal insights that numbers alone cannot.

## Prerequisites

- **Module 1 completed**: Deep Learning Foundations with PyTorch
- **PyTorch installed**: [Install Guide](https://pytorch.org/get-started/locally/)
- **matplotlib installed**: `pip install matplotlib` (for visualizations)

## Running the Exercises

```bash
cd module-02/pytorch-workflow
python 01_data_preparation.py
python 02_building_models.py
python 03_training_models.py
python 04_inference_and_saving.py
python 05_complete_workflow.py
```

## Key Concepts

### The 6-Step PyTorch Workflow

| Step | Action | Description |
|------|--------|-------------|
| 1 | **Data Preparation** | Create and split data into train/val/test sets |
| 2 | **Build Model** | Define architecture by subclassing `nn.Module` |
| 3 | **Train** | Implement training loop with loss and optimizer |
| 4 | **Evaluate** | Test model on unseen data |
| 5 | **Save** | Persist trained parameters with `state_dict` |
| 6 | **Load** | Reload model for inference or deployment |

### The Three Data Splits

| Split | Purpose | Typical Usage |
|-------|---------|---------------|
| **Training** | Fit model parameters | 70% of data |
| **Validation** | Tune hyperparameters | 15% of data |
| **Test** | Final evaluation | 15% of data |

### The 5-Step Training Loop

```python
# 1. Forward pass
y_pred = model(X_train)

# 2. Calculate loss
loss = criterion(y_pred, y_train)

# 3. Zero gradients
optimizer.zero_grad()

# 4. Backward pass
loss.backward()

# 5. Update parameters
optimizer.step()
```

## Module Summary

After completing this module, you should:

- **Understand** the complete PyTorch workflow from start to finish
- **Be able to** build, train, and save your own models
- **Know how to** evaluate models and make predictions
- **Be ready** for Module 3: Neural Network Classification

## Additional Resources

### External References
- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [matplotlib Documentation](https://matplotlib.org/stable/)

### Internal Documentation
- [Complete Study Guide](../README.md) - Overall training navigation
- [Module 1: Deep Learning Foundations](../module-01/README.md) - Prerequisite concepts

## Next Steps

1. **Start with** [Introduction & Setup](01-introduction-setup.md) to understand the big picture
2. **Complete all exercises** in the `pytorch-workflow/` directory
3. **Review the key concepts** summary above
4. **Move to Module 3:** [Neural Network Classification](../module-03/README.md)

---

**Module Overview:** [../../module-02/](../../module-02/)

**Last Updated:** January 2026
