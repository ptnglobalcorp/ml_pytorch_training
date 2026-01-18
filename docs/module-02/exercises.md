# Module 2 Exercises

Quick reference to all hands-on exercises for Module 2: PyTorch Workflow Fundamentals.

## Exercise Files Location

All exercise files are located in:
```
module-02/pytorch-workflow/
```

## Exercise Map

| Exercise File | Concepts Covered | Documentation Section | Difficulty |
|---------------|------------------|----------------------|------------|
| `01_data_preparation.py` | Synthetic data, splits, visualization | [Data Preparation](02-data-preparation.md) | Beginner |
| `02_building_models.py` | nn.Module, parameters, forward | [Building Models](03-building-models.md) | Beginner |
| `03_training_models.py` | Loss functions, training loop | [Training Loop](04-training-loop.md) | Intermediate |
| `04_inference_and_saving.py` | Inference, saving/loading | [Saving & Loading](05-saving-loading.md) | Intermediate |
| `05_complete_workflow.py` | End-to-end workflow | [Saving & Loading](05-saving-loading.md) | Intermediate |

## Running the Exercises

```bash
cd module-02/pytorch-workflow
python 01_data_preparation.py
python 02_building_models.py
python 03_training_models.py
python 04_inference_and_saving.py
python 05_complete_workflow.py
```

---

## Exercise Overview

### Exercise 1: Data Preparation

**File:** `01_data_preparation.py`

**What you'll learn:**
- Creating synthetic linear data
- Visualizing data distributions
- Splitting data into train/val/test sets
- Understanding the importance of data splits

**Key exercises:**
1. Create synthetic data with different weights/biases
2. Experiment with different split ratios
3. Add noise to the data
4. Visualize the data splits
5. Understand why we use three splits

---

### Exercise 2: Building Models

**File:** `02_building_models.py`

**What you'll learn:**
- Creating models by subclassing `nn.Module`
- Using `nn.Parameter` for learnable weights
- Implementing the `forward()` method
- Inspecting model parameters

**Key exercises:**
1. Create a linear regression model
2. Make predictions with untrained model
3. Print and inspect parameters
4. Visualize initial predictions
5. Create models with different initializations

---

### Exercise 3: Training Models

**File:** `03_training_models.py`

**What you'll learn:**
- Setting up loss functions and optimizers
- Implementing the 5-step training loop
- Tracking training progress
- Visualizing training curves

**Key exercises:**
1. Train model for different numbers of epochs
2. Experiment with different learning rates
3. Compare SGD vs Adam optimizers
4. Implement early stopping
5. Plot training and validation curves

---

### Exercise 4: Inference and Saving

**File:** `04_inference_and_saving.py`

**What you'll learn:**
- Making predictions in inference mode
- Saving model state with `state_dict()`
- Loading saved models
- Saving and loading checkpoints

**Key exercises:**
1. Save and load a trained model
2. Make predictions on new data
3. Save complete checkpoints with optimizer state
4. Load from checkpoint and resume training
5. Compare multiple trained models

---

### Exercise 5: Complete Workflow

**File:** `05_complete_workflow.py`

**What you'll learn:**
- Putting everything together end-to-end
- Writing device-agnostic code
- Hyperparameter experimentation
- Comparing multiple experiments

**Key exercises:**
1. Implement the complete workflow
2. Run experiments with different hyperparameters
3. Compare and analyze results
4. Visualize training curves
5. Make predictions with loaded models

---

## Exercise Tips

### Apply the Learning Methodology

Remember the three mottos from [Learning Methodology](../module-01/03-learning-methodology.md):

1. **If in doubt, run the code!**
   - Don't just read the exercise files—run them and observe the output
   - Modify parameters and see how the output changes
   - Print intermediate values to understand what's happening

2. **Experiment, experiment, experiment!**
   - Try different hyperparameter values
   - Break the code intentionally to understand error messages
   - Combine concepts in new ways

3. **Visualize, visualize, visualize!**
   - Plot data before training
   - Visualize training progress
   - Plot predictions vs actual values
   - Use different colors for different splits

### Getting the Most Out of Exercises

**Before coding:**
- Read the relevant documentation section first
- Review the learning objectives
- Understand what the exercise is trying to teach

**While coding:**
- Read the comments and TODO markers
- Run the code frequently to see outputs
- Don't skip the exercises at the end of each file

**After completing:**
- Try the challenge exercises
- Modify the code to test your understanding
- Create your own small variations

---

## Challenge Exercises

Once you've completed the basic exercises, try these challenges:

### Challenge 1: Multiple Features

```python
# Extend to multiple input features
# y = w1*x1 + w2*x2 + b
```

**What you'll learn:** Handling multi-dimensional data, extending linear regression.

### Challenge 2: Polynomial Regression

```python
# Fit a quadratic relationship
# y = ax² + bx + c
```

**What you'll learn:** Feature engineering, non-linear relationships.

### Challenge 3: Early Stopping

```python
# Implement automatic early stopping
# when validation loss doesn't improve for N epochs
```

**What you'll learn:** Preventing overfitting, model checkpointing.

### Challenge 4: Learning Rate Scheduling

```python
# Decrease learning rate during training
# when loss plateaus
```

**What you'll learn:** Optimization strategies, training dynamics.

### Challenge 5: Batch Training

```python
# Implement mini-batch training
# instead of full-batch gradient descent
```

**What you'll learn:** Data loading, efficiency considerations.

---

## Troubleshooting

### Common Issues

**Issue: CUDA out of memory**
```python
# Solution: Use smaller tensors or CPU
device = torch.device('cpu')  # Force CPU
```

**Issue: Runtime error: Expected all tensors to be on the same device**
```python
# Solution: Move tensors to the same device
X = X.to(device)
model = model.to(device)
```

**Issue: Loss becomes NaN**
```python
# Solution: Learning rate too high, decrease it
optimizer = optim.SGD(model.parameters(), lr=0.001)  # Was 0.1
```

**Issue: Model not learning (loss stays constant)**
```python
# Solution: Learning rate too low, increase it
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Was 0.0001
```

**Issue: Visualizations don't appear**
```python
# Solution: Make sure to call plt.show()
plt.plot(x, y)
plt.show()  # Don't forget this!
```

---

## Next Steps

After completing all Module 2 exercises:

1. **Review** the key concepts from each documentation section
2. **Build something small**: Create your own linear regression experiment
3. **Move to Module 3**: [Neural Network Classification](../module-03/README.md)

## Additional Practice

Looking for more practice?

- **Kaggle:** Try the [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition
- **PyTorch Tutorials:** Work through [official PyTorch tutorials](https://pytorch.org/tutorials/)
- **Create your own exercises:** Pick a real-world problem and solve it with regression

---

## Exercise Checklist

Use this checklist to track your progress:

- [ ] Run `01_data_preparation.py` and understand data splits
- [ ] Run `02_building_models.py` and understand nn.Module
- [ ] Run `03_training_models.py` and understand the training loop
- [ ] Run `04_inference_and_saving.py` and understand model persistence
- [ ] Run `05_complete_workflow.py` and understand the end-to-end workflow
- [ ] Complete at least 2 challenge exercises
- [ ] Create your own experiment with different hyperparameters

---

**Need Help?**

- Check the [PyTorch Documentation](https://pytorch.org/docs/stable/)
- Ask questions in the [PyTorch Forums](https://discuss.pytorch.org/)
- Review the documentation sections linked above

**Last Updated:** January 2026
