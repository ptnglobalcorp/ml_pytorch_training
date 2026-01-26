# Exercises Quick Reference

This page provides a quick overview of all hands-on exercises for Module 3: Neural Network Classification.

---

## Exercise Overview

| # | Exercise File | Topics Covered | Documentation |
|---|---------------|---------------|---------------|
| 1 | [`01_binary_classification_intro.py`](../../module-03/neural-network-classification/01_binary_classification_intro.py) | make_circles, visualization, device setup, CircleModelV0 | [Classification Introduction](01-classification-intro.md), [Data Preparation](03-data-preparation.md), [Building Models](04-building-models.md) |
| 2 | [`02_training_and_predictions.py`](../../module-03/neural-network-classification/02_training_and_predictions.py) | Training loop, logits→labels pipeline, inference | [Training & Evaluation](05-training-evaluation.md) |
| 3 | [`03_non_linear_classification.py`](../../module-03/neural-network-classification/03_non_linear_classification.py) | ReLU activation, decision boundaries, linear vs non-linear | [Improving Models](06-improving-models.md) |
| 4 | [`04_multi_class_classification.py`](../../module-03/neural-network-classification/04_multi_class_classification.py) | make_blobs, CrossEntropyLoss, argmax | [Building Models](04-building-models.md#multi-class-classification) |
| 5 | [`05_evaluation_metrics.py`](../../module-03/neural-network-classification/05_evaluation_metrics.py) | torchmetrics, precision/recall/F1, confusion matrix | [Evaluation Metrics](07-evaluation-metrics.md) |
| 6 | [`06_complete_classification_workflow.py`](../../module-03/neural-network-classification/06_complete_classification_workflow.py) | End-to-end workflow, hyperparameter tuning, model comparison | All lessons |

---

## Running the Exercises

```bash
# Navigate to exercise directory
cd module-03/neural-network-classification

# Run exercises in order
python 01_binary_classification_intro.py
python 02_training_and_predictions.py
python 03_non_linear_classification.py
python 04_multi_class_classification.py
python 05_evaluation_metrics.py
python 06_complete_classification_workflow.py
```

---

## Exercise Details

### Exercise 1: Binary Classification Introduction

**File:** `01_binary_classification_intro.py`

**What You'll Learn:**
- Creating synthetic data with `make_circles`
- Visualizing classification data with matplotlib
- Setting up device-agnostic code (CPU/GPU)
- Building a linear model (CircleModelV0)
- Making initial predictions

**Key Concepts:**
- Binary classification (2 classes)
- BCEWithLogitsLoss for binary classification
- Device-agnostic code patterns
- Data visualization

**Prerequisites:**
- Module 1: PyTorch Fundamentals
- Module 2: PyTorch Workflow Fundamentals

**Exercises:**
1. Experiment with different noise levels in make_circles
2. Try different random seeds
3. Visualize train/test splits

**Estimated Time:** 20-30 minutes

---

### Exercise 2: Training and Predictions

**File:** `02_training_and_predictions.py`

**What You'll Learn:**
- Implementing the 5-step training loop
- Converting logits → probabilities → labels
- Using `model.eval()` and `torch.inference_mode()`
- Tracking training progress
- Calculating accuracy

**Key Concepts:**
- Training loop mechanics
- Logits, probabilities, and labels pipeline
- Inference mode for predictions
- Binary classification metrics

**Prerequisites:**
- Exercise 1: Binary Classification Introduction

**Exercises:**
1. Try different thresholds (not just 0.5)
2. Experiment with learning rate
3. Track loss over epochs
4. Add more epochs to training

**Estimated Time:** 30-40 minutes

---

### Exercise 3: Non-Linear Classification

**File:** `03_non_linear_classification.py`

**What You'll Learn:**
- Why linear models fail on non-linear data
- Adding non-linearity with ReLU
- Building CircleModelV1 (non-linear)
- Visualizing decision boundaries
- Comparing linear vs non-linear models

**Key Concepts:**
- Non-linear activation functions (ReLU)
- Decision boundary visualization
- Model comparison
- When to add non-linearity

**Prerequisites:**
- Exercise 2: Training and Predictions

**Exercises:**
1. Add more hidden layers
2. Change number of hidden units
3. Train for more epochs
4. Compare different activation functions

**Estimated Time:** 30-40 minutes

---

### Exercise 4: Multi-Class Classification

**File:** `04_multi_class_classification.py`

**What You'll Learn:**
- Creating multi-class data with `make_blobs`
- Building multi-class models
- Using CrossEntropyLoss
- Converting logits with softmax
- Using argmax for predictions

**Key Concepts:**
- Multi-class classification (3+ classes)
- CrossEntropyLoss for multi-class
- Softmax activation
- Argmax for class selection

**Prerequisites:**
- Exercise 3: Non-Linear Classification

**Exercises:**
1. Vary number of classes (2, 4, 8)
2. Try different cluster configurations
3. Compare binary vs multi-class
4. Visualize multi-class decision boundaries

**Estimated Time:** 30-40 minutes

---

### Exercise 5: Evaluation Metrics

**File:** `05_evaluation_metrics.py`

**What You'll Learn:**
- Setting up torchmetrics
- Calculating accuracy, precision, recall, F1-score
- Creating confusion matrices
- Visualizing confusion matrices
- Comparing models using multiple metrics

**Key Concepts:**
- Beyond accuracy
- Precision vs recall trade-off
- F1-score for imbalanced data
- Confusion matrix interpretation

**Prerequisites:**
- Exercise 4: Multi-Class Classification

**Exercises:**
1. Compare metrics on imbalanced data
2. Visualize confusion matrix
3. Implement custom metric
4. Compare binary vs multi-class metrics

**Estimated Time:** 30-40 minutes

---

### Exercise 6: Complete Classification Workflow

**File:** `06_complete_classification_workflow.py`

**What You'll Learn:**
- End-to-end binary classification workflow
- End-to-end multi-class workflow
- Hyperparameter experimentation
- Model comparison and selection
- Saving and loading trained models

**Key Concepts:**
- Complete workflow integration
- Hyperparameter tuning
- Model persistence
- Model comparison

**Prerequisites:**
- Exercise 5: Evaluation Metrics

**Exercises:**
1. Design your own experiment
2. Compare different architectures
3. Create a model comparison report
4. Implement early stopping

**Estimated Time:** 40-60 minutes

---

## Challenge Exercises

After completing all exercises, try these challenges:

### Challenge 1: Improve make_circles Accuracy

**Goal:** Achieve >99% accuracy on make_circles dataset

**Hints:**
- Try deeper networks (3-4 hidden layers)
- Experiment with different activation functions
- Tune learning rate and epochs

### Challenge 2: Handle Class Imbalance

**Goal:** Build a classifier that works well on imbalanced data (90% class 0, 10% class 1)

**Hints:**
- Use weighted loss functions
- Experiment with different evaluation metrics
- Consider oversampling/undersampling

### Challenge 3: Multi-label Classification

**Goal:** Build a multi-label classifier (e.g., movie genres: action, comedy, drama)

**Hints:**
- Use BCEWithLogitsLoss with multiple outputs
- Each label is independent
- Threshold each label separately

### Challenge 4: Create Your Own Dataset

**Goal:** Create and classify your own synthetic dataset

**Ideas:**
- Spiral patterns (use polar coordinates)
- Multiple concentric circles
- Overlapping moons

### Challenge 5: Build a Classifier Pipeline

**Goal:** Create a reusable classifier pipeline class

**Features:**
- Accept any dataset
- Auto-detect binary vs multi-class
- Automatic metric calculation
- Decision boundary visualization
- Model saving/loading

---

## Troubleshooting

### Common Issues

**Issue: Model not learning (loss not decreasing)**
- Check learning rate (try 0.01 or 0.001)
- Verify loss function matches your classification type
- Ensure targets have correct dtype (float for binary, long for multi-class)

**Issue: Poor accuracy on make_circles**
- Add non-linearity (ReLU activation)
- Add more hidden layers or units
- Train for more epochs

**Issue: CUDA out of memory**
- Reduce batch size
- Use smaller model
- Close other GPU applications

**Issue: Metrics seem wrong**
- Verify prediction conversion (sigmoid/softmax)
- Check target shapes match predictions
- Ensure correct averaging for multi-class

---

## Learning Path

### Beginner Path

Complete exercises in order:
1. Exercise 1 → 2 → 3 (Binary classification)
2. Exercise 4 (Multi-class classification)
3. Exercise 5 (Evaluation metrics)
4. Exercise 6 (Complete workflow)

### Advanced Path

Focus on specific topics:
- **Model architecture:** Exercise 3 (non-linearity), Exercise 6 (hyperparameter tuning)
- **Evaluation:** Exercise 5 (metrics), Challenge 2 (imbalanced data)
- **Production:** Exercise 6 (saving/loading), Challenge 5 (pipeline)

---

## Additional Resources

### Documentation
- [PyTorch Classification Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [scikit-learn User Guide: Classification](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
- [torchmetrics Documentation](https://torchmetrics.readthedocs.io/)

### Practice Datasets
- [sklearn.datasets](https://scikit-learn.org/stable/datasets.html) - make_circles, make_blobs, make_moons
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php) - Real classification datasets

---

## Progress Tracker

Track your progress through the exercises:

| # | Exercise | Completed | Notes |
|---|----------|-----------|-------|
| 1 | Binary Classification Intro | ☐ | |
| 2 | Training and Predictions | ☐ | |
| 3 | Non-Linear Classification | ☐ | |
| 4 | Multi-Class Classification | ☐ | |
| 5 | Evaluation Metrics | ☐ | |
| 6 | Complete Workflow | ☐ | |

**Challenge Exercises:**
- ☐ Challenge 1: Improve make_circles Accuracy
- ☐ Challenge 2: Handle Class Imbalance
- ☐ Challenge 3: Multi-label Classification
- ☐ Challenge 4: Create Your Own Dataset
- ☐ Challenge 5: Build a Classifier Pipeline

---

**Last Updated:** January 2026
