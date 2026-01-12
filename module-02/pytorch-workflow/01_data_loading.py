"""
Exercise 1: Data Loading and Preparation
PyTorch Workflow Fundamentals - Module 2

This exercise covers:
- Creating custom datasets
- Using DataLoader for batching
- Applying data transformations
- Splitting data into train/val/test sets
"""

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import numpy as np
from torchvision import transforms

# ============================================
# Part 1: Creating a Custom Dataset
# ============================================

print("=" * 60)
print("Part 1: Creating a Custom Dataset")
print("=" * 60)


class CustomDataset(Dataset):
    """Custom dataset for tabular data"""

    def __init__(self, features, labels, transform=None):
        """
        Args:
            features: Array-like of features
            labels: Array-like of labels
            transform: Optional transform to apply to features
        """
        # TODO: Convert features and labels to tensors
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        """Return the number of samples"""
        # TODO: Return the length
        return len(self.features)

    def __getitem__(self, idx):
        """
        Return a single sample and its label

        Args:
            idx: Index of the sample to return

        Returns:
            tuple: (feature, label)
        """
        # TODO: Get feature and label at index
        feature = self.features[idx]
        label = self.labels[idx]

        # TODO: Apply transform if provided
        if self.transform:
            feature = self.transform(feature)

        return feature, label


# Create sample data
np.random.seed(42)
n_samples = 1000
n_features = 20

features = np.random.randn(n_samples, n_features)
labels = np.random.randint(0, 5, size=n_samples)  # 5 classes

# TODO: Create dataset instance
dataset = CustomDataset(features, labels)

print(f"Dataset size: {len(dataset)}")
print(f"First sample: {dataset[0]}")

# TODO: Access multiple samples using a loop
print("\nFirst 3 samples:")
for i in range(3):
    feature, label = dataset[i]
    print(f"  Sample {i}: feature shape={feature.shape}, label={label}")


# ============================================
# Part 2: Using DataLoader
# ============================================

print("\n" + "=" * 60)
print("Part 2: Using DataLoader")
print("=" * 60)

# TODO: Create a DataLoader with batch size 32
train_loader = DataLoader(
    dataset=dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

print(f"Number of batches: {len(train_loader)}")

# TODO: Iterate through one batch
print("\nOne batch of data:")
for features_batch, labels_batch in train_loader:
    print(f"  Features shape: {features_batch.shape}")
    print(f"  Labels shape: {labels_batch.shape}")
    break  # Just show first batch


# ============================================
# Part 3: Data Transformations
# ============================================

print("\n" + "=" * 60)
print("Part 3: Data Transformations")
print("=" * 60)


class NormalizeTransform:
    """Custom normalization transform"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        """Normalize the tensor"""
        # TODO: Implement normalization: (x - mean) / std
        return (x - self.mean) / self.std


# TODO: Create dataset with transformation
mean = features.mean(axis=0)
std = features.std(axis=0)
transform = NormalizeTransform(mean, std)

normalized_dataset = CustomDataset(features, labels, transform=transform)

print(f"Original first feature: {dataset[0][0][:3]}")
print(f"Normalized first feature: {normalized_dataset[0][0][:3]}")


# ============================================
# Part 4: Train/Val/Test Split
# ============================================

print("\n" + "=" * 60)
print("Part 4: Train/Val/Test Split")
print("=" * 60)

# TODO: Define split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# TODO: Calculate split sizes
total_size = len(dataset)
train_size = int(train_ratio * total_size)
val_size = int(val_ratio * total_size)
test_size = total_size - train_size - val_size

print(f"Train size: {train_size}")
print(f"Validation size: {val_size}")
print(f"Test size: {test_size}")

# TODO: Split the dataset
train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# TODO: Create DataLoaders for each split
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"\nTrain batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")


# ============================================
# Exercises
# ============================================

print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

# Exercise 1: Create a custom dataset that adds Gaussian noise to features
print("\nExercise 1: Add Gaussian noise to features")
class NoisyDataset(Dataset):
    """Dataset that adds noise to features"""
    def __init__(self, features, labels, noise_std=0.1):
        # TODO: Implement this
        pass

    def __len__(self):
        # TODO: Return length
        pass

    def __getitem__(self, idx):
        # TODO: Return noisy feature and label
        pass

# Exercise 2: Implement min-max normalization
print("\nExercise 2: Min-max normalization")
class MinMaxNormalize:
    """Min-max normalization to [0, 1]"""
    def __init__(self, min_val, max_val):
        # TODO: Store min and max
        pass

    def __call__(self, x):
        # TODO: Implement: (x - min) / (max - min)
        pass

# Exercise 3: Create a dataset that handles missing values
print("\nExercise 3: Handle missing values")
# Add NaN values to some features and implement a dataset
# that fills them with the mean

# Exercise 4: Implement weighted random sampling
print("\nExercise 4: Weighted random sampling")
# Use WeightedRandomSampler to handle class imbalance

# Exercise 5: Create a data augmentation pipeline for images
print("\nExercise 5: Image augmentation pipeline")
# Use torchvision.transforms to create an augmentation pipeline
# with at least 3 different transformations


print("\n" + "=" * 60)
print("Exercise 1 Complete!")
print("=" * 60)
