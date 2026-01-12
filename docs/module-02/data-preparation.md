# Data Preparation

## Learning Objectives

By the end of this lesson, you will be able to:
- Create custom Dataset classes for your data
- Use DataLoader for efficient batch processing
- Apply data transformations and augmentations
- Handle different data types (images, text, tabular)
- Split data into training, validation, and test sets

## Introduction to Data Handling

PyTorch provides two key utilities for data handling:
- **`Dataset`**: Stores samples and their labels
- **`DataLoader`**: Wraps an iterable around the Dataset for easy batch processing

```python
from torch.utils.data import Dataset, DataLoader
```

## The Dataset Class

### Creating a Custom Dataset

```python
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        """
        Args:
            data: Array-like containing the samples
            labels: Array-like containing the labels
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        """Returns the total number of samples"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a single sample and its label
        This is called by the DataLoader to get each item
        """
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label
```

### Using the Custom Dataset

```python
# Sample data
data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
labels = [0, 1, 0]

# Create dataset
dataset = CustomDataset(data, labels)

# Access individual items
sample, label = dataset[0]
print(f"Sample: {sample}, Label: {label}")

# Get dataset length
print(f"Dataset length: {len(dataset)}")
```

### Dataset from Tensors

```python
from torch.utils.data import TensorDataset

# Create tensors
features = torch.randn(100, 10)
labels = torch.randint(0, 2, (100,))

# Create dataset
dataset = TensorDataset(features, labels)

# Access items
feature, label = dataset[0]
print(f"Feature shape: {feature.shape}, Label: {label}")
```

## The DataLoader

### Basic DataLoader Usage

```python
from torch.utils.data import DataLoader

# Create a simple dataset
data = torch.randn(100, 10)
labels = torch.randint(0, 2, (100,))
dataset = TensorDataset(data, labels)

# Create DataLoader
batch_size = 16
dataloader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True  # Shuffle data at every epoch
)

# Iterate through batches
for batch_idx, (features, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx}: Features shape {features.shape}, Labels shape {labels.shape}")
    if batch_idx >= 2:  # Show first 3 batches
        break
```

### DataLoader Parameters

| Parameter | Description | Default | Common Values |
|-----------|-------------|---------|---------------|
| `batch_size` | Number of samples per batch | `1` | `16`, `32`, `64`, `128` |
| `shuffle` | Shuffle data at every epoch | `False` | `True` for training |
| `num_workers` | Subprocesses for data loading | `0` | `2`, `4`, `8` |
| `drop_last` | Drop last incomplete batch | `False` | `True` for consistent batch size |
| `pin_memory` | Pin memory for faster GPU transfer | `False` | `True` when using GPU |

### Advanced DataLoader Configuration

```python
dataloader = DataLoader(
    dataset=dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,      # Use 4 worker processes
    drop_last=True,     # Drop incomplete last batch
    pin_memory=True,    # Pin memory for faster GPU transfer
    prefetch_factor=2,  # Prefetch 2 batches per worker
)

# Training loop with DataLoader
for epoch in range(num_epochs):
    for batch_idx, (features, labels) in enumerate(dataloader):
        # Move to GPU if available
        if torch.cuda.is_available():
            features = features.to('cuda')
            labels = labels.to('cuda')

        # Training code here
        pass
```

## Data Transformations

### Using torchvision.transforms

```python
from torchvision import transforms

# Common transformations for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),      # Resize image
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.ToTensor(),              # Convert to tensor
    transforms.Normalize(               # Normalize with mean and std
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### Custom Transformations

```python
class AddGaussianNoise:
    """Add Gaussian noise to a tensor"""
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise

# Use custom transform
transform = transforms.Compose([
    transforms.ToTensor(),
    AddGaussianNoise(mean=0.0, std=0.1),
])
```

### Applying Transforms in Dataset

```python
class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label
```

## Handling Different Data Types

### Tabular Data

```python
import pandas as pd

class TabularDataset(Dataset):
    """Dataset for tabular data from CSV"""
    def __init__(self, csv_file, feature_columns, label_column, transform=None):
        self.data = pd.read_csv(csv_file)
        self.features = self.data[feature_columns].values
        self.labels = self.data[label_column].values
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            features = self.transform(features)

        return features, label
```

### Image Data

```python
from PIL import Image
import os

class ImageFolderDataset(Dataset):
    """Dataset for images in a folder structure"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.labels = []

        # Load all images and labels
        for label, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                self.image_files.append(os.path.join(class_dir, img_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
```

### Text Data

```python
class TextDataset(Dataset):
    """Dataset for text classification"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }
```

## Data Splitting

### Train/Validation/Test Split

```python
import numpy as np
from torch.utils.data import random_split, Subset

# Method 1: Using random_split
dataset = TensorDataset(features, labels)

# Define split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Calculate split sizes
total_size = len(dataset)
train_size = int(train_ratio * total_size)
val_size = int(val_ratio * total_size)
test_size = total_size - train_size - val_size

# Split dataset
train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)  # For reproducibility
)

print(f"Train size: {len(train_dataset)}")
print(f"Validation size: {len(val_dataset)}")
print(f"Test size: {len(test_dataset)}")
```

### Method 2: Using Indices

```python
# Shuffle indices
indices = np.arange(len(dataset))
np.random.shuffle(indices)

# Calculate split points
train_end = int(train_ratio * len(indices))
val_end = train_end + int(val_ratio * len(indices))

# Create subsets
train_indices = indices[:train_end]
val_indices = indices[train_end:val_end]
test_indices = indices[val_end:]

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)
```

### Creating DataLoaders for Each Split

```python
# Training DataLoader (with shuffling)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,  # Shuffle training data
    num_workers=4
)

# Validation DataLoader (no shuffling)
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,  # No need to shuffle validation data
    num_workers=4
)

# Test DataLoader (no shuffling)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4
)
```

## Practical Examples

### Example 1: Complete Image Classification Data Pipeline

```python
from torchvision import datasets, transforms

# Define transformations
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(
    root='data/train',
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root='data/val',
    transform=test_transform
)

test_dataset = datasets.ImageFolder(
    root='data/test',
    transform=test_transform
)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)
```

### Example 2: Custom Dataset for Regression

```python
class RegressionDataset(Dataset):
    """Dataset for regression tasks"""
    def __init__(self, n_samples=1000, n_features=20):
        # Generate synthetic data
        self.X = torch.randn(n_samples, n_features)
        # Generate labels as linear combination + noise
        self.y = self.X.sum(dim=1, keepdim=True) + torch.randn(n_samples, 1) * 0.1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Usage
dataset = RegressionDataset(n_samples=1000, n_features=20)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for X_batch, y_batch in dataloader:
    print(f"Features: {X_batch.shape}, Targets: {y_batch.shape}")
    break
```

## Best Practices

| Practice | Description |
|----------|-------------|
| **Pin Memory** | Use `pin_memory=True` when using GPU for faster data transfer |
| **Multiple Workers** | Use `num_workers > 0` for parallel data loading |
| **Shuffle Training** | Always shuffle training data, never shuffle validation/test |
| **Consistent Batch Size** | Use `drop_last=True` for consistent batch sizes during training |
| **Reproducibility** | Set random seed when splitting data |
| **Data Augmentation** | Apply augmentations only to training data |

## Key Takeaways

| Concept | Description |
|---------|-------------|
| **Dataset** stores samples and labels | Custom classes implement `__len__` and `__getitem__` |
| **DataLoader** handles batching and shuffling | Key parameters: `batch_size`, `shuffle`, `num_workers` |
| **Transformations** preprocess data | Use `transforms.Compose` for transformation pipelines |
| **Data splits** prevent overfitting | Split into train/validation/test sets |
| **Type-specific handling** for different data | Images, text, and tabular data have different requirements |

## Practice Exercises

1. Create a custom Dataset for CSV data with 5 features and 1 label column
2. Implement data augmentation for images with at least 3 transformations
3. Split a dataset into 70% train, 15% validation, and 15% test sets
4. Create DataLoaders with appropriate settings for GPU training
5. Implement a custom transformation that adds random noise to tabular data

## Next Steps

- [Building Neural Networks](building-models.md) - Creating models with nn.Module
- [Training Loop Fundamentals](training-loop.md) - Implementing the training process

---

**Last Updated**: January 2026
