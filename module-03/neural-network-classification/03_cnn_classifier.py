"""
Exercise 3: CNN Image Classifier
Neural Network Classification - Module 3

This exercise covers:
- Building a Convolutional Neural Network (CNN)
- Training an image classifier
- Data augmentation for images
- Evaluating image classification models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================
# Part 1: CNN Architecture
# ============================================

print("=" * 60)
print("Part 1: CNN Architecture")
print("=" * 60)


class CNN(nn.Module):
    """Convolutional Neural Network for image classification"""

    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        # TODO: Define convolutional layers
        self.features = nn.Sequential(
            # Conv block 1: 3 -> 32 channels
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 224 -> 112

            # Conv block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112 -> 56

            # Conv block 3: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56 -> 28

            # Conv block 4: 128 -> 256 channels
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28 -> 14
        )

        # TODO: Define classifier layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """Forward pass"""
        x = self.features(x)
        x = self.classifier(x)
        return x


# TODO: Create model
model = CNN(num_classes=10)
print(model)

# TODO: Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# TODO: Test with dummy input
dummy_input = torch.randn(4, 3, 224, 224)
output = model(dummy_input)
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")


# ============================================
# Part 2: Data Augmentation
# ============================================

print("\n" + "=" * 60)
print("Part 2: Data Augmentation")
print("=" * 60)

# TODO: Define training transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# TODO: Define validation/test transformations
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("Training transformations:")
print(train_transform)

print("\nValidation transformations:")
print(test_transform)


# ============================================
# Part 3: Create Dummy Dataset
# ============================================

print("\n" + "=" * 60)
print("Part 3: Create Dummy Dataset")
print("=" * 60)


class DummyImageDataset(torch.utils.data.Dataset):
    """Dummy image dataset for demonstration"""

    def __init__(self, num_samples=1000, num_classes=10, transform=None):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random RGB image
        image = torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8)
        image = transforms.ToPILImage()(image)

        label = torch.randint(0, self.num_classes, (1,)).item()

        if self.transform:
            image = self.transform(image)

        return image, label


# TODO: Create datasets
train_dataset = DummyImageDataset(num_samples=1000, num_classes=10, transform=train_transform)
val_dataset = DummyImageDataset(num_samples=200, num_classes=10, transform=test_transform)
test_dataset = DummyImageDataset(num_samples=200, num_classes=10, transform=test_transform)

# TODO: Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

print(f"Train batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# TODO: Visualize a batch
images, labels = next(iter(train_loader))
print(f"\nBatch images shape: {images.shape}")
print(f"Batch labels: {labels[:5]}")


# ============================================
# Part 4: Training Function
# ============================================

print("\n" + "=" * 60)
print("Part 4: Training Function")
print("=" * 60)


def train_cnn(model, train_loader, val_loader, criterion, optimizer,
              device, num_epochs=5):
    """Train CNN model"""

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss /= total
        train_acc = 100. * correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= total
        val_acc = 100. * correct / total

        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_cnn.pth')
            print(f"  ✓ New best model saved!")

    return history


# TODO: Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Training on: {device}\n")
history = train_cnn(model, train_loader, val_loader, criterion, optimizer, device)


# ============================================
# Part 5: Evaluation
# ============================================

print("\n" + "=" * 60)
print("Part 5: Evaluation")
print("=" * 60)


def evaluate_cnn(model, test_loader, device):
    """Evaluate CNN model"""

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = 100. * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)

    return accuracy, all_preds, all_labels


# TODO: Load best model and evaluate
model.load_state_dict(torch.load('best_cnn.pth'))
test_acc, preds, labels = evaluate_cnn(model, test_loader, device)

print(f"Test Accuracy: {test_acc:.2f}%")


# ============================================
# Exercises
# ============================================

print("\n" + "=" * 60)
print("Exercises")
print("=" * 60)

# Exercise 1: Implement a residual block
print("\nExercise 1: Residual block")
# Create a ResidualBlock class and use it in the CNN

# Exercise 2: Implement learning rate scheduler
print("\nExercise 2: Learning rate scheduler")
# Add ReduceLROnPlateau or CosineAnnealingLR

# Exercise 3: Implement test-time augmentation (TTA)
print("\nExercise 3: Test-time augmentation")
# Average predictions over multiple augmented versions of each test image

# Exercise 4: Implement Grad-CAM visualization
print("\nExercise 4: Grad-CAM")
# Create Grad-CAM to visualize what the CNN focuses on

# Exercise 5: Implement mixup augmentation
print("\nExercise 5: Mixup augmentation")
# Implement mixup: blend two images and their labels


print("\n" + "=" * 60)
print("Exercise 3 Complete!")
print("=" * 60)
