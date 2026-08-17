---
title: Convolutional Networks (CNN)
track: nn
order: 8
estimatedTime: 40
difficulty: advanced
---

# Convolutional Networks (CNN)

Regular neural networks treat an image as a flat list of pixel values. A 224x224 color image would need 224 x 224 x 3 = 150,528 inputs. This is wasteful, because pixels close together are related in ways that pixels far apart are not. CNNs fix this by looking at small patches of the image at a time.

---

## 1. The Convolution Operation

A convolution slides a small **filter** (or kernel) across the image and produces a new image called a **feature map**.

Step through it yourself before reading the code:

```widget
{
  "type": "conv-stepper",
  "title": "Watch a 3×3 filter slide across a 5×5 input"
}
```

Each step: the filter looks at a 3×3 patch, multiplies element-wise, sums everything up → one output number. Then it moves one step right.

```python
import torch
import torch.nn as nn

# A convolution layer
conv = nn.Conv2d(
    in_channels=1,    # 1 for grayscale, 3 for RGB
    out_channels=8,   # number of different filters to learn
    kernel_size=3,    # 3x3 filter
    padding=1         # add border so output is same size as input
)
```

After learning, each filter picks up a different feature: edges, colors, textures.

**Quick check:** If your input has shape `(batch, 3, 64, 64)` (RGB, 64x64 pixels) and you apply a `Conv2d(3, 16, 3, padding=1)`, what is the output shape?
> `(batch, 16, 64, 64)` - 16 feature maps, same spatial size because of `padding=1`.

---

## 2. Stride and Pooling

**Stride** controls how far the filter moves each step. Stride 2 means the output is half the size in each dimension.

**Pooling** shrinks the feature maps and makes the network less sensitive to exact positions:

```python
# Max pooling: take the max in each 2x2 region
pool = nn.MaxPool2d(kernel_size=2, stride=2)
# Input:  (batch, 8, 64, 64)
# Output: (batch, 8, 32, 32)

# Average pooling: take the average
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

# Global average pooling: collapse each feature map to one number
# Turn (batch, 128, 7, 7) into (batch, 128)
gap = nn.AdaptiveAvgPool2d(1)
```

**The standard pattern:** Conv -> ReLU -> MaxPool, repeated. Each repetition learns more complex features and smaller spatial maps.

---

## 3. A Complete CNN in PyTorch

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # (batch, 32, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(2),                  # (batch, 32, 16, 16)

            nn.Conv2d(32, 64, 3, padding=1), # (batch, 64, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(2),                  # (batch, 64, 8, 8)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),                     # (batch, 64*8*8) = (batch, 4096)
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# For 32x32 RGB images, 10 classes (like CIFAR-10)
model = SimpleCNN(num_classes=10)
dummy = torch.randn(8, 3, 32, 32)   # batch of 8 images
print(model(dummy).shape)            # torch.Size([8, 10])
```

---

## 4. Transfer Learning (Very Important for Competitions)

Training a CNN from scratch needs a lot of data and time. Transfer learning lets you reuse a network already trained on millions of images (like ImageNet).

```python
import torchvision.models as models

# Load a pretrained ResNet18
model = models.resnet18(weights='IMAGENET1K_V1')

# Replace the final layer for your number of classes
num_classes = 5
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Option 1: Train only the final layer (fast, good when your data is small)
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad_(True)

# Option 2: Fine-tune everything (slower, better when you have more data)
for param in model.parameters():
    param.requires_grad = True
```

**Quick check:** Your dataset has 200 labeled images for a 4-class problem. Should you train from scratch or use transfer learning?
> Transfer learning. 200 images is far too few to train a CNN from scratch. Transfer learning gives you features already learned from millions of images.

Good pretrained models to know: `resnet18`, `resnet50`, `efficientnet_b0`, `vit_b_16` (Vision Transformer).

---

## 5. Image Data Transforms

Images need to be resized and normalized before going into a CNN:

```python
import torchvision.transforms as transforms

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),       # data augmentation
    transforms.RandomCrop(224, padding=8),   # data augmentation
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet statistics
        std =[0.229, 0.224, 0.225]    # use these for pretrained models
    )
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])
```

Data augmentation (flips, crops, rotations) is applied only to training data. Validation data is only resized and normalized.

---

## 6. Loading an Image Dataset

```python
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Expects folder structure: data/train/class_A/, data/train/class_B/, etc.
train_dataset = ImageFolder('data/train', transform=transform_train)
val_dataset   = ImageFolder('data/val',   transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=2)
```

---

## Summary

| Concept | What it does |
|---|---|
| `Conv2d(in, out, k, padding=1)` | Learn `out` filters of size kxk |
| `MaxPool2d(2)` | Halve the spatial size, keep strongest signals |
| `Flatten()` | Convert 2D feature maps to 1D for a Linear layer |
| `Dropout(0.5)` | Randomly zero 50% of neurons during training (prevents overfitting) |
| Transfer learning | Start from pretrained weights, change final layer |
| Augmentation | Random flips/crops during training to reduce overfitting |

**IOAI tip:** For most image tasks in competition, load a pretrained `resnet50` or `efficientnet_b0`, freeze the backbone, train only the head for a few epochs, then unfreeze and fine-tune with a small learning rate. This beats training from scratch in almost every case.
