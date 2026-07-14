---
title: Convolutional Neural Networks (CNN)
track: nn
order: 8
estimatedTime: 55
difficulty: advanced
---

# Convolutional Neural Networks (CNNs)

Convolutional Neural Networks are the backbone of modern computer vision. They power image classifiers, object detectors, face recognition systems, and medical image analysis. Understanding CNNs deeply is essential for IOAI, where vision tasks appear frequently.

> **Note on running code:** All PyTorch code in this lesson requires a local Python environment with PyTorch installed. Run it locally — it cannot run in the browser.

---

## 1. The Problem with MLPs for Images

Before CNNs, people tried to use standard fully-connected networks (MLPs) for images. This fails for several reasons.

### Too Many Parameters

Consider a small 28x28 grayscale image (like MNIST). Flattened, it has 784 pixels. A single hidden layer with 512 neurons requires:

```
784 × 512 = 401,408 weights — just for the first layer
```

Now consider a 224x224 RGB image (like ImageNet):

```
224 × 224 × 3 = 150,528 inputs
First layer with 4096 neurons: 150,528 × 4096 = 616,562,688 weights
```

That is 616 million weights for ONE layer. This is:
- Impossibly memory-hungry
- Prone to overfitting
- Slow to train

### No Spatial Awareness

An MLP treats every pixel independently. If you shift a cat one pixel to the right, the network sees an entirely different input — there is no concept of "nearby pixels are related."

MLPs have **no translation invariance**. A cat in the top-left corner and a cat in the bottom-right corner look completely different to an MLP.

### No Reuse

The weights connecting pixel 1 to the first hidden neuron are completely separate from the weights connecting pixel 2. There is no concept of reusing the same "edge detector" at different positions.

CNNs solve all three problems.

---

## 2. The Convolution Operation

A convolution slides a small filter (also called a **kernel**) across the input, computing a dot product at each position. The filter captures a local pattern.

### Intuition with a 5x5 Example

Suppose we have this 5x5 image (values are pixel intensities):

```
Input (5x5):
 1  2  3  4  5
 6  7  8  9 10
11 12 13 14 15
16 17 18 19 20
21 22 23 24 25
```

And a 3x3 filter:

```
Filter (3x3):
 1  0 -1
 2  0 -2
 1  0 -1
```

This is a **Sobel filter** — it detects vertical edges (left side minus right side).

To compute the output at position (0,0), we align the filter with the top-left 3x3 block of the image:

```
Image patch:        Filter:         Element-wise product:
1  2  3             1  0 -1         1  0 -3
6  7  8       ×     2  0 -2     =  12  0 -16
11 12 13            1  0 -1         11  0 -13

Sum = 1 + 0 + (-3) + 12 + 0 + (-16) + 11 + 0 + (-13) = -8
```

We slide the filter one step right (stride=1) for position (0,1):

```
Image patch:        Filter:         Element-wise product:
2  3  4             1  0 -1         2  0 -4
7  8  9       ×     2  0 -2     =  14  0 -18
12 13 14            1  0 -1         12  0 -14

Sum = 2 + 0 + (-4) + 14 + 0 + (-18) + 12 + 0 + (-14) = -8
```

We continue sliding the filter across all positions. For a 5x5 input and 3x3 filter with stride 1 and no padding:

```
Output size = (input_size - filter_size) / stride + 1
            = (5 - 3) / 1 + 1
            = 3
```

So the output is a 3x3 **feature map**.

### Output Size Formula

For a single spatial dimension:

```
output_size = floor((input_size - kernel_size + 2 * padding) / stride) + 1
```

Examples:
- Input 28, kernel 3, stride 1, padding 0: output = (28 - 3 + 0) / 1 + 1 = **26**
- Input 28, kernel 3, stride 1, padding 1: output = (28 - 3 + 2) / 1 + 1 = **28** (same size!)
- Input 28, kernel 3, stride 2, padding 1: output = (28 - 3 + 2) / 2 + 1 = **14** (halved)

**Padding = 1 with kernel = 3 preserves spatial size.** This is called "same" padding.

---

## 3. What Filters Learn

The Sobel filter above was hand-designed. In a CNN, filters are **learned from data** through backpropagation.

### First Layer: Low-Level Features

The first convolutional layer learns simple patterns:
- Horizontal edges (bright-to-dark transitions in the vertical direction)
- Vertical edges
- Diagonal edges
- Colour blobs

If you visualise the filters of a trained CNN's first layer, you see small edge and colour detectors.

### Second Layer: Mid-Level Features

These filters respond to combinations of first-layer features:
- Corners (two edges meeting)
- Textures (repeating patterns of edges)
- Simple curves

### Deep Layers: High-Level Features

Deep filters respond to:
- Eyes, nose, wheels, windows — parts of objects
- Whole objects in the final layers

This hierarchy — from edges to parts to objects — is why CNNs work so well on natural images.

### Multiple Filters = Multiple Feature Maps

A single filter produces a single feature map. In practice, we use many filters simultaneously:

- Layer 1: 32 filters → 32 feature maps, each detecting a different low-level pattern
- Layer 2: 64 filters → 64 feature maps
- etc.

The number of filters is the number of **channels** in the output.

---

## 4. Stride and Padding

### Stride

Stride controls how far the filter moves at each step.

- **Stride 1**: move one pixel at a time — output is nearly the same size as input.
- **Stride 2**: move two pixels at a time — output is roughly half the size. Used as an alternative to pooling.

```
Input: 8x8
Filter: 3x3
Stride 1, padding 0: output = (8-3)/1 + 1 = 6x6
Stride 2, padding 0: output = (8-3)/2 + 1 = 3x3
```

### Padding

Padding adds zeros around the border of the input before applying the filter.

- **padding=0** (no padding, "valid"): Output shrinks with each conv layer.
- **padding=1** with a 3x3 filter: Output stays the same size ("same" padding).

```python
# Run this in your local Python environment with PyTorch installed
import torch.nn as nn

# Same-size output (stride=1, padding=1, kernel=3)
conv_same = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)

# Halve spatial dimensions (stride=2)
conv_half = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
```

---

## 5. The Channel Dimension

### Greyscale vs RGB

- A greyscale image has shape `(H, W)` — one intensity value per pixel.
- An RGB image has shape `(H, W, 3)` — three channels (Red, Green, Blue).
- In PyTorch, images are stored as `(C, H, W)` where C = channels.

### How Convolution Handles Channels

When the input has C_in channels and we apply a single filter, that filter must also have depth C_in. The filter shape is `(C_in, kH, kW)` and we compute a dot product across all channels simultaneously.

If we have C_out filters, the output has C_out channels. So:
- Input shape: `(C_in, H, W)`
- Filter bank shape: `(C_out, C_in, kH, kW)` — C_out filters, each of size C_in × kH × kW
- Output shape: `(C_out, H_out, W_out)`

```python
# Run this in your local Python environment with PyTorch installed
import torch
import torch.nn as nn

# Simulate a batch of 8 RGB images, 32x32 pixels
x = torch.randn(8, 3, 32, 32)   # (batch, channels, height, width)

# Conv layer: 3 input channels, 16 output channels, 3x3 kernel
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

out = conv(x)
print(out.shape)   # torch.Size([8, 16, 32, 32])
```

The parameter count of this layer:
```
(C_out × C_in × kH × kW) + C_out   [bias]
= (16 × 3 × 3 × 3) + 16
= 432 + 16 = 448 parameters
```

Versus a fully-connected layer with the same input/output size:
```
(3 × 32 × 32) × (16 × 32 × 32) = 3,072 × 16,384 = 50,331,648 parameters
```

CNNs use **parameter sharing** — the same filter is applied at every spatial location. This reduces parameters by 100,000x here.

---

## 6. Pooling

After convolution, pooling reduces the spatial dimensions. This:
- Reduces computation in subsequent layers
- Provides some translation invariance (small shifts don't matter)
- Increases the receptive field

### Max Pooling

Takes the maximum value in each pooling window. Most commonly used.

```
Input:           Pool 2x2, stride 2:    Output:
1  3  2  4           →                  6  8
5  6  1  2           max(1,3,5,6)=6     7  9
3  2  1  0           max(2,4,1,2)=4
1  2  7  9           max(3,2,1,2)=3
                     max(1,0,7,9)=9

Wait — 2x2 pool on 4x4 with stride 2:
Top-left 2x2: max(1,3,5,6)=6
Top-right 2x2: max(2,4,1,2)=4
Bottom-left 2x2: max(3,2,1,2)=3
Bottom-right 2x2: max(1,0,7,9)=9

Output (2x2): [[6, 4], [3, 9]]
```

```python
# Run this in your local Python environment with PyTorch installed
import torch.nn as nn

pool = nn.MaxPool2d(kernel_size=2, stride=2)

import torch
x = torch.tensor([[[[1., 3., 2., 4.],
                    [5., 6., 1., 2.],
                    [3., 2., 1., 0.],
                    [1., 2., 7., 9.]]]])  # (1, 1, 4, 4)
out = pool(x)
print(out)   # tensor([[[[6., 4.], [3., 9.]]]])
print(out.shape)  # torch.Size([1, 1, 2, 2])
```

### Average Pooling

Takes the average instead of the maximum. Used less often than max pooling for intermediate layers, but common at the end of modern architectures (`nn.AdaptiveAvgPool2d`).

### Global Average Pooling

Collapses each feature map to a single value by averaging all positions. Used in modern CNNs to replace flattening before the dense layers.

```python
# Run this in your local Python environment with PyTorch installed
import torch
import torch.nn as nn

# Adaptive average pooling: any input size → fixed output size
gap = nn.AdaptiveAvgPool2d((1, 1))

x = torch.randn(8, 64, 7, 7)    # (batch, channels, H, W)
out = gap(x)
print(out.shape)                  # torch.Size([8, 64, 1, 1])
out = out.flatten(1)
print(out.shape)                  # torch.Size([8, 64])
```

---

## 7. Standard CNN Architecture Pattern

The classic CNN architecture alternates between:
1. **Conv + ReLU** — extract features
2. **Pooling** — reduce spatial size
3. **Repeat** for deeper features
4. **Flatten** — convert to 1D
5. **Dense layers** — classify

```
Input image (e.g. 32×32×3)
     ↓
Conv(32 filters, 3×3) + ReLU   → 32×32×32
     ↓
MaxPool(2×2)                    → 16×16×32
     ↓
Conv(64 filters, 3×3) + ReLU   → 16×16×64
     ↓
MaxPool(2×2)                    → 8×8×64
     ↓
Conv(128 filters, 3×3) + ReLU  → 8×8×128
     ↓
Flatten                          → 8192
     ↓
Linear(8192, 256) + ReLU
     ↓
Linear(256, num_classes)
     ↓
Loss (CrossEntropyLoss)
```

---

## 8. PyTorch CNN: Complete Code

```python
# Run this in your local Python environment with PyTorch installed
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

# ---- Model ----
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()

        # Feature extraction
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # same padding
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),         # 32x32 -> 16x16

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),         # 16x16 -> 8x8

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # No pool here — keep some spatial information
        )

        # Global average pooling: 8x8x128 -> 128
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)        # (B, 128, 8, 8)
        x = self.gap(x)             # (B, 128, 1, 1)
        x = x.flatten(1)            # (B, 128)
        x = self.classifier(x)      # (B, num_classes)
        return x

# ---- Test the shape ----
model = SimpleCNN(in_channels=3, num_classes=10)
dummy = torch.randn(4, 3, 32, 32)
out   = model(dummy)
print(f"Output shape: {out.shape}")   # (4, 10)

# Count parameters
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")

# ---- Synthetic training data (32x32 RGB, 10 classes) ----
np.random.seed(42)
N = 2000
X = torch.randn(N, 3, 32, 32)
y = torch.randint(0, 10, (N,))

dataset  = TensorDataset(X, y)
n_val    = 400
n_train  = N - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)

# ---- Training ----
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    # Train
    model.train()
    train_loss = 0.0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        out  = model(X_b)
        loss = criterion(out, y_b)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validate
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            preds = model(X_b).argmax(1)
            val_correct += (preds == y_b).sum().item()
            val_total   += y_b.size(0)

    print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss/len(train_loader):.4f} | "
          f"Val Acc: {100*val_correct/val_total:.1f}%")
```

---

## 9. Receptive Field

The **receptive field** of a neuron is the area of the original input image that influences its value.

- After the first 3×3 conv layer, each output pixel sees a 3×3 region.
- After the first 2×2 pool, each output pixel sees a 4×4 region.
- After the second 3×3 conv, each output pixel sees a 6×6 region (accounting for the pool: actually 8×8 in the original).

Deeper layers have larger receptive fields and can integrate more global context. Very deep networks (like ResNet-50) have receptive fields larger than the entire image — every output neuron "sees" the whole image.

Why does this matter? A neuron needs a large enough receptive field to understand high-level structure (like a whole face) rather than just local texture.

---

## 10. Classic Architectures

Understanding classic architectures helps you design your own and understand research papers.

### LeNet-5 (1998)

The first practical CNN, designed for handwritten digit recognition (MNIST-like).

```
Input: 32×32×1 (greyscale)
Conv(6 filters, 5×5) → Pool(2×2) → Conv(16 filters, 5×5) → Pool(2×2) → Flatten → FC(120) → FC(84) → FC(10)
```

Very small by modern standards, but proved the CNN concept.

```python
# Run this in your local Python environment with PyTorch installed
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),        # 28x28 -> 24x24
            nn.Tanh(),
            nn.AvgPool2d(2, 2),                    # 24x24 -> 12x12
            nn.Conv2d(6, 16, kernel_size=5),       # 12x12 -> 8x8
            nn.Tanh(),
            nn.AvgPool2d(2, 2),                    # 8x8 -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)
```

### AlexNet (2012)

Won ImageNet with a top-5 error of 15.3% (compared to 26% for the previous winner). Key innovations:
- ReLU activations (much faster to train than tanh/sigmoid)
- Dropout for regularisation
- Data augmentation
- Multi-GPU training (split across 2 GPUs)
- 5 conv layers + 3 FC layers, 60M parameters

AlexNet showed that deep CNNs trained on GPUs can dominate computer vision.

### VGG (2014)

The insight: **depth matters more than large kernels**. VGG uses only 3×3 convolutions, stacking many of them.

```
Two 3×3 convolutions have the same receptive field as one 5×5 convolution,
but fewer parameters and an extra nonlinearity.

Two 3×3:   2 × (3×3×C×C) = 18C² parameters
One 5×5:   1 × (5×5×C×C) = 25C² parameters
→ 3×3 is cheaper!
```

VGG-16: 16 weight layers, all 3×3 conv. Simple but large (138M parameters).

### ResNet (2015)

The key problem: as networks get deeper, they get HARDER to train (vanishing gradients, degradation problem). ResNet introduces **skip connections** (residual connections):

```
Instead of learning:   output = F(x)
ResNet learns:         output = F(x) + x   ← the identity is added back
```

If F(x) is hard to learn, the network can just set F(x) ≈ 0 and pass x through unchanged. This makes very deep networks trainable.

```python
# Run this in your local Python environment with PyTorch installed
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out = out + residual    # skip connection: add the input back
        out = self.relu(out)
        return out

# Test
block = ResidualBlock(64)
x = torch.randn(8, 64, 16, 16)
out = block(x)
print(out.shape)   # torch.Size([8, 64, 16, 16]) — same shape
```

ResNet enables training networks with 50, 100, even 1000+ layers. ResNet-50 and ResNet-101 are still widely used today.

---

## 11. Data Augmentation

For real image datasets, training data is limited. Data augmentation creates variations of existing images during training, making the model more robust.

```python
# Run this in your local Python environment with PyTorch installed
import torchvision.transforms as transforms

# Training transforms: include augmentations
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),       # 50% chance of flip
    transforms.RandomCrop(32, padding=4),          # random crop with padding
    transforms.ColorJitter(brightness=0.2,         # random colour changes
                           contrast=0.2,
                           saturation=0.2),
    transforms.ToTensor(),                          # convert PIL → tensor [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats
                         std=[0.229, 0.224, 0.225]),
])

# Validation/test transforms: NO augmentation (only normalise)
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

Key principle: **augmentation only on training set**, never on validation or test.

Common augmentations:
- Horizontal flip (not vertical for most tasks)
- Random crop / random resized crop
- Colour jitter (brightness, contrast, saturation, hue)
- Random rotation (small angles)
- Cutout / RandomErasing (mask out patches)
- Gaussian blur

---

## 12. Transfer Learning

Training a CNN from scratch requires a large dataset and many hours of GPU time. **Transfer learning** uses a model already trained on a large dataset (usually ImageNet with 1.2M images) as a starting point.

### Why it Works

The early layers of a CNN learn general features (edges, textures, colours) that are useful for ANY visual task. Only the final layers learn task-specific patterns. So you can reuse the feature extractor and only train a new classifier head.

### Two Approaches

**Feature extraction**: Freeze all pretrained weights, only train the new head.

```python
# Run this in your local Python environment with PyTorch installed
import torchvision.models as models
import torch.nn as nn

# Load pretrained ResNet-18
model = models.resnet18(weights='IMAGENET1K_V1')

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer for your number of classes
num_classes = 5
model.fc = nn.Linear(model.fc.in_features, num_classes)
# model.fc is trainable (requires_grad=True by default for new layers)

# Now only model.fc will be updated during training
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
```

**Fine-tuning**: Unfreeze some or all layers and train with a small learning rate.

```python
# Run this in your local Python environment with PyTorch installed
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

model = models.resnet18(weights='IMAGENET1K_V1')

# Replace head
model.fc = nn.Linear(model.fc.in_features, 5)

# Fine-tune entire model with a small LR
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)

# Alternative: different LRs for different parts
optimizer = optim.Adam([
    {'params': model.fc.parameters(),    'lr': 1e-3},   # new head: higher LR
    {'params': list(model.parameters())[:-2], 'lr': 1e-5},  # pretrained: tiny LR
])
```

### When to Use Each

| Situation                        | Approach           |
|----------------------------------|--------------------|
| Very small dataset (<1000 imgs)  | Feature extraction |
| Medium dataset, similar to ImageNet | Fine-tune last few layers |
| Large dataset, very different domain | Fine-tune all layers |

---

## 13. Batch Normalisation in CNNs

For 2D convolution, use `nn.BatchNorm2d` (not `BatchNorm1d`):

```python
# Run this in your local Python environment with PyTorch installed
import torch.nn as nn

# BatchNorm2d normalises over (N, H, W) for each channel C
bn = nn.BatchNorm2d(num_features=64)  # 64 channels

# Placement: after conv, BEFORE activation (common convention)
block = nn.Sequential(
    nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),  # bias=False when using BN
    nn.BatchNorm2d(64),
    nn.ReLU(),
)
```

Note: when using `BatchNorm`, set `bias=False` in the preceding Conv layer — BatchNorm already includes a learnable shift parameter (beta) that acts as a bias.

---

## 14. Visualising What CNNs Learn

Understanding what your CNN has learned is important for debugging and trust.

### Activation Visualisation

```python
# Run this in your local Python environment with PyTorch installed
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

model.eval()
activations = {}

def make_hook(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

# Register hooks to capture intermediate activations
model.features[0].register_forward_hook(make_hook('conv1'))
model.features[4].register_forward_hook(make_hook('conv2'))

with torch.no_grad():
    _ = model(some_image.unsqueeze(0))

# Plot feature maps from conv1
feat = activations['conv1'][0]   # (C, H, W)
fig, axes = plt.subplots(4, 8, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    if i < feat.shape[0]:
        ax.imshow(feat[i].cpu(), cmap='viridis')
    ax.axis('off')
plt.suptitle('Conv1 Feature Maps')
plt.tight_layout()
plt.show()
```

---

## 15. IOAI: CNNs in Competition

In IOAI competitions, CNN tasks typically include:
- **Image classification**: given images, predict a class label
- **Feature extraction**: use a pretrained CNN to create embeddings for downstream tasks
- **Transfer learning**: adapt a pretrained model to a new task with limited data

### Common IOAI Pitfalls

**Wrong input shape**: PyTorch CNNs expect `(N, C, H, W)`. If your images are `(N, H, W, C)` (NumPy convention), transpose them:

```python
# From (N, H, W, C) to (N, C, H, W)
x = x.transpose(0, 3, 1, 2)  # NumPy
# or
x = torch.tensor(x).permute(0, 3, 1, 2)  # PyTorch
```

**Forgetting normalisation**: Pretrained models expect images normalised to the same statistics used during pretraining. Always apply the same `transforms.Normalize` parameters.

**Not resizing images**: Pretrained ResNet, VGG, etc. expect 224×224 inputs. Always resize:

```python
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

**Using a CPU for large images**: Always move both model AND data to GPU.

### Competition Template: Image Classification with Transfer Learning

```python
# Run this in your local Python environment with PyTorch installed
"""
IOAI CNN Classification — Transfer Learning Template
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

SEED = 42
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Transforms ----
train_tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ---- Data ----
# train_dataset = ImageFolder("path/to/train", transform=train_tfm)
# val_dataset   = ImageFolder("path/to/val",   transform=val_tfm)
# For demo: simulate with random tensors
NUM_CLASSES = 5
N_TRAIN, N_VAL = 800, 200
X_train = torch.randn(N_TRAIN, 3, 224, 224)
y_train = torch.randint(0, NUM_CLASSES, (N_TRAIN,))
X_val   = torch.randn(N_VAL, 3, 224, 224)
y_val   = torch.randint(0, NUM_CLASSES, (N_VAL,))

from torch.utils.data import TensorDataset
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=32)

# ---- Model: ResNet-18 with custom head ----
model = models.resnet18(weights='IMAGENET1K_V1')

# Optional: freeze backbone, only train head
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, NUM_CLASSES)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

best_val_acc, best_model_state = 0.0, None

for epoch in range(15):
    # Train
    model.train()
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimizer.step()
    scheduler.step()

    # Validate
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            preds = model(X_b).argmax(1)
            correct += (preds == y_b).sum().item()
            total   += y_b.size(0)
    acc = correct / total

    if acc > best_val_acc:
        best_val_acc  = acc
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"Epoch {epoch+1:2d} | Val Acc: {acc*100:.1f}%")

print(f"\nBest Val Acc: {best_val_acc*100:.1f}%")
```

---

## 16. Summary

| Concept           | Key point                                              |
|-------------------|--------------------------------------------------------|
| Convolution       | Slides a filter over the input; dot product at each position |
| Kernel / Filter   | Small weight matrix learned during training            |
| Stride            | Step size of the filter; stride 2 halves the output   |
| Padding           | Adds zeros; padding=1 with 3×3 kernel keeps same size |
| Output size       | `(input - kernel + 2*padding) / stride + 1`           |
| Feature map       | Output of one filter applied over the input            |
| Channels          | Number of filters = number of output channels          |
| Max Pooling       | Downsample by taking max in each window               |
| Receptive field   | Input region that influences one output neuron         |
| Skip connections  | Add input to output (ResNet); enables very deep nets  |
| Transfer learning | Reuse pretrained features; only retrain the head       |
| BatchNorm2d       | Normalise over (N, H, W) per channel                  |

---

## Practice Problems

1. **Output size calculation**: Compute the output size for input 64×64, kernel 5×5, stride 2, padding 2. Work through the formula by hand.

2. **Parameter counting**: How many parameters does `nn.Conv2d(16, 32, kernel_size=3, padding=1)` have? Include biases.

3. **Implement a VGG-style block**: Two 3×3 conv layers with BatchNorm and ReLU, followed by MaxPool. Make it a reusable `nn.Module`.

4. **ResNet experiment**: Add a residual block to the `SimpleCNN` above. Does validation accuracy improve?

5. **Receptive field calculation**: Compute the receptive field after:
   - Conv(3×3, stride=1)
   - MaxPool(2×2, stride=2)
   - Conv(3×3, stride=1)
   Start from a 1×1 output unit and work backwards.

6. **Transfer learning experiment**: Load a pretrained ResNet-18, freeze all layers except the last residual block and the FC layer. Train on a small dataset and compare to freezing everything.
