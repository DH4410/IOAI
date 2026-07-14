---
title: Training in PyTorch
track: nn
order: 7
estimatedTime: 60
difficulty: intermediate
---

# Training in PyTorch

PyTorch is the framework used by most top AI research labs and competitions. If you want to build and train neural networks from scratch — and understand exactly what is happening — PyTorch is the right tool. This lesson takes you from zero to training a full neural network, step by step.

> **Note on running code:** PyTorch does not run in the browser. All code in this lesson is valid PyTorch but must be run in your local Python environment. Install PyTorch at [pytorch.org](https://pytorch.org).

---

## 1. What Is PyTorch?

PyTorch is an open-source deep learning library built at Facebook AI (now Meta AI). It has two main jobs:

1. **Tensor computation** — like NumPy, but with GPU support.
2. **Automatic differentiation** — computing gradients automatically so you can train neural networks without deriving calculus by hand.

PyTorch uses a **dynamic computation graph**. This means the graph that tracks which operations were performed is built as your code runs, not before it. This makes debugging easy — you can use normal Python print statements, breakpoints, and conditional logic anywhere in your model.

Compare this to older frameworks like TensorFlow 1.x (which used static graphs compiled before execution). Dynamic graphs are more flexible and feel like ordinary Python.

```python
# Run this in your local Python environment with PyTorch installed
import torch
import torch.nn as nn
import torch.optim as optim

print(torch.__version__)  # e.g., 2.1.0
```

---

## 2. Tensors: The Core Data Structure

A **tensor** is a multi-dimensional array. It generalises:
- A **scalar** is a 0-dimensional tensor: `3.14`
- A **vector** is a 1-dimensional tensor: `[1, 2, 3]`
- A **matrix** is a 2-dimensional tensor: `[[1, 2], [3, 4]]`
- An **image batch** is a 4-dimensional tensor: shape `(batch, channels, height, width)`

### Creating Tensors

```python
# Run this in your local Python environment with PyTorch installed
import torch

# From Python list
a = torch.tensor([1.0, 2.0, 3.0])
print(a)          # tensor([1., 2., 3.])
print(a.shape)    # torch.Size([3])
print(a.dtype)    # torch.float32

# Zeros, ones, random
zeros = torch.zeros(3, 4)        # 3 rows, 4 columns of zeros
ones  = torch.ones(2, 2)         # 2x2 of ones
rand  = torch.rand(5, 5)         # uniform random in [0, 1)
randn = torch.randn(100, 10)     # standard normal: mean=0, std=1

# Like another tensor's shape
b = torch.zeros_like(a)
c = torch.ones_like(randn)

# Range (like np.arange)
r = torch.arange(0, 10, step=2)  # tensor([0, 2, 4, 6, 8])

# From NumPy (shares memory)
import numpy as np
arr = np.array([1.0, 2.0, 3.0])
t   = torch.from_numpy(arr)
```

### Tensor Operations

```python
# Run this in your local Python environment with PyTorch installed
import torch

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

# Arithmetic (element-wise)
print(x + y)       # tensor([5., 7., 9.])
print(x * y)       # tensor([ 4., 10., 18.])
print(x ** 2)      # tensor([1., 4., 9.])

# Matrix multiplication
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = A @ B          # shape: (3, 5)
# or: torch.matmul(A, B)

# Reduction
print(x.sum())     # tensor(6.)
print(x.mean())    # tensor(2.)
print(x.max())     # tensor(3.)

# Reshape
x = torch.arange(12, dtype=torch.float32)
print(x.shape)             # torch.Size([12])
x = x.reshape(3, 4)        # (3, 4)
x = x.view(2, 6)           # same data, different shape (must be contiguous)
x = x.unsqueeze(0)         # add dim at 0 -> (1, 2, 6)
x = x.squeeze(0)           # remove dim of size 1 -> (2, 6)

# Indexing (same as NumPy)
A = torch.randn(5, 5)
print(A[0])         # first row
print(A[:, 1])      # second column
print(A[1:3, 2:4])  # slice
```

### Moving to GPU

GPUs train networks much faster. You move tensors and models to GPU with `.to(device)` or `.cuda()`:

```python
# Run this in your local Python environment with PyTorch installed
import torch

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create tensor on CPU, move to GPU
x = torch.randn(100, 100)
x = x.to(device)

# Or create directly on GPU
x = torch.randn(100, 100, device=device)

# Move back to CPU (for NumPy conversion or plotting)
x_cpu = x.cpu()
```

For IOAI: most competition environments provide a GPU. Always check and use it.

---

## 3. Autograd: Automatic Differentiation

This is PyTorch's superpower. When you set `requires_grad=True` on a tensor, PyTorch records every operation done to it. When you call `.backward()` on the final result (the loss), PyTorch computes the gradient of that loss with respect to every tensor that had `requires_grad=True`.

### Simple Example

```python
# Run this in your local Python environment with PyTorch installed
import torch

# A single weight
w = torch.tensor(2.0, requires_grad=True)

# Compute something (like a loss)
loss = w ** 2 + 3 * w + 1   # f(w) = w^2 + 3w + 1
                              # f'(w) = 2w + 3
                              # At w=2: f'(2) = 7

# Backpropagate
loss.backward()

# Read the gradient
print(w.grad)    # tensor(7.)  ← correct!
```

### How It Works Internally

Every time you perform an operation on a tensor with `requires_grad=True`, PyTorch builds a tiny node in a graph recording:
- What operation was performed (`mul`, `add`, `pow`, ...)
- Which tensors were inputs
- The backward function for that operation

When you call `.backward()`, PyTorch traverses this graph backwards using the **chain rule**:

```
loss = f(g(h(x)))
dL/dx = dL/df * df/dg * dg/dh * dh/dx
```

This is exactly what you calculated by hand in the backpropagation lesson — PyTorch just does it automatically.

### Gradient Accumulation Warning

Gradients **accumulate** by default. If you call `.backward()` twice without resetting, the gradients add up:

```python
# Run this in your local Python environment with PyTorch installed
import torch

w = torch.tensor(2.0, requires_grad=True)

loss = w ** 2
loss.backward()
print(w.grad)   # tensor(4.)   ← correct

loss = w ** 2
loss.backward()
print(w.grad)   # tensor(8.)   ← WRONG! Accumulated, not replaced
```

This is why you must call `optimizer.zero_grad()` before each backward pass. More on this in the training loop section.

### Detaching from the Graph

Sometimes you want a tensor's values but not its gradient history — for example when logging or computing validation metrics:

```python
# Run this in your local Python environment with PyTorch installed
import torch

w = torch.tensor(3.0, requires_grad=True)
y = w * 2 + 1

# Detach: same value, no grad connection
y_detached = y.detach()
print(y_detached)            # tensor(7.)
print(y_detached.requires_grad)  # False

# torch.no_grad() context manager — disables tracking for a block
with torch.no_grad():
    val = w * 2 + 1
    print(val)               # tensor(7.), no graph recorded
```

---

## 4. Building Models with `nn.Module`

In PyTorch, every neural network is a Python class that inherits from `nn.Module`. You implement two methods:

- `__init__`: Define the layers (the trainable parameters live here).
- `forward`: Define how data flows through the network.

```python
# Run this in your local Python environment with PyTorch installed
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Define layers — their parameters are automatically tracked
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu   = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Define the forward pass
        x = self.layer1(x)   # linear transformation
        x = self.relu(x)     # activation
        x = self.layer2(x)   # output layer
        return x

# Instantiate
model = SimpleNet(input_size=4, hidden_size=8, output_size=3)
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

# Run a forward pass
x = torch.randn(16, 4)   # batch of 16, each with 4 features
output = model(x)         # calls model.forward(x)
print(output.shape)       # torch.Size([16, 3])
```

The call `model(x)` automatically invokes `forward(x)` — do not call `model.forward(x)` directly.

### Why `super().__init__()`?

`nn.Module` has its own `__init__` that sets up internal bookkeeping (parameter tracking, hooks, etc.). You must call it before defining any layers.

---

## 5. Built-in Layers

### `nn.Linear`

A fully connected (dense) layer. Computes `y = xW^T + b`.

```python
# Run this in your local Python environment with PyTorch installed
import torch
import torch.nn as nn

layer = nn.Linear(in_features=10, out_features=5)
print(layer.weight.shape)   # torch.Size([5, 10])
print(layer.bias.shape)     # torch.Size([5])

x = torch.randn(32, 10)     # batch of 32
out = layer(x)
print(out.shape)             # torch.Size([32, 5])
```

### `nn.ReLU`

Activation function: `ReLU(x) = max(0, x)`.

```python
relu = nn.ReLU()
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(relu(x))  # tensor([0., 0., 0., 1., 2.])
```

There is also `nn.Sigmoid`, `nn.Tanh`, `nn.GELU`, `nn.LeakyReLU`, `nn.ELU`.

### `nn.Sequential`

A container that stacks layers in order. Useful for simple architectures:

```python
# Run this in your local Python environment with PyTorch installed
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Equivalent to the custom class approach, but less flexible
```

Use `nn.Sequential` for simple forward pipelines. Use the class approach when you need branching, skip connections, or multiple inputs/outputs.

### `nn.Dropout`

Randomly zeros out activations during training to prevent overfitting.

```python
import torch.nn as nn

dropout = nn.Dropout(p=0.5)   # 50% of activations set to 0

# During training: randomly drops activations (and scales rest by 1/(1-p))
# During eval (model.eval()): does nothing, acts as identity
```

Important: dropout is **only active during training**. Call `model.eval()` before validation/inference.

### `nn.BatchNorm1d`

Normalises each feature across the batch to have mean ~0 and std ~1. Helps training stability and speed.

```python
import torch
import torch.nn as nn

bn = nn.BatchNorm1d(num_features=128)

# During training: normalises using batch statistics
# During eval: uses running mean/variance accumulated during training

class BetterNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)     # normalise before activation
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

---

## 6. Loss Functions

The loss function measures how wrong your model's predictions are. Choosing the right one is critical.

### `nn.MSELoss` — Mean Squared Error

Use for **regression** (predicting a continuous value).

```
MSE = (1/n) * sum((y_pred - y_true)^2)
```

```python
# Run this in your local Python environment with PyTorch installed
import torch
import torch.nn as nn

criterion = nn.MSELoss()

y_pred = torch.tensor([2.5, 0.0, 2.0, 8.0])
y_true = torch.tensor([3.0, 0.5, 2.0, 7.5])

loss = criterion(y_pred, y_true)
print(loss)   # tensor(0.1875)
```

### `nn.CrossEntropyLoss` — Cross Entropy

Use for **multi-class classification**. Expects raw logits (no softmax), targets as class indices.

```
CrossEntropy = -log(softmax(logits)[true_class])
```

```python
# Run this in your local Python environment with PyTorch installed
import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

# logits: raw output from last linear layer (NOT softmaxed)
logits = torch.tensor([[2.0, 1.0, 0.5],   # sample 1 favors class 0
                        [0.5, 2.5, 0.3]])  # sample 2 favors class 1

targets = torch.tensor([0, 1])  # true class indices

loss = criterion(logits, targets)
print(loss)   # small, because the model is right
```

**Common mistake:** Applying `nn.Softmax` before `nn.CrossEntropyLoss`. Don't. The loss already includes softmax internally (it uses `log_softmax` for numerical stability).

### `nn.BCELoss` and `nn.BCEWithLogitsLoss`

Use for **binary classification** (output is a single probability).

- `nn.BCELoss` expects inputs already in `[0, 1]` (after sigmoid).
- `nn.BCEWithLogitsLoss` expects raw logits — **prefer this** for numerical stability.

```python
# Run this in your local Python environment with PyTorch installed
import torch
import torch.nn as nn

criterion = nn.BCEWithLogitsLoss()

logits  = torch.tensor([2.0, -1.0, 0.5, -0.3])  # raw scores
targets = torch.tensor([1.0,  0.0, 1.0,  0.0])   # binary labels

loss = criterion(logits, targets)
print(loss)
```

### Summary Table

| Task                  | Output layer            | Loss function             |
|-----------------------|-------------------------|---------------------------|
| Regression            | `nn.Linear` (1 output)  | `nn.MSELoss`              |
| Binary classification | `nn.Linear` (1 output)  | `nn.BCEWithLogitsLoss`    |
| Multi-class           | `nn.Linear` (C outputs) | `nn.CrossEntropyLoss`     |

---

## 7. Optimizers

The optimizer updates model parameters using the computed gradients.

### `torch.optim.SGD`

Stochastic Gradient Descent (with optional momentum):

```
w = w - lr * grad_w
```

```python
# Run this in your local Python environment with PyTorch installed
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

`momentum=0.9` keeps a running average of gradients, which smooths out noisy updates and helps escape shallow local minima.

### `torch.optim.Adam`

Adaptive Moment Estimation. Adjusts the learning rate for each parameter individually based on the history of its gradients.

```python
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
```

Adam is the default choice for most problems. It is less sensitive to the learning rate than SGD.

### `torch.optim.AdamW`

Like Adam but with proper weight decay (L2 regularisation). Preferred over Adam in modern practice.

```python
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

### The Update Cycle

Every training step follows this exact sequence:

```python
optimizer.zero_grad()   # 1. Clear old gradients
output = model(x)       # 2. Forward pass
loss = criterion(output, y)  # 3. Compute loss
loss.backward()         # 4. Compute gradients
optimizer.step()        # 5. Update parameters
```

Never skip step 1. Never swap the order.

---

## 8. The Full Training Loop

Here is the complete training loop with comments explaining every line:

```python
# Run this in your local Python environment with PyTorch installed
import torch
import torch.nn as nn
import torch.optim as optim

# ---- Model definition ----
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

# ---- Setup ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model     = MLP(input_size=10, hidden_size=64, output_size=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---- Synthetic data ----
N = 1000  # number of samples
X = torch.randn(N, 10)
y = torch.randint(0, 3, (N,))   # random class labels 0, 1, 2

dataset = torch.utils.data.TensorDataset(X, y)
loader  = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# ---- Training loop ----
num_epochs = 20

for epoch in range(num_epochs):
    model.train()          # set to training mode (enables dropout, batchnorm train behavior)
    running_loss = 0.0

    for batch_X, batch_y in loader:
        # Move data to device
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        # Step 1: Zero out old gradients
        optimizer.zero_grad()

        # Step 2: Forward pass
        outputs = model(batch_X)         # shape: (batch_size, 3)

        # Step 3: Compute loss
        loss = criterion(outputs, batch_y)

        # Step 4: Backward pass — compute gradients
        loss.backward()

        # Step 5: Update weights
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    print(f"Epoch {epoch+1:2d}/{num_epochs} | Loss: {avg_loss:.4f}")
```

---

## 9. Datasets and DataLoaders

### `torch.utils.data.Dataset`

A Dataset is an object that provides samples one at a time. You subclass it and implement `__len__` and `__getitem__`.

```python
# Run this in your local Python environment with PyTorch installed
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X   # feature tensor
        self.y = y   # label tensor

    def __len__(self):
        # How many samples are there?
        return len(self.X)

    def __getitem__(self, idx):
        # Return the idx-th sample
        return self.X[idx], self.y[idx]

# Usage
X = torch.randn(500, 20)
y = torch.randint(0, 5, (500,))
dataset = MyDataset(X, y)
print(len(dataset))       # 500
print(dataset[0])         # (tensor of shape [20], scalar)
```

For simple cases where X and y are already tensors, use `TensorDataset` directly:

```python
from torch.utils.data import TensorDataset
dataset = TensorDataset(X, y)
```

### `DataLoader`

The DataLoader wraps a Dataset and yields **batches** of data. It handles:
- **Batching**: grouping samples into batches
- **Shuffling**: randomising order each epoch
- **Parallel loading**: using multiple worker processes

```python
# Run this in your local Python environment with PyTorch installed
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=64,     # samples per batch
    shuffle=True,      # shuffle at the start of each epoch
    num_workers=2,     # parallel data loading (use 0 on Windows if issues occur)
    drop_last=False    # whether to drop the last incomplete batch
)

for batch_X, batch_y in loader:
    print(batch_X.shape)  # (64, 20) — except maybe last batch
    break
```

### Train/Validation Split

```python
# Run this in your local Python environment with PyTorch installed
from torch.utils.data import random_split

# Split 80% train, 20% validation
train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False)
```

---

## 10. The Validation Loop

After each epoch, evaluate the model on validation data. This is how you detect overfitting.

```python
# Run this in your local Python environment with PyTorch installed
def evaluate(model, loader, criterion, device):
    model.eval()     # switch to eval mode: disables dropout, uses running stats for BN
    total_loss   = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():    # disable gradient computation (saves memory, speeds up)
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X)
            loss    = criterion(outputs, batch_y)

            total_loss += loss.item()

            # Compute accuracy
            preds = outputs.argmax(dim=1)         # class with highest logit
            total_correct  += (preds == batch_y).sum().item()
            total_samples  += batch_y.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy
```

### Full Training + Validation Loop

```python
# Run this in your local Python environment with PyTorch installed
for epoch in range(num_epochs):
    # ---- Training ----
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # ---- Validation ----
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    print(f"Epoch {epoch+1:2d} | "
          f"Train Loss: {train_loss/len(train_loader):.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_acc*100:.1f}%")
```

Key signs:
- **Train loss falls, val loss also falls**: good.
- **Train loss falls, val loss rises**: overfitting. Add dropout, reduce model size, or get more data.
- **Both stay high**: underfitting. Increase model size or training time.

---

## 11. Saving and Loading Models

### Save Just the Parameters (`state_dict`)

This is the recommended approach. It saves only the weights, not the model architecture.

```python
# Run this in your local Python environment with PyTorch installed

# --- Saving ---
torch.save(model.state_dict(), "model_weights.pth")

# --- Loading ---
model = MLP(input_size=10, hidden_size=64, output_size=3)  # re-create the architecture
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.eval()
```

### Save the Entire Model

Saves the architecture and weights together. Less portable (tied to the exact class definition).

```python
# Save
torch.save(model, "full_model.pth")

# Load
model = torch.load("full_model.pth", map_location=device)
model.eval()
```

### Checkpointing During Training

Save the best model based on validation loss:

```python
# Run this in your local Python environment with PyTorch installed
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # ... training ...
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  Saved new best model (val_loss={val_loss:.4f})")
```

---

## 12. Complete Example 1: XOR Problem

XOR cannot be solved by a linear model (a single neuron). A neural network with one hidden layer can. This is the classic proof that hidden layers give power.

| x1 | x2 | XOR |
|----|----|-----|
| 0  | 0  |  0  |
| 0  | 1  |  1  |
| 1  | 0  |  1  |
| 1  | 1  |  0  |

```python
# Run this in your local Python environment with PyTorch installed
import torch
import torch.nn as nn
import torch.optim as optim

# ---- Data ----
X = torch.tensor([[0., 0.],
                   [0., 1.],
                   [1., 0.],
                   [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])   # binary targets

# ---- Model ----
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    # No sigmoid here — BCEWithLogitsLoss handles it
)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# ---- Training ----
for epoch in range(1000):
    optimizer.zero_grad()
    output = model(X)
    loss   = criterion(output, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        preds = (torch.sigmoid(output) > 0.5).float()
        acc = (preds == y).float().mean()
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Acc: {acc.item():.2f}")

# ---- Evaluate ----
model.eval()
with torch.no_grad():
    logits = model(X)
    preds  = (torch.sigmoid(logits) > 0.5).float()
    print("\nFinal predictions:")
    for i in range(4):
        print(f"  XOR({int(X[i,0])}, {int(X[i,1])}) = {int(preds[i,0])} "
              f"(true: {int(y[i,0])})")
```

---

## 13. Complete Example 2: Multi-class Classification

A realistic example: synthetic 2D data with 4 classes.

```python
# Run this in your local Python environment with PyTorch installed
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np

# ---- Generate synthetic data ----
np.random.seed(42)
torch.manual_seed(42)

def make_multiclass_data(n=1000, n_classes=4, n_features=8):
    X_list, y_list = [], []
    for c in range(n_classes):
        center = np.random.randn(n_features) * 3
        X_c    = np.random.randn(n // n_classes, n_features) + center
        y_c    = np.full(n // n_classes, c)
        X_list.append(X_c)
        y_list.append(y_c)
    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.int64)
    return X, y

X_np, y_np = make_multiclass_data(n=1200, n_classes=4, n_features=8)
X_tensor = torch.from_numpy(X_np)
y_tensor = torch.from_numpy(y_np)

# ---- Dataset & Loaders ----
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False)

# ---- Model ----
class Classifier(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x)

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = Classifier(n_features=8, n_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# ---- Training loop ----
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        out  = model(X_b)
        loss = criterion(out, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            out  = model(X_b)
            loss = criterion(out, y_b)
            total_loss += loss.item()
            preds   = out.argmax(dim=1)
            correct += (preds == y_b).sum().item()
            total   += y_b.size(0)
    return total_loss / len(loader), correct / total

# ---- Run training ----
best_val_loss = float('inf')

for epoch in range(30):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_classifier.pth")

    print(f"Epoch {epoch+1:2d} | "
          f"Train: {train_loss:.4f} | "
          f"Val: {val_loss:.4f} | "
          f"Acc: {val_acc*100:.1f}%")

print(f"\nBest val loss: {best_val_loss:.4f}")
```

---

## 14. Learning Rate Schedulers

A fixed learning rate is rarely optimal. Schedulers reduce the learning rate during training.

```python
# Run this in your local Python environment with PyTorch installed
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Reduce LR by factor 0.1 every 10 epochs
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Cosine annealing: LR decreases following a cosine curve
scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

# Reduce on plateau: reduce LR when val_loss stops improving
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

for epoch in range(num_epochs):
    train_epoch(...)
    val_loss, _ = eval_epoch(...)

    # For most schedulers:
    scheduler.step()

    # For ReduceLROnPlateau, pass the metric:
    # scheduler.step(val_loss)
```

---

## 15. Common Mistakes and How to Avoid Them

### Mistake 1: Forgetting `optimizer.zero_grad()`

```python
# WRONG
for X_b, y_b in loader:
    out  = model(X_b)
    loss = criterion(out, y_b)
    loss.backward()   # gradients accumulate!
    optimizer.step()

# CORRECT
for X_b, y_b in loader:
    optimizer.zero_grad()   # ← always first
    out  = model(X_b)
    loss = criterion(out, y_b)
    loss.backward()
    optimizer.step()
```

### Mistake 2: Wrong loss function

```python
# WRONG: Softmax + CrossEntropyLoss (double softmax)
model = nn.Sequential(nn.Linear(10, 5), nn.Softmax(dim=1))
criterion = nn.CrossEntropyLoss()   # already does softmax internally

# CORRECT: Raw logits + CrossEntropyLoss
model = nn.Sequential(nn.Linear(10, 5))   # no softmax
criterion = nn.CrossEntropyLoss()

# CORRECT: Softmax + NLLLoss
import torch.nn.functional as F
model = nn.Sequential(nn.Linear(10, 5), nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
```

### Mistake 3: Not calling `.backward()`

```python
# WRONG: Skip backward — no gradients, no learning
for X_b, y_b in loader:
    optimizer.zero_grad()
    out  = model(X_b)
    loss = criterion(out, y_b)
    # loss.backward()   ← forgot!
    optimizer.step()   # step with zero gradients — model doesn't change

# Always include loss.backward()!
```

### Mistake 4: Not moving data to the same device as the model

```python
# WRONG: Model on GPU, data on CPU
model = model.to("cuda")
for X_b, y_b in loader:
    # X_b and y_b are on CPU — will crash!
    out = model(X_b)

# CORRECT
for X_b, y_b in loader:
    X_b = X_b.to(device)
    y_b = y_b.to(device)
    out = model(X_b)
```

### Mistake 5: Forgetting `model.eval()` during validation

```python
# WRONG: Dropout is still active during validation → inconsistent results
model.train()   # left in train mode
val_loss, val_acc = evaluate(model, val_loader, ...)

# CORRECT
model.eval()    # always switch mode before evaluating
with torch.no_grad():
    ...
```

### Mistake 6: Shape mismatches

Always check tensor shapes. Use `print(x.shape)` aggressively.

```python
# Cross-entropy expects (N, C) logits and (N,) targets
logits  = model(X_b)            # shape: (32, 5)
targets = y_b                   # shape: (32,)  ← correct: 1D class indices
loss    = criterion(logits, targets)

# WRONG: targets as (32, 1) or one-hot (32, 5) will cause errors
```

---

## 16. IOAI Tips: Structuring Your Training Code

In IOAI competitions, you typically receive a dataset and must train a model and submit predictions. Here is a clean, submission-ready template:

```python
# Run this in your local Python environment with PyTorch installed
"""
IOAI Submission Template - Classification Task
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np

# ==============================================================================
# CONFIG — edit these
# ==============================================================================
SEED         = 42
BATCH_SIZE   = 64
NUM_EPOCHS   = 50
LR           = 1e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT    = 0.15
MODEL_PATH   = "submission_model.pth"

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# MODEL — define your architecture here
# ==============================================================================
class Model(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_output)
        )

    def forward(self, x):
        return self.net(x)

# ==============================================================================
# DATA — load your actual data here
# ==============================================================================
# X_train = ...   # numpy array, shape (N, features)
# y_train = ...   # numpy array, shape (N,)
# X_test  = ...   # numpy array, shape (M, features)

# For demonstration: synthetic data
X_train = np.random.randn(1000, 20).astype(np.float32)
y_train = np.random.randint(0, 5, 1000).astype(np.int64)
X_test  = np.random.randn(200, 20).astype(np.float32)

X_t = torch.from_numpy(X_train)
y_t = torch.from_numpy(y_train)
X_test_t = torch.from_numpy(X_test)

dataset  = TensorDataset(X_t, y_t)
n_val    = int(VAL_SPLIT * len(dataset))
n_train  = len(dataset) - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

# ==============================================================================
# TRAINING
# ==============================================================================
n_classes = len(np.unique(y_train))
model     = Model(n_input=X_train.shape[1], n_output=n_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

best_val_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    # Train
    model.train()
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        out  = model(X_b)
        loss = criterion(out, y_b)
        loss.backward()
        optimizer.step()
    scheduler.step()

    # Validate
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            out  = model(X_b)
            val_loss += criterion(out, y_b).item()
            correct  += (out.argmax(1) == y_b).sum().item()
            total    += y_b.size(0)
    val_loss /= len(val_loader)
    val_acc   = correct / total

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_PATH)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:2d} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}%")

# ==============================================================================
# INFERENCE — generate predictions on test set
# ==============================================================================
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

with torch.no_grad():
    test_logits = model(X_test_t.to(device))
    predictions = test_logits.argmax(dim=1).cpu().numpy()

print(f"\nPredictions shape: {predictions.shape}")
print(f"Sample predictions: {predictions[:10]}")

# Save predictions
np.save("predictions.npy", predictions)
print("Saved predictions.npy")
```

### IOAI Checklist

Before submitting:
- [ ] Set a random seed (`torch.manual_seed`, `np.random.seed`)
- [ ] Move both model and data to device
- [ ] Use validation set to pick the best checkpoint (not the last epoch)
- [ ] Load the best checkpoint before inference
- [ ] Call `model.eval()` and `torch.no_grad()` during inference
- [ ] Confirm output format matches the competition spec (class index, probability, etc.)
- [ ] Check your predictions are not all the same class (common bug)

---

## 17. Summary

| Concept          | What it does                              | Key method/class            |
|------------------|-------------------------------------------|-----------------------------|
| Tensor           | N-dimensional array with GPU support      | `torch.tensor()`, `.to()`   |
| Autograd         | Automatic gradient computation            | `requires_grad`, `.backward()` |
| `nn.Module`      | Base class for all models                 | `__init__`, `forward`       |
| `nn.Linear`      | Fully connected layer                     | `nn.Linear(in, out)`        |
| Loss function    | Measures prediction error                 | `nn.CrossEntropyLoss`, etc. |
| Optimizer        | Updates weights using gradients           | `optim.Adam`, `.step()`     |
| `DataLoader`     | Batches and shuffles data                 | `DataLoader(dataset, ...)`  |
| `model.train()`  | Enables dropout, batch norm train mode    | Call before training        |
| `model.eval()`   | Disables dropout, uses fixed BN stats     | Call before validation      |
| `torch.no_grad()`| Disables gradient tracking               | Use in eval loops           |
| `state_dict`     | Dictionary of all model parameters       | Save/load with `torch.save` |

---

## Practice Problems

1. **Modify the XOR example** to use a deeper network (3 hidden layers). Does it still converge?

2. **Add early stopping** to the multi-class example: stop training if val loss doesn't improve for 5 epochs.

3. **Implement L1 regularisation** manually: after computing the loss, add `lambda * sum(|w|)` for all weight tensors before calling `.backward()`.

4. **Experiment with learning rates**: train the classifier with lr=0.1, 0.01, 0.001, 0.0001. Plot the validation loss curves. Which is best?

5. **Make a regression model**: generate 1D data `y = sin(x) + noise`, train a network to predict y from x. Use `nn.MSELoss`.
