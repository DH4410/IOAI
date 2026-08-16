---
title: Training in PyTorch
track: nn
order: 7
estimatedTime: 45
difficulty: intermediate
---

# Training in PyTorch

PyTorch is the main tool used in IOAI competitions for neural networks. This lesson walks through the full training loop step by step. By the end you will have a template you can copy and adapt for any task.

---

## 1. Tensors: NumPy with GPU Support

A PyTorch tensor is like a NumPy array, but it can run on a GPU and supports automatic gradients.

```python
import torch
import torch.nn as nn

# Create tensors
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.zeros(3, 4)
c = torch.ones(2, 3)
d = torch.randn(100, 10)   # random normal

# Check properties (same as NumPy)
print(d.shape)   # torch.Size([100, 10])
print(d.dtype)   # torch.float32

# Move to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
d = d.to(device)
```

Convert between NumPy and PyTorch:

```python
import numpy as np

np_array = np.array([1, 2, 3])
tensor    = torch.from_numpy(np_array)
back      = tensor.numpy()
```

**Quick check:** Why might you want tensors on the GPU?
> GPU can do many operations in parallel, making training 10-100x faster for large networks.

---

## 2. Autograd: Automatic Gradients

PyTorch tracks math operations and computes gradients automatically. This is what makes backpropagation work without you writing it by hand.

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2   # y = x^2

y.backward()       # compute dy/dx
print(x.grad)      # tensor(6.) - because d/dx of x^2 is 2x, and x=3
```

You do not call `.backward()` yourself during training. The training loop does it for you.

---

## 3. Building a Model

Subclass `nn.Module` to define your network:

```python
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleNet(input_size=20, hidden_size=64, output_size=3)
model = model.to(device)
print(model)
```

`nn.Sequential` chains layers in order. `forward()` defines what happens when you pass data through.

---

## 4. Loss Functions and Optimizers

```python
# For classification (targets are class indices like 0, 1, 2)
criterion = nn.CrossEntropyLoss()

# For regression (targets are continuous numbers)
criterion = nn.MSELoss()

# Optimizer - Adam is a safe default
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

**Quick check:** You are classifying emails into 3 categories. Which loss do you use?
> `CrossEntropyLoss`. It expects raw scores (logits) from your model and class indices as labels.

---

## 5. The Training Loop

This is the core pattern. Memorize it:

```python
model.train()   # put model in training mode

for epoch in range(50):
    total_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # 1. Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # 2. Backward pass
        optimizer.zero_grad()   # clear old gradients
        loss.backward()         # compute new gradients
        optimizer.step()        # update weights

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: loss = {avg_loss:.4f}')
```

The four steps inside the loop never change: forward, zero_grad, backward, step. In that order, every time.

**Quick check:** Why do we call `optimizer.zero_grad()` before `loss.backward()`?
> PyTorch accumulates gradients by default. If you do not zero them out, gradients from the previous batch add up and corrupt the update.

---

## 6. DataLoaders

A DataLoader batches your data and shuffles it automatically:

```python
from torch.utils.data import TensorDataset, DataLoader

# Convert numpy arrays to tensors
X_tensor = torch.FloatTensor(X_train)
y_tensor  = torch.LongTensor(y_train)   # LongTensor for class labels

# Wrap in a dataset
dataset = TensorDataset(X_tensor, y_tensor)

# Create the loader
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

`batch_size=32` means the model sees 32 samples at a time. Common choices: 16, 32, 64, 128.

---

## 7. Validation

Always check how the model does on data it has not seen during training:

```python
model.eval()   # turn off dropout and batch norm (if used)

with torch.no_grad():   # do not compute gradients - saves memory and time
    val_outputs = model(X_val_tensor.to(device))
    val_loss = criterion(val_outputs, y_val_tensor.to(device))

    # For classification: get predicted class
    predicted = val_outputs.argmax(dim=1)
    accuracy = (predicted == y_val_tensor.to(device)).float().mean()
    print(f'Validation loss: {val_loss:.4f}, Accuracy: {accuracy:.2%}')
```

If training loss drops but validation loss rises, you are overfitting. Add dropout or reduce model size.

---

## 8. A Complete Working Example

```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# --- Data ---
np.random.seed(42)
X = np.random.randn(500, 10).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)  # binary classification

split = 400
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
    batch_size=32, shuffle=True
)

# --- Model ---
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- Train ---
for epoch in range(30):
    model.train()
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimizer.step()

# --- Evaluate ---
model.eval()
X_val_t = torch.from_numpy(X_val).to(device)
y_val_t  = torch.from_numpy(y_val).to(device)

with torch.no_grad():
    preds = model(X_val_t).argmax(dim=1)
    acc = (preds == y_val_t).float().mean()
    print(f'Validation accuracy: {acc:.2%}')
```

This template works for most tabular classification tasks. Change the architecture, data, and loss for other tasks.

---

## Summary

| Step | Code |
|---|---|
| Create model | Subclass `nn.Module`, use `nn.Sequential` |
| Loss (classification) | `nn.CrossEntropyLoss()` |
| Loss (regression) | `nn.MSELoss()` |
| Optimizer | `torch.optim.Adam(model.parameters(), lr=0.001)` |
| Training mode | `model.train()` |
| Forward pass | `outputs = model(X_batch)` |
| Clear gradients | `optimizer.zero_grad()` |
| Backprop | `loss.backward()` |
| Update weights | `optimizer.step()` |
| Eval mode | `model.eval()` + `torch.no_grad():` |
| Get predictions | `outputs.argmax(dim=1)` |
