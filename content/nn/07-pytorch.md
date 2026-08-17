---
title: Training in PyTorch
track: nn
order: 7
estimatedTime: 65
difficulty: intermediate
---

# Training in PyTorch

PyTorch is the framework used in IOAI competitions for neural networks. This lesson walks through the complete training loop — from tensors to a production-quality training template that handles early stopping, model saving, and GPU training.

---

## 1. Tensors: The Foundation

PyTorch tensors are like NumPy arrays but with GPU support and automatic gradient tracking:

```python
import torch
import torch.nn as nn
import numpy as np

# Creating tensors
a = torch.tensor([1.0, 2.0, 3.0])       # from list
b = torch.zeros(3, 4)                    # all zeros
c = torch.ones(2, 3)                     # all ones
d = torch.randn(100, 10)                 # random normal (mean=0, std=1)
e = torch.arange(0, 10, dtype=torch.float32)

print(d.shape)    # torch.Size([100, 10])
print(d.dtype)    # torch.float32
print(d.device)   # cpu (or cuda:0 if GPU)

# Operations (same as NumPy)
print((d * 2).mean())
print(d @ d.T)   # matrix multiply → shape [100, 100]
```

**NumPy ↔ PyTorch:**
```python
arr = np.array([1.0, 2.0, 3.0])
t   = torch.from_numpy(arr)   # shares memory
arr_back = t.numpy()          # back to NumPy (CPU only)

# Common conversion for model outputs
preds_np = model(X_tensor).detach().cpu().numpy()
```

**GPU setup** (always do this at the top of your competition code):
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using: {device}")

# Move to GPU
X_tensor = X_tensor.to(device)
model = model.to(device)
```

---

## 2. Autograd: Automatic Differentiation

PyTorch builds a computational graph as you do math. Calling `.backward()` propagates gradients through that graph:

```python
# Simple example: y = 3x² + 2x
x = torch.tensor(2.0, requires_grad=True)
y = 3 * x**2 + 2 * x

y.backward()                 # compute dy/dx
print(x.grad)                # dy/dx = 6x + 2 = 14

# Multi-variable
a = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)
c = a**2 * b + b**3

c.backward()
print(a.grad)   # dc/da = 2ab = 4
print(b.grad)   # dc/db = a² + 3b² = 13
```

Inside neural network training, PyTorch tracks every operation in `model(X)`, so `loss.backward()` can compute gradients for all parameters automatically.

---

## 3. Building Models with nn.Module

```python
class MLP(nn.Module):
    """Multi-layer perceptron for classification."""
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),     # normalizes activations per batch
                nn.ReLU(),
                nn.Dropout(dropout)    # randomly zeroes units
            ]
            prev = h
        layers.append(nn.Linear(prev, output_dim))  # output layer (no activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Instantiate
model = MLP(input_dim=20, hidden_dims=[128, 64, 32], output_dim=3, dropout=0.3)
model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,}   Trainable: {train_params:,}")
print(model)
```

---

## 4. Loss Functions

```python
# Multi-class classification: inputs are raw logits, targets are class indices
criterion = nn.CrossEntropyLoss()
logits = torch.randn(8, 3)       # batch of 8, 3 classes
labels = torch.tensor([0, 2, 1, 0, 2, 1, 0, 2])
loss = criterion(logits, labels)
print(f"CE loss: {loss:.4f}")    # mean negative log-likelihood

# Binary classification
criterion_binary = nn.BCEWithLogitsLoss()   # takes logits, NOT probabilities

# Regression
criterion_mse = nn.MSELoss()
criterion_mae = nn.L1Loss()      # L1 / MAE

# Huber loss: MSE for small errors, MAE for large (robust to outliers)
criterion_huber = nn.HuberLoss(delta=1.0)
```

---

## 5. Optimizers

```python
# Adam — safe default for most tasks
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# AdamW — Adam with decoupled weight decay; often better than Adam
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# SGD with momentum — slower but sometimes better generalization
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# Only optimize specific layers (e.g., fine-tuning)
optimizer = torch.optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-5},  # small lr for frozen layers
    {'params': model.head.parameters(),     'lr': 1e-3},  # larger lr for new layers
])
```

---

## 6. Learning Rate Schedulers

The learning rate rarely stays constant in practice — you decay it over training:

```python
from torch.optim.lr_scheduler import (
    StepLR, CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
)

# StepLR: multiply lr by gamma every step_size epochs
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)   # halve every 10 epochs

# CosineAnnealingLR: lr follows a cosine curve — often best
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# ReduceLROnPlateau: reduce lr if validation loss plateaus
scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)

# Usage in training loop:
for epoch in range(n_epochs):
    train_one_epoch(model, train_loader)
    val_loss = validate(model, val_loader)
    scheduler.step(val_loss)   # for ReduceLROnPlateau
    # scheduler.step()          # for StepLR, CosineAnnealingLR
```

---

## 7. DataLoaders

```python
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

# Simple: from NumPy arrays
X_t = torch.FloatTensor(X)
y_t = torch.LongTensor(y)
dataset = TensorDataset(X_t, y_t)

# Split
train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                 generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0)

# Custom Dataset (for image data, etc.)
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

---

## 8. The Complete Training Loop

This template includes early stopping, scheduler, and model saving — copy-paste for competitions:

```python
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from copy import deepcopy

def train_model(model, X_train, y_train, X_val, y_val,
                lr=1e-3, epochs=200, batch_size=64,
                patience=15, device='cpu'):
    """
    Full training loop with early stopping and model checkpointing.
    Returns the model with the best validation loss.
    """
    model = model.to(device)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=batch_size, shuffle=True
    )
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    best_weights  = None
    patience_count = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # ── Training ─────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
            optimizer.step()
            train_loss += loss.item() * len(X_b)
        train_loss /= len(X_train)

        # ── Validation ───────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_out   = model(X_val_t)
            val_loss  = criterion(val_out, y_val_t).item()
            val_preds = val_out.argmax(dim=1)
            val_acc   = (val_preds == y_val_t).float().mean().item()

        scheduler.step()
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if epoch % 20 == 0:
            print(f"Ep {epoch:3d} | Train {train_loss:.4f} | Val {val_loss:.4f} | Acc {val_acc:.2%}")

        # ── Early stopping ───────────────────────────────────
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_weights  = deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"Early stop at epoch {epoch} (best val loss: {best_val_loss:.4f})")
                break

    # Restore best weights
    model.load_state_dict(best_weights)
    return model, history

# Save / load model
def save_model(model, path='best_model.pt'):
    torch.save(model.state_dict(), path)

def load_model(model_class, path, *args, **kwargs):
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model
```

---

## 9. GPU / Mixed Precision Training

For competitions with GPU access, mixed precision can speed up training 2-3×:

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()  # manages gradient scaling for fp16

for X_b, y_b in train_loader:
    X_b, y_b = X_b.to(device), y_b.to(device)
    optimizer.zero_grad()

    with autocast():    # runs forward in float16 where safe
        outputs = model(X_b)
        loss    = criterion(outputs, y_b)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

---

## 10. Sort These PyTorch Steps

```widget
{
  "type": "concept-sort",
  "title": "Training Loop Order: Which Step Comes First?",
  "categories": [
    { "name": "Before backward()", "color": "#5B5BD6" },
    { "name": "After backward()", "color": "#22C55E" }
  ],
  "items": [
    { "text": "optimizer.zero_grad()", "category": "Before backward()" },
    { "text": "optimizer.step()", "category": "After backward()" },
    { "text": "loss = criterion(outputs, y)", "category": "Before backward()" },
    { "text": "loss.backward()", "category": "Before backward()" },
    { "text": "clip_grad_norm_()", "category": "After backward()" },
    { "text": "outputs = model(X_batch)", "category": "Before backward()" }
  ]
}
```

---

## 11. Debugging PyTorch

Common errors and fixes:

```python
# Error: Expected scalar type Long but found Float
y_t = y_t.long()   # CrossEntropyLoss expects Long (int64) targets

# Error: size mismatch
print(X.shape, model(X).shape, y.shape)  # check shapes at every step

# Gradient not flowing (loss is constant):
for name, param in model.named_parameters():
    if param.grad is None: print(f"No grad: {name}")

# Check for NaN loss:
if torch.isnan(loss):
    print("NaN loss! Check learning rate (too high?) or data (any inf?)")

# Make forward pass debuggable:
x = torch.randn(1, input_dim)   # dummy input
with torch.no_grad():
    out = model(x)
print(f"Input: {x.shape} → Output: {out.shape}")
```

---

## 12. IOAI PyTorch Template

```python
# ── IOAI Competition Neural Network Template ──────────────────
import torch, torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CompetitionNet(nn.Module):
    def __init__(self, inp, out, hidden=[256, 128, 64], drop=0.3):
        super().__init__()
        layers, prev = [], inp
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(drop)]
            prev = h
        layers.append(nn.Linear(prev, out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5-fold CV
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(y), n_classes))

for fold, (tr, va) in enumerate(kf.split(X_scaled, y)):
    model = CompetitionNet(inp=X.shape[1], out=n_classes).to(device)
    trained, _ = train_model(
        model, X_scaled[tr], y[tr], X_scaled[va], y[va],
        lr=3e-4, epochs=300, batch_size=256, patience=20, device=device
    )
    with torch.no_grad():
        trained.eval()
        logits = trained(torch.FloatTensor(X_scaled[va]).to(device))
        oof_preds[va] = torch.softmax(logits, dim=1).cpu().numpy()

from sklearn.metrics import accuracy_score
print(f"OOF accuracy: {accuracy_score(y, oof_preds.argmax(1)):.4f}")
```

---

## Summary

| Concept | Code |
|---|---|
| Device | `'cuda' if torch.cuda.is_available() else 'cpu'` |
| Model | Subclass `nn.Module`, define `forward()` |
| Loss | `CrossEntropyLoss` (classification), `MSELoss` (regression) |
| Optimizer | `AdamW` — Adam with decoupled weight decay |
| Scheduler | `CosineAnnealingLR` — smooth lr decay |
| Training mode | `model.train()` for training, `model.eval()` for validation |
| Loop order | zero_grad → forward → loss → backward → clip → step |
| Gradient clip | `nn.utils.clip_grad_norm_(model.parameters(), 1.0)` |
| Early stopping | Save best weights, restore when done |
| Mixed precision | `GradScaler` + `autocast()` — 2-3× faster on GPU |

The training loop is always the same skeleton — just change the model, data, and loss function.
