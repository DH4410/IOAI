---
title: Gradient Descent
track: ml
order: 3
estimatedTime: 50
difficulty: intermediate
---

# Gradient Descent

Gradient descent is the engine that drives almost all of modern machine learning and deep learning. The Normal Equation we saw in Lesson 2 only works for linear regression. Gradient descent is a general optimizer that works for any differentiable loss function — from simple linear models to 100-billion-parameter language models.

---

## The Mountain Analogy

Imagine you are standing in a mountain range covered in thick fog. You cannot see where the valley is, but you can feel the slope under your feet. The best strategy to find the lowest point (the valley) is:

1. Look around in all directions
2. Take a step in the direction that goes **most steeply downward**
3. Repeat until you stop descending

This is gradient descent. The "mountain landscape" is the loss function surface. The "valley" is the minimum — where the loss is smallest and our model performs best.

Gradient descent does this mathematically by computing the **gradient** — a vector of partial derivatives — that tells us which direction is "uphill," then stepping in the **opposite** direction.

---

## What is a Derivative?

For a function f(x), the derivative f'(x) tells you how fast f changes as x changes.

- If f'(x) > 0: f is increasing at x (moving right increases f)
- If f'(x) < 0: f is decreasing at x (moving right decreases f)
- If f'(x) = 0: f is flat at x (a local minimum, maximum, or saddle point)

The gradient (∇L) is the multi-dimensional version: a vector of partial derivatives, one per parameter.

```python
import numpy as np

# Numerical derivative — useful for verification
def numerical_gradient(f, x, h=1e-5):
    """
    Approximate the gradient of f at point x using finite differences.
    Works for scalar x or numpy array x.
    """
    grad = np.zeros_like(x, dtype=float)
    x = x.astype(float)
    for i in range(len(x)):
        x_plus  = x.copy(); x_plus[i]  += h
        x_minus = x.copy(); x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

# Example: f(x, y) = x² + 2y²
# True gradient: [2x, 4y]
def f(params):
    x, y = params
    return x**2 + 2*y**2

point = np.array([3.0, 1.0])
numerical_grad = numerical_gradient(f, point)
true_grad      = np.array([2 * point[0], 4 * point[1]])

print(f"Function value at (3, 1): {f(point)}")
print(f"Numerical gradient: {numerical_grad}")
print(f"True gradient:      {true_grad}")
print(f"Match: {np.allclose(numerical_grad, true_grad, atol=1e-4)}")
```

---

## The Gradient Descent Update Rule

The update rule is simple:

```
θ = θ - α * ∇L(θ)
```

Where:
- `θ` (theta) are the model parameters (weights + bias)
- `α` (alpha) is the **learning rate** — how big a step to take
- `∇L(θ)` is the gradient of the loss at the current parameters
- The minus sign means we step in the opposite direction of the gradient (downhill)

For linear regression with MSE loss:

```
L(w, b) = (1/n) Σ(yᵢ - (wᵀxᵢ + b))²
```

The gradients are:

```
∂L/∂w = -(2/n) Xᵀ(y - ŷ)
∂L/∂b = -(2/n) Σ(yᵢ - ŷᵢ)
```

---

## Implementing Gradient Descent for Linear Regression

```python
import numpy as np

class LinearRegressionGD:
    """Linear regression trained with gradient descent."""

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n, p = X.shape
        # Initialize weights to zero
        self.weights = np.zeros(p)
        self.bias    = 0.0

        for i in range(self.n_iter):
            # Forward pass: compute predictions
            y_pred = X @ self.weights + self.bias

            # Compute loss (MSE)
            loss = np.mean((y - y_pred) ** 2)
            self.loss_history.append(loss)

            # Compute gradients
            error     = y_pred - y            # shape: (n,)
            grad_w    = (2 / n) * X.T @ error # shape: (p,)
            grad_b    = (2 / n) * error.sum() # scalar

            # Update parameters
            self.weights -= self.lr * grad_w
            self.bias    -= self.lr * grad_b

        return self

    def predict(self, X):
        return X @ self.weights + self.bias


# ── Test on 1D linear data ────────────────────────────────────────
np.random.seed(42)
n = 100
x = np.random.uniform(0, 10, n)
y = 2.5 * x + 4.0 + np.random.randn(n) * 1.5

X = x.reshape(-1, 1)

# Standardize features for faster convergence
X_mean, X_std = X.mean(), X.std()
X_scaled = (X - X_mean) / X_std

model = LinearRegressionGD(learning_rate=0.1, n_iterations=500)
model.fit(X_scaled, y)

print(f"Final MSE: {model.loss_history[-1]:.4f}")
print(f"Learned weights: {model.weights}")
print(f"Learned bias:    {model.bias:.4f}")

# Show convergence
print("\nLoss at key iterations:")
checkpoints = [0, 10, 50, 100, 200, 499]
for it in checkpoints:
    print(f"  Iteration {it:>4}: loss = {model.loss_history[it]:.6f}")
```

---

## The Learning Rate: The Most Important Hyperparameter

The learning rate α controls how large each step is. Getting it right is crucial.

### Too Small: Converges but takes forever

```python
import numpy as np

# Simple 1D problem: minimize f(x) = x²
# True minimum at x = 0, gradient = 2x

def optimize_1d(start, lr, n_steps):
    x = start
    history = [x]
    for _ in range(n_steps):
        gradient = 2 * x   # df/dx = 2x
        x = x - lr * gradient
        history.append(x)
    return history

# Try different learning rates
start = 10.0
n_steps = 50

print(f"{'LR':>8} | {'x after 10 steps':>18} | {'x after 50 steps':>18}")
print("-" * 52)
for lr in [0.001, 0.01, 0.1, 0.5, 0.9, 1.1]:
    history = optimize_1d(start, lr, n_steps)
    print(f"{lr:>8.3f} | {history[10]:>18.4f} | {history[-1]:>18.4f}")
```

### Too Large: Diverges (bounces around or explodes)

```python
import numpy as np

def gradient_descent_trace(lr, n_steps=20):
    """Show the path of gradient descent on f(w) = w²."""
    w = 10.0
    trace = [w]
    for _ in range(n_steps):
        w = w - lr * (2 * w)   # gradient of w² is 2w
        trace.append(w)
        if abs(w) > 1e6:       # diverged
            break
    return trace

print("Learning rate effects on minimizing f(w) = w²:")
print(f"  True minimum: w = 0\n")

for lr, label in [(0.001, "too small"), (0.1, "good"), (0.99, "borderline"), (1.05, "diverges")]:
    trace = gradient_descent_trace(lr)
    final = trace[-1]
    print(f"  lr={lr:.3f} ({label:>12}): final w = {final:>12.4f}")
```

### Finding a Good Learning Rate

A useful heuristic: start at 0.01 and watch the loss curve. If loss decreases smoothly, good. If it oscillates or increases, decrease lr. If it decreases but very slowly, increase lr.

---

## Batch vs. Stochastic vs. Mini-Batch Gradient Descent

The three variants differ in **how much data** they use to compute each gradient update.

### Batch Gradient Descent (BGD)

Uses the **entire training set** to compute one gradient update.

```
Pros: Smooth, stable loss curve. Exact gradient.
Cons: Slow — must process all n examples before updating weights.
```

### Stochastic Gradient Descent (SGD)

Uses **one random example** per update.

```
Pros: Very fast updates. Can escape local minima due to noise.
Cons: Very noisy — loss bounces around a lot.
```

### Mini-Batch Gradient Descent

Uses a **small batch** (typically 32–512 examples) per update.

```
Pros: Balance of speed and stability. Works well with modern hardware (GPUs).
Cons: Introduces a hyperparameter (batch size).
```

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n = 1000
X = np.random.randn(n, 3)
y = 2*X[:,0] - 1*X[:,1] + 0.5*X[:,2] + 3 + np.random.randn(n) * 0.5

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def mse_loss(X, y, w, b):
    pred = X @ w + b
    return np.mean((y - pred) ** 2)

def compute_grad(X_batch, y_batch, w, b):
    n_b   = len(y_batch)
    pred  = X_batch @ w + b
    error = pred - y_batch
    grad_w = (2 / n_b) * X_batch.T @ error
    grad_b = (2 / n_b) * error.sum()
    return grad_w, grad_b

def train(X, y, batch_size, lr=0.05, n_epochs=10):
    """Train with given batch size. batch_size=n → BGD, batch_size=1 → SGD."""
    n, p = X.shape
    w = np.zeros(p)
    b = 0.0
    losses = []

    for epoch in range(n_epochs):
        # Shuffle data each epoch
        idx = np.random.permutation(n)
        X_shuf, y_shuf = X[idx], y[idx]

        epoch_losses = []
        for start in range(0, n, batch_size):
            X_b = X_shuf[start:start+batch_size]
            y_b = y_shuf[start:start+batch_size]
            gw, gb = compute_grad(X_b, y_b, w, b)
            w -= lr * gw
            b -= lr * gb
            epoch_losses.append(mse_loss(X, y, w, b))

        losses.append(np.mean(epoch_losses))

    return w, b, losses

print("Comparing gradient descent variants:")
print(f"{'Variant':>15} {'batch':>6} {'updates/epoch':>14} {'final loss':>12}")
print("-" * 52)

for name, bs in [("BGD", n), ("Mini-batch(32)", 32), ("SGD", 1)]:
    updates_per_epoch = n // bs
    w, b, losses = train(X_scaled, y, batch_size=bs)
    print(f"{name:>15} {bs:>6} {updates_per_epoch:>14} {losses[-1]:>12.6f}")
```

---

## Full Implementation: Tracking the Optimization Path

Let's visualize how gradient descent moves through the loss landscape for a 2D problem:

```python
import numpy as np

def gd_path_2d(lr=0.1, n_steps=30):
    """
    Minimize f(w1, w2) = w1² + 10*w2²
    (elongated bowl — demonstrates why scaling matters)
    Gradient: [2*w1, 20*w2]
    True minimum: (0, 0)
    """
    w = np.array([8.0, 8.0])
    path = [w.copy()]

    for _ in range(n_steps):
        grad = np.array([2 * w[0], 20 * w[1]])
        w    = w - lr * grad
        path.append(w.copy())

    return np.array(path)

print("Optimization path on f(w1,w2) = w1² + 10*w2²")
print("(Unscaled features — elongated loss surface)\n")

path = gd_path_2d(lr=0.09, n_steps=40)
print(f"{'Step':>6} {'w1':>10} {'w2':>10} {'f(w)':>12}")
print("-" * 42)
for i in [0, 1, 2, 5, 10, 20, 39]:
    w1, w2 = path[i]
    fval = w1**2 + 10*w2**2
    print(f"{i:>6} {w1:>10.4f} {w2:>10.4f} {fval:>12.6f}")

print("\nWith scaled features (f(w1,w2) = w1² + w2²):")

def gd_path_scaled(lr=0.5, n_steps=20):
    w = np.array([8.0, 8.0])
    path = [w.copy()]
    for _ in range(n_steps):
        grad = np.array([2*w[0], 2*w[1]])  # symmetric bowl
        w    = w - lr * grad
        path.append(w.copy())
    return np.array(path)

path_s = gd_path_scaled()
print(f"\n{'Step':>6} {'w1':>10} {'w2':>10} {'f(w)':>12}")
print("-" * 42)
for i in [0, 1, 2, 5, 10, 19]:
    w1, w2 = path_s[i]
    fval = w1**2 + w2**2
    print(f"{i:>6} {w1:>10.4f} {w2:>10.4f} {fval:>12.6f}")
```

---

## Convergence

Gradient descent is said to have **converged** when the loss stops decreasing significantly. In practice, you set a stopping condition:

1. **Fixed number of iterations** (simplest — common in practice)
2. **Loss threshold**: stop when |loss_t - loss_{t-1}| < ε
3. **Gradient norm threshold**: stop when ||∇L|| < ε

```python
import numpy as np

class GradientDescentWithConvergence:
    """Linear regression GD with convergence detection."""

    def __init__(self, lr=0.01, max_iter=10000, tol=1e-6):
        self.lr       = lr
        self.max_iter = max_iter
        self.tol      = tol

    def fit(self, X, y):
        n, p = X.shape
        w = np.zeros(p)
        b = 0.0
        self.losses = []
        self.n_iter_run = 0

        for i in range(self.max_iter):
            pred  = X @ w + b
            error = pred - y
            loss  = np.mean(error ** 2)
            self.losses.append(loss)

            grad_w = (2/n) * X.T @ error
            grad_b = (2/n) * error.sum()

            # Check convergence
            if i > 0 and abs(self.losses[-2] - loss) < self.tol:
                self.n_iter_run = i + 1
                print(f"Converged at iteration {i+1} (loss change < {self.tol})")
                break

            w -= self.lr * grad_w
            b -= self.lr * grad_b
        else:
            self.n_iter_run = self.max_iter
            print(f"Reached max iterations ({self.max_iter})")

        self.weights = w
        self.bias    = b
        return self

# Test
np.random.seed(42)
X = np.random.randn(200, 2)
y = 3*X[:,0] - 1.5*X[:,1] + 2 + np.random.randn(200) * 0.3

model = GradientDescentWithConvergence(lr=0.1, max_iter=10000, tol=1e-7)
model.fit(X, y)

print(f"Iterations run: {model.n_iter_run}")
print(f"Weights:        {model.weights}")
print(f"Bias:           {model.bias:.4f}")
```

---

## Saddle Points and Why They Matter

A **saddle point** is a point where the gradient is zero but it is neither a minimum nor a maximum — it curves upward in some directions and downward in others (like a horse saddle).

For convex functions (like MSE for linear regression), there are **no saddle points** — only a single global minimum.

For non-convex functions (like neural networks), saddle points are common. SGD's noise often helps escape them.

```python
import numpy as np

# f(w1, w2) = w1² - w2²  has a saddle point at (0, 0)
# Gradient: [2*w1, -2*w2]

def f_saddle(w):
    return w[0]**2 - w[1]**2

def grad_saddle(w):
    return np.array([2*w[0], -2*w[1]])

print("Gradient descent near a saddle point:")
print(f"  f(w1, w2) = w1² - w2²")
print(f"  Saddle point at (0, 0)\n")

# Start near the saddle point but not exactly on it
w = np.array([0.1, 0.1])
print(f"{'Step':>6} {'w1':>10} {'w2':>10} {'f(w)':>12}")
print("-" * 42)

for i in range(20):
    if i % 4 == 0:
        print(f"{i:>6} {w[0]:>10.4f} {w[1]:>10.4f} {f_saddle(w):>12.6f}")
    w = w - 0.1 * grad_saddle(w)

print(f"\nFinal point: {w}  ← escaped via w2 direction (function was increasing)")
```

---

## Momentum: Accelerating Gradient Descent

Momentum is a simple but powerful improvement. Instead of only using the current gradient, we also remember the direction we were moving before.

```
velocity = β * velocity - α * gradient
θ = θ + velocity
```

Think of it like a ball rolling downhill: it picks up speed and carries through small bumps.

```python
import numpy as np

class MomentumGD:
    """Gradient descent with momentum."""

    def __init__(self, lr=0.01, momentum=0.9, n_iter=500):
        self.lr       = lr
        self.momentum = momentum
        self.n_iter   = n_iter

    def fit(self, X, y):
        n, p = X.shape
        w    = np.zeros(p)
        b    = 0.0
        vw   = np.zeros(p)   # velocity for weights
        vb   = 0.0            # velocity for bias
        self.losses = []

        for _ in range(self.n_iter):
            pred  = X @ w + b
            error = pred - y
            self.losses.append(np.mean(error**2))

            gw = (2/n) * X.T @ error
            gb = (2/n) * error.sum()

            # Update velocities (momentum)
            vw = self.momentum * vw - self.lr * gw
            vb = self.momentum * vb - self.lr * gb

            # Update parameters
            w += vw
            b += vb

        self.weights = w
        self.bias    = b
        return self


# Compare standard GD vs. Momentum
np.random.seed(42)
X = np.random.randn(200, 5)
y = X @ np.array([2, -1, 0.5, 1.5, -2]) + 3 + np.random.randn(200) * 0.5

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

# Standard GD
gd = LinearRegressionGD(learning_rate=0.05, n_iterations=200)
gd.fit(X_s, y)

# Momentum GD
mom = MomentumGD(lr=0.05, momentum=0.9, n_iter=200)
mom.fit(X_s, y)

print("Standard GD loss at iterations 10, 50, 100, 200:")
for it in [9, 49, 99, 199]:
    print(f"  iter {it+1:>4}: {gd.loss_history[it]:.6f}")

print("\nMomentum GD loss at iterations 10, 50, 100, 200:")
for it in [9, 49, 99, 199]:
    print(f"  iter {it+1:>4}: {mom.losses[it]:.6f}")
```

---

## Why Deep Learning Uses SGD

The Normal Equation requires inverting an `(n_features × n_features)` matrix. For a neural network with 1 million parameters, that matrix would be 1M × 1M — completely impossible to store or invert.

SGD's key advantages for deep learning:
1. **Memory efficient** — process one mini-batch at a time
2. **Scales to any model size** — only needs gradients, not the Hessian
3. **Noise helps** — the stochastic nature helps escape local minima and saddle points
4. **Fast updates** — processes thousands of batches per second on a GPU

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(42)
X, y = make_regression(n_samples=5000, n_features=20, noise=10, random_state=42)

scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

def sgd_linear_regression(X_train, y_train, lr=0.01, batch_size=32, n_epochs=20):
    n, p   = X_train.shape
    w      = np.zeros(p)
    b      = 0.0
    losses = []

    for epoch in range(n_epochs):
        idx   = np.random.permutation(n)
        X_shuf = X_train[idx]
        y_shuf = y_train[idx]
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n, batch_size):
            Xb = X_shuf[start:start+batch_size]
            yb = y_shuf[start:start+batch_size]
            nb = len(yb)

            pred   = Xb @ w + b
            error  = pred - yb
            gw     = (2/nb) * Xb.T @ error
            gb     = (2/nb) * error.sum()
            w     -= lr * gw
            b     -= lr * gb

            epoch_loss += np.mean(error**2)
            n_batches  += 1

        losses.append(epoch_loss / n_batches)

    return w, b, losses

w, b, losses = sgd_linear_regression(X_train, y_train, lr=0.05, batch_size=32, n_epochs=30)

# Evaluate
y_pred  = X_test @ w + b
test_mse = np.mean((y_test - y_pred)**2)
from sklearn.metrics import r2_score
r2       = r2_score(y_test, y_pred)

print(f"Mini-batch SGD (batch=32) on 5000 examples with 20 features:")
print(f"  Final train loss: {losses[-1]:.2f}")
print(f"  Test MSE:  {test_mse:.2f}")
print(f"  Test R²:   {r2:.4f}")

print("\nLoss across epochs:")
for ep in [0, 4, 9, 19, 29]:
    print(f"  Epoch {ep+1:>3}: {losses[ep]:.4f}")
```

---

## Learning Rate Schedules

In practice, you often start with a larger learning rate and decay it over time. This gives fast initial progress and fine-tuning at the end.

```python
import numpy as np

def lr_schedule(initial_lr, epoch, schedule='step'):
    """Learning rate schedule examples."""
    if schedule == 'step':
        # Halve every 10 epochs
        return initial_lr * (0.5 ** (epoch // 10))
    elif schedule == 'exponential':
        return initial_lr * np.exp(-0.05 * epoch)
    elif schedule == 'cosine':
        n_epochs = 50
        return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / n_epochs))
    elif schedule == 'constant':
        return initial_lr
    return initial_lr

initial_lr = 0.1
print(f"{'Epoch':>6} {'constant':>10} {'step-decay':>10} {'exponential':>12} {'cosine':>10}")
print("-" * 54)
for epoch in [0, 5, 10, 20, 30, 40, 49]:
    c   = lr_schedule(initial_lr, epoch, 'constant')
    s   = lr_schedule(initial_lr, epoch, 'step')
    e   = lr_schedule(initial_lr, epoch, 'exponential')
    cos = lr_schedule(initial_lr, epoch, 'cosine')
    print(f"{epoch:>6} {c:>10.4f} {s:>10.4f} {e:>12.4f} {cos:>10.4f}")
```

---

## Summary

| Concept | Key Idea |
|---|---|
| Gradient | Direction of steepest ascent in the loss landscape |
| Gradient descent | Repeatedly step in the negative gradient direction |
| Learning rate α | Step size — too small = slow, too large = diverge |
| Batch GD | Use all data per update — stable but slow |
| Stochastic GD | Use 1 example per update — fast but noisy |
| Mini-batch GD | Use batch of 32-512 — best of both worlds |
| Momentum | Add velocity to escape flat regions faster |
| Convergence | Loss stops decreasing significantly |
| Saddle points | Gradient = 0 but not a minimum — SGD noise helps escape |

Gradient descent is the foundation of all modern ML optimization. Next, we apply everything learned so far to **logistic regression** — our first classification algorithm.
