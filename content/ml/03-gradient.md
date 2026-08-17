---
title: Gradient Descent
track: ml
order: 3
estimatedTime: 35
difficulty: intermediate
---

# Gradient Descent

The closed-form solution for linear regression works, but most models (neural networks, logistic regression) do not have a formula you can just compute. Instead, they learn by taking small steps in the direction that reduces the loss. This process is called gradient descent.

---

## 1. The Mountain Analogy

Imagine you are on a mountain in thick fog. You cannot see the valley below. Your only information is: "which way is downhill from where I stand right now?"

Gradient descent does this for a loss function. At each step:
1. Look at the slope (gradient) of the loss at the current position
2. Take one small step in the direction that reduces the loss (downhill)
3. Repeat until the loss stops decreasing

The **gradient** is a vector that points in the direction of steepest increase. To decrease the loss, you move in the opposite direction.

---

## 2. The Math

For a single weight `w`, the update rule is:

```
w = w - learning_rate * (d Loss / d w)
```

The `d Loss / d w` part is the derivative of the loss with respect to `w`. For MSE loss, this works out to:

```
d MSE / d w = (2/n) * X^T * (Xw - y)
```

You do not need to derive this by hand. You need to understand the update rule.

**Quick check:** The gradient at your current position is 5.0 and the learning rate is 0.1. By how much does the weight change?
> `0.1 * 5.0 = 0.5`. The weight decreases by 0.5.

---

```widget
{
  "type": "gradient-slider",
  "title": "See gradient descent move toward the minimum"
}
```

Try these experiments:
- Set learning rate to **0.90** — watch it overshoot and bounce
- Set it to **0.05** — watch it creep slowly
- Set it to **0.30** — the sweet spot

---

## 3. Implementation from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt

# Make some data
np.random.seed(42)
X = np.random.randn(200, 1)
y = 3 * X.squeeze() + 2 + np.random.randn(200) * 0.5

# Initialize weights
w = 0.0
b = 0.0
lr = 0.05
n = len(X)

loss_history = []

for epoch in range(100):
    # Forward pass
    y_pred = X.squeeze() * w + b

    # Loss (MSE)
    loss = ((y_pred - y) ** 2).mean()
    loss_history.append(loss)

    # Gradients
    dw = (2 / n) * (X.squeeze() * (y_pred - y)).sum()
    db = (2 / n) * (y_pred - y).sum()

    # Update
    w -= lr * dw
    b -= lr * db

print(f'w = {w:.4f}  (true: 3.0)')
print(f'b = {b:.4f}  (true: 2.0)')

plt.plot(loss_history)
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.show()
```

Watch the loss curve. It should go down quickly at first, then flatten out. If it goes up or bounces wildly, your learning rate is too high.

---

## 4. The Learning Rate

The learning rate is the most important setting (hyperparameter) in gradient descent.

**Too high:** Steps are so big you jump over the minimum — loss bounces or goes up.

**Too low:** Steps are tiny — takes forever to converge.

**Just right:** Loss decreases smoothly and flattens out.

```widget
{
  "type": "overfit-curve",
  "title": "How model complexity affects train vs validation accuracy"
}
```

This isn't directly the learning rate — but notice the same shape: too simple = underfit, too complex = overfit. The "just right" zone in the middle is what you're always aiming for.

```python
# Compare learning rates
for lr in [0.5, 0.1, 0.01]:
    w, b = 0.0, 0.0
    losses = []
    for _ in range(50):
        y_pred = X.squeeze() * w + b
        loss = ((y_pred - y) ** 2).mean()
        losses.append(loss)
        dw = (2/n) * (X.squeeze() * (y_pred - y)).sum()
        db = (2/n) * (y_pred - y).sum()
        w -= lr * dw
        b -= lr * db
    plt.plot(losses, label=f'lr={lr}')

plt.legend()
plt.title('Effect of learning rate')
plt.show()
```

**Quick check:** After 50 epochs with lr=0.5, your loss is oscillating between 1.2 and 2.0. What should you try?
> Lower the learning rate. The steps are too large and you are jumping around the minimum.

---

## 5. Batch vs. Mini-Batch vs. SGD

In practice you do not compute the gradient on the full dataset at once. You use a subset.

**Full batch (Gradient Descent):** Use all data to compute one gradient. Stable but slow for large datasets.

**Mini-batch:** Use a small random batch (32-256 samples) for each gradient update. Fast and good estimates. This is what you use in practice.

**SGD (Stochastic Gradient Descent):** Use one sample at a time. Very noisy but can escape local optima.

```python
def gradient_descent_minibatch(X, y, lr=0.05, epochs=50, batch_size=32):
    w, b = 0.0, 0.0
    n = len(X)
    X_sq = X.squeeze()

    for epoch in range(epochs):
        # Shuffle data at the start of each epoch
        idx = np.random.permutation(n)
        X_shuf, y_shuf = X_sq[idx], y[idx]

        for i in range(0, n, batch_size):
            X_batch = X_shuf[i:i+batch_size]
            y_batch = y_shuf[i:i+batch_size]

            y_pred = X_batch * w + b
            dw = (2 / len(X_batch)) * (X_batch * (y_pred - y_batch)).sum()
            db = (2 / len(X_batch)) * (y_pred - y_batch).sum()
            w -= lr * dw
            b -= lr * db

    return w, b

w, b = gradient_descent_minibatch(X, y)
print(f'w={w:.3f}, b={b:.3f}')
```

Shuffling before each epoch is important. Without it, the model sees the same batches in the same order every time.

---

## 6. Convergence

Training stops when the loss is low enough or when it stops improving. In practice:

```python
prev_loss = float('inf')
tolerance = 1e-6

for epoch in range(1000):
    # ... compute loss ...
    if abs(prev_loss - loss) < tolerance:
        print(f'Converged at epoch {epoch}')
        break
    prev_loss = loss
```

In deep learning you usually just set a fixed number of epochs and optionally use early stopping (stop when validation loss has not improved for N epochs).

---

## Sort the Gradient Descent Variants

```widget
{
  "type": "concept-sort",
  "title": "Full Batch, Mini-Batch, or SGD?",
  "categories": [
    { "name": "Full Batch GD", "color": "#5B5BD6" },
    { "name": "Mini-Batch GD", "color": "#22C55E" },
    { "name": "SGD (1 sample)", "color": "#F97316" }
  ],
  "items": [
    { "text": "Most stable gradient estimate", "category": "Full Batch GD" },
    { "text": "Standard in modern deep learning (batch_size=32-256)", "category": "Mini-Batch GD" },
    { "text": "Very noisy updates, one sample at a time", "category": "SGD (1 sample)" },
    { "text": "Too slow for large datasets", "category": "Full Batch GD" },
    { "text": "Best balance of speed and stability", "category": "Mini-Batch GD" },
    { "text": "Can escape local optima more easily", "category": "SGD (1 sample)" }
  ]
}
```

---

## Summary

| Concept | Key point |
|---|---|
| Gradient | Direction of steepest increase. Go the opposite way to decrease loss. |
| Update rule | `w = w - lr * gradient` |
| Learning rate | Too high = unstable. Too low = very slow. Try 0.01 or 0.001 first. |
| Mini-batch | Compute gradient on a batch (32-256), not the full dataset |
| Convergence | When the loss stops decreasing significantly |

Every neural network you will ever train uses a variant of this. The core idea is always the same: measure the error, compute how each weight contributed, update each weight to reduce the error. Repeat.
