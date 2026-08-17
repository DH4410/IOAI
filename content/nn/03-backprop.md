---
title: Backpropagation
track: nn
order: 3
estimatedTime: 55
difficulty: advanced
---

# Backpropagation

We can now build a network and push data through it (the forward pass). But we've been *hand-picking* weights. The real question — the one that makes deep learning actually work — is:

> **How do we automatically figure out which weights to change, and by how much, so the network gets better?**

The answer is **backpropagation** ("backprop" for short). It's the algorithm that made neural networks trainable, and it is nothing more than the **chain rule of calculus** applied cleverly. This is the most important lesson in the track. Take it slowly; it's worth it.

---

## See the network first

Before diving into the math, watch how a forward pass moves through a network. Click "Animate" to see activations flow layer by layer:

```widget
{
  "type": "neuron-diagram",
  "title": "Forward pass — signals flow from input to output",
  "layers": [3, 4, 4, 1],
  "labels": ["Input", "Hidden 1", "Hidden 2", "Output"]
}
```

Backpropagation is the **reverse** of this — gradients flow from output back to input, one layer at a time.

---

## The core problem: assigning blame

Imagine your network makes a wrong prediction. The final error came from *thousands* of weights all contributing a little. To improve, you need to know, for **each individual weight**: *"If I nudge you up a tiny bit, does the error go up or down, and how strongly?"*

That quantity — how much the loss changes when you tweak a weight — is the **gradient** of the loss with respect to that weight, written $\frac{\partial L}{\partial w}$. If we know it for every weight, we can nudge each one in the direction that lowers the loss. That's **gradient descent**:

$$w \leftarrow w - \eta \frac{\partial L}{\partial w}$$

*In English:* move each weight a small step ($\eta$) in the **opposite** direction of its gradient, because the gradient points uphill (toward higher loss) and we want to go downhill.

So the entire game is: **compute $\frac{\partial L}{\partial w}$ for every weight.** For a network with millions of weights, doing this naively (one weight at a time) would be impossibly slow. Backpropagation computes *all* of them in a single efficient backward sweep. That efficiency is the whole point.

---

## First, the loss function

Before we can reduce error, we need to *measure* it with a **loss function** $L$ — a single number saying how wrong the network is. Lower is better.

A common one for regression is **Mean Squared Error (MSE)**:

$$L = \frac{1}{2}(\hat{y} - y)^2$$

*In English:* take the difference between prediction $\hat{y}$ and truth $y$, square it (so it's always positive and big mistakes hurt more), and halve it (the $\frac12$ is a convenience that makes the derivative clean).

Its derivative with respect to the prediction is beautifully simple:

$$\frac{\partial L}{\partial \hat{y}} = \hat{y} - y$$

*In English:* the loss's sensitivity to the prediction is just "how far off we are." The $\frac{1}{2}$ and the square cancel neatly, which is exactly why we put the $\frac12$ there. This number — "prediction minus truth" — is the very first thing backprop computes, and it's where the backward sweep begins.

---

## The chain rule, without the pain

Backprop is the chain rule. So let's build genuine intuition for the chain rule, gently.

The chain rule answers: *if $a$ affects $b$, and $b$ affects $c$, how much does $a$ affect $c$?* The answer is you **multiply the sensitivities**:

$$\frac{\partial c}{\partial a} = \frac{\partial c}{\partial b} \cdot \frac{\partial b}{\partial a}$$

**A gears analogy.** Imagine three gears. Gear $a$ turns gear $b$, which turns gear $c$. If turning $a$ by one tooth turns $b$ by 3 teeth, and turning $b$ by one tooth turns $c$ by 2 teeth, then turning $a$ by one tooth turns $c$ by $3 \times 2 = 6$ teeth. **You multiply the ratios along the chain.** That's the chain rule.

**A money analogy.** 1 dollar = 3 widgets, 1 widget = 2 gadgets. So 1 dollar = 6 gadgets. Conversion rates multiply along the chain.

A neural network is a long chain: weights affect pre-activations, which affect activations, which affect the next layer's pre-activations, ..., which affect the loss. To find how a weight deep inside affects the final loss, we **multiply all the little sensitivities along the path from that weight to the loss.** Backprop is a bookkeeping method for doing all these multiplications without redundant work.

For longer chains it just keeps going:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}$$

Each factor is a *local* derivative — easy to compute for one simple operation. Backprop chains them.

---

## The computational graph

The clean way to think about backprop is the **computational graph**: draw the network's computation as a graph where each node is a simple operation.

Consider a tiny example: $L = (wx + b - y)^2 / 2$. Break it into atomic steps:

```
   w ──┐
       ×──► s ──┐
   x ──┘         +──► z ──┐
                b ─┘        (−)──► d ──► square/2 ──► L
                          y ─┘
```

- $s = w \cdot x$
- $z = s + b$
- $d = z - y$
- $L = \frac12 d^2$

**Forward pass:** compute left to right, storing every intermediate value ($s, z, d, L$). *We must remember these* — the backward pass needs them.

**Backward pass:** start at $L$ with $\frac{\partial L}{\partial L} = 1$, then walk **right to left**, and at each node multiply by that node's local derivative. Each edge carries a gradient. By the time we reach $w$, we've multiplied all the local derivatives along the path — the chain rule, done mechanically.

This graph view is exactly how PyTorch and TensorFlow think internally. Every operation knows how to (1) compute its output forward and (2) compute its local derivative backward. Chain them and you can differentiate *anything*.

---

## Forward pass and backward pass for a 2-layer net

Let's derive backprop concretely for the network from Lesson 2: input $\mathbf{x}$ → hidden layer ($W_1, \mathbf{b}_1$, activation $\sigma$) → output layer ($W_2, \mathbf{b}_2$) → loss $L$.

**Forward pass** (compute and remember every intermediate):

$$\mathbf{z}_1 = W_1\mathbf{x} + \mathbf{b}_1$$
$$\mathbf{h} = \sigma(\mathbf{z}_1)$$
$$\mathbf{z}_2 = W_2\mathbf{h} + \mathbf{b}_2$$
$$\hat{\mathbf{y}} = \mathbf{z}_2 \quad(\text{linear output, for regression})$$
$$L = \tfrac{1}{2}\lVert \hat{\mathbf{y}} - \mathbf{y}\rVert^2$$

**Backward pass** (compute gradients right to left). We'll use the shorthand $\delta$ ("delta") for the gradient of the loss with respect to a pre-activation — this is the key reusable quantity.

**Step 1 — gradient at the output:**
$$\delta_2 = \frac{\partial L}{\partial \mathbf{z}_2} = \hat{\mathbf{y}} - \mathbf{y}$$
*In English:* since the output is linear, the loss's sensitivity to the output pre-activation is just "prediction minus truth."

**Step 2 — gradients of the output layer's parameters:**
$$\frac{\partial L}{\partial W_2} = \delta_2 \, \mathbf{h}^\top \qquad \frac{\partial L}{\partial \mathbf{b}_2} = \delta_2$$
*In English:* how much a weight $W_2$ matters equals its incoming activation $\mathbf{h}$ times the error signal $\delta_2$ flowing back into that output. A weight is "to blame" in proportion to how active its input was and how wrong its output was. The bias gradient is just $\delta_2$ (its "input" is a constant 1).

**Step 3 — propagate the error back to the hidden layer:**
$$\delta_1 = (W_2^\top \delta_2) \odot \sigma'(\mathbf{z}_1)$$
*In English:* push the error backward through the output weights ($W_2^\top \delta_2$ — this is the "propagate" in backpropagation), then multiply elementwise ($\odot$) by the activation's slope $\sigma'(\mathbf{z}_1)$. The slope matters because if a neuron was in a flat region of its activation (slope ≈ 0), changing it barely affects anything, so its error is scaled down.

**Step 4 — gradients of the hidden layer's parameters:**
$$\frac{\partial L}{\partial W_1} = \delta_1 \, \mathbf{x}^\top \qquad \frac{\partial L}{\partial \mathbf{b}_1} = \delta_1$$
Same pattern as Step 2, one layer down.

**Notice the repeating structure:** at each layer we (a) have an incoming error $\delta$, (b) use it to compute parameter gradients as $\delta \times (\text{layer input})^\top$, and (c) pass a new error backward as $W^\top\delta \odot \sigma'$. This pattern repeats for *any* number of layers. That regularity is why backprop scales to 1000-layer networks — it's the same three moves, over and over.

For sigmoid specifically, the slope is $\sigma'(z) = \sigma(z)(1-\sigma(z)) = h(1-h)$, which is why we saved $h$.

---

## Full numpy backprop implementation

Now let's actually **train** the XOR network from scratch — forward pass, backward pass, weight update, repeat. No PyTorch, no autograd, every gradient computed by hand. This is the code that "does deep learning" in ~40 lines.

```python
import numpy as np

rng = np.random.default_rng(1)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_deriv(a):
    # a is sigmoid(z); derivative is a*(1-a)
    return a * (1.0 - a)

# ---- XOR data (batched: 4 examples, 2 features) ----
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)   # (4, 2)
Y = np.array([[0],[1],[1],[0]], dtype=float)           # (4, 1)

# ---- Initialize weights (small random) and biases (zero) ----
W1 = rng.normal(0, 1.0, size=(2, 4))   # 2 inputs -> 4 hidden
b1 = np.zeros((1, 4))
W2 = rng.normal(0, 1.0, size=(4, 1))   # 4 hidden -> 1 output
b2 = np.zeros((1, 1))

lr = 0.5
N = X.shape[0]

for epoch in range(5000):
    # ---------- FORWARD PASS ----------
    Z1 = X @ W1 + b1        # (4, 4)
    H  = sigmoid(Z1)        # (4, 4)
    Z2 = H @ W2 + b2        # (4, 1)
    Yhat = sigmoid(Z2)      # (4, 1)  output in (0,1)

    loss = np.mean((Yhat - Y) ** 2)

    # ---------- BACKWARD PASS ----------
    # d loss / d Yhat for MSE, then through the output sigmoid:
    dYhat = (2.0 / N) * (Yhat - Y)          # (4,1)
    delta2 = dYhat * sigmoid_deriv(Yhat)    # (4,1)  error at output pre-activation

    dW2 = H.T @ delta2                       # (4,1)  gradient for W2
    db2 = delta2.sum(axis=0, keepdims=True)  # (1,1)

    # propagate error back to hidden layer
    delta1 = (delta2 @ W2.T) * sigmoid_deriv(H)   # (4,4)
    dW1 = X.T @ delta1                        # (2,4)  gradient for W1
    db1 = delta1.sum(axis=0, keepdims=True)  # (1,4)

    # ---------- GRADIENT DESCENT UPDATE ----------
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if epoch % 1000 == 0:
        print(f"epoch {epoch:5d} | loss = {loss:.4f}")

# ---- Final predictions ----
Z1 = sigmoid(X @ W1 + b1)
Yhat = sigmoid(Z1 @ W2 + b2)
print("\nFinal XOR predictions:")
for x, p in zip(X, Yhat.ravel()):
    print(f"  {x} -> {p:.3f}  (rounded {round(p)})")
```

Run it. The loss falls from ~0.25 toward ~0, and the final predictions snap to XOR: `[0,0]→0, [0,1]→1, [1,0]→1, [1,1]→0`. **You just trained a neural network from scratch.** No library did the calculus for you — you computed every gradient by hand and the network learned. This exact loop, scaled up, is how every deep model is trained.

**Match the code to the math:**

| Code | Math | Meaning |
|---|---|---|
| `delta2 = dYhat * sigmoid_deriv(Yhat)` | $\delta_2$ | error at output pre-activation |
| `dW2 = H.T @ delta2` | $\partial L/\partial W_2 = \delta_2 \mathbf{h}^\top$ | output weight gradient |
| `delta1 = (delta2 @ W2.T) * sigmoid_deriv(H)` | $\delta_1 = (W_2^\top\delta_2)\odot\sigma'$ | error pushed back to hidden |
| `dW1 = X.T @ delta1` | $\partial L/\partial W_1 = \delta_1\mathbf{x}^\top$ | hidden weight gradient |
| `W -= lr * dW` | $w \leftarrow w - \eta\,\partial L/\partial w$ | gradient descent |

The transposes (`.T`) are just about lining up the batch dimension so the sums over examples happen correctly. If you ever get confused, check shapes — they must match.

---

## Why PyTorch autograd exists

Deriving those gradient formulas by hand was doable for 2 layers. Now imagine a 50-layer network with convolutions, attention, batch norm, and skip connections. Deriving and coding every gradient by hand would be a nightmare — and one algebra slip means silent, wrong training.

**Autograd** (automatic differentiation) solves this. Frameworks like **PyTorch** build the computational graph *as you run the forward pass*, recording every operation. Then a single call — `loss.backward()` — walks that graph backward and computes every gradient automatically, using exactly the chain-rule bookkeeping we did by hand.

```python
# The PyTorch version of everything above is essentially:
#   yhat = model(x)
#   loss = loss_fn(yhat, y)
#   loss.backward()      # autograd computes ALL gradients for you
#   optimizer.step()     # updates all weights
```

So why did we do it by hand? Because **autograd feels like magic until you've done it manually once.** When training goes wrong — exploding loss, NaNs, a layer that won't learn — understanding what `.backward()` actually computes is what lets you diagnose it. You'll meet PyTorch properly in Lesson 7; you now know what it's doing under the hood.

---

## Common backprop bugs (and how to avoid them)

These are the bugs that bite everyone, including at IOAI. Recognizing them saves hours.

### 1. Forgetting to zero the gradients

In PyTorch, gradients **accumulate** (add up) across `.backward()` calls by default. If you don't reset them each step, you're summing gradients from all previous batches — training goes haywire. The fix:

```python
optimizer.zero_grad()   # MUST call before each backward pass!
loss.backward()
optimizer.step()
```

*Why does accumulation exist at all?* It's occasionally useful (e.g. simulating a large batch by summing several small ones). But 99% of the time you want a fresh gradient each step, so **`zero_grad()` is mandatory** in the loop. Forgetting it is the single most common beginner bug.

### 2. Exploding gradients

If gradients get multiplied through many layers and each factor is bigger than 1, the product blows up exponentially — weights become huge, then `NaN` (not-a-number), and training dies. Symptoms: loss suddenly jumps to `inf` or `nan`. Fixes:

- **Smaller learning rate.**
- **Gradient clipping** — cap the gradient norm at some maximum.
- **Better weight initialization** (Xavier/He init) and **normalization** (BatchNorm, Lesson 6).

### 3. Vanishing gradients

The opposite problem: if each factor is smaller than 1, the product shrinks toward zero through many layers, so early layers get almost no gradient and barely learn. Sigmoid's derivative maxes out at 0.25, so deep sigmoid networks vanish badly. This is a big reason **ReLU** (Lesson 4) took over — its gradient is exactly 1 for positive inputs, so it doesn't shrink.

### 4. Not saving forward-pass values

The backward pass *needs* the intermediates ($\mathbf{h}$, $\mathbf{z}_1$, etc.) from the forward pass. If you overwrite them or recompute wrong, gradients are wrong. In our numpy code we carefully kept `H`; autograd stores these for you automatically.

### 5. Shape mismatches / wrong transposes

A gradient must have the **same shape** as the thing it's the gradient of ($\partial L/\partial W_1$ has the shape of $W_1$). If it doesn't, a transpose is misplaced. Checking `dW.shape == W.shape` catches many bugs instantly.

> **Debugging mantra:** if the loss isn't decreasing, check in this order: (1) did you `zero_grad()`? (2) is the learning rate sane? (3) do all gradient shapes match their parameters? (4) is the loss even connected to the parameters (did you accidentally detach something)?

---

## Gradient checking: verifying backprop numerically

How do you *know* your hand-coded gradient is correct? Compare it to a **numerical gradient** estimated by the definition of a derivative:

$$\frac{\partial L}{\partial w} \approx \frac{L(w + \epsilon) - L(w - \epsilon)}{2\epsilon}$$

*In English:* nudge one weight up by a tiny $\epsilon$, measure the loss; nudge it down, measure again; the gradient is the change in loss divided by the total nudge. This is slow (one forward pass per weight) but it's a ground-truth check.

```python
import numpy as np

def loss_fn(w, x=2.0, y=1.0):
    # tiny model: yhat = w*x, MSE loss
    yhat = w * x
    return 0.5 * (yhat - y) ** 2

w = 0.7
eps = 1e-5
# Numerical gradient (central difference)
num_grad = (loss_fn(w + eps) - loss_fn(w - eps)) / (2 * eps)

# Analytic gradient: dL/dw = (w*x - y) * x
x, y = 2.0, 1.0
analytic_grad = (w * x - y) * x

print(f"numerical : {num_grad:.6f}")
print(f"analytic  : {analytic_grad:.6f}")
print(f"difference: {abs(num_grad - analytic_grad):.2e}")
```

If the two agree to ~6 decimal places, your backprop is almost certainly right. This is a genuinely useful trick when a custom layer won't train — verify the gradient before blaming everything else.

---

## Why this matters for IOAI

- **Backprop is the beating heart of training.** Every model you train at IOAI is trained by it. Even when PyTorch hides it, knowing what `.backward()` computes is what separates people who can debug training from people who just pray.
- **The bug list is a checklist.** Vanishing/exploding gradients, forgotten `zero_grad()`, and NaN losses are *routine* at competitions. Recognizing them from symptoms turns a lost afternoon into a two-minute fix.
- **Gradient intuition guides architecture choices.** Why ReLU over sigmoid? Why residual connections? Why normalization? All are answers to "keep gradients flowing." Those choices (Lessons 4, 6, 8) only make sense once you understand backprop.
- **Numerical gradient checking is a real tool.** If you implement a custom operation at IOAI, checking its gradient numerically can save you from silently-wrong results.

---

## Summary

- Training means computing $\frac{\partial L}{\partial w}$ for **every** weight, then doing gradient descent $w \leftarrow w - \eta\frac{\partial L}{\partial w}$.
- **Backpropagation** computes all these gradients efficiently in one backward sweep, using the **chain rule** (multiply local sensitivities along the path).
- A network is a **computational graph**; the forward pass stores intermediates, the backward pass walks right-to-left multiplying local derivatives.
- The reusable quantity is $\delta$ (error at a pre-activation); at each layer: parameter gradient $= \delta \times \text{input}^\top$, and back-propagated error $= W^\top\delta \odot \sigma'$.
- **Autograd** (PyTorch) automates this exact process so you don't derive gradients by hand.
- Watch for the classic bugs: forgetting **`zero_grad()`**, **exploding** and **vanishing** gradients, and shape mismatches.
- **Numerical gradient checking** verifies a hand-coded gradient against the definition of the derivative.
