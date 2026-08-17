---
title: Multi-Layer Perceptrons
track: nn
order: 2
estimatedTime: 50
difficulty: intermediate
---

# Multi-Layer Perceptrons

In the last lesson we hit a wall: a single perceptron can only draw a straight line, so it can't solve XOR. The escape was to **stack** perceptrons into layers. That stack is the **Multi-Layer Perceptron (MLP)** — also called a **fully-connected** or **dense** neural network — and it is the workhorse that every other architecture (CNNs, transformers) is built around.

By the end of this lesson you'll understand exactly how data flows through an MLP, why it can represent almost any function, and you'll build a working 2-layer network in pure numpy.

> **Big picture:** An MLP is layers of neurons where every neuron in one layer connects to every neuron in the next. Each layer transforms its input into a new, richer representation. Depth lets simple pieces combine into complex behavior.

---

## From one neuron to a network

Recall a single neuron computes $\sigma(\mathbf{w}\cdot\mathbf{x}+b)$ — one weighted sum, one activation, one output number.

Now put **several** neurons side by side, all looking at the same input $\mathbf{x}$ but each with its **own** weights. That's a **layer**. If we have 4 neurons in a layer, the layer turns an input vector into 4 output numbers — one per neuron.

Then take those 4 numbers and feed them as the input to *another* layer. And maybe another. Each layer's outputs become the next layer's inputs. That chaining is what "multi-layer" means.

```
   input        hidden layer         output layer
   x1  ─┐      ┌── h1 ──┐
        ├──────┤        ├─────►  ŷ
   x2  ─┘      ├── h2 ──┤
               └── h3 ──┘
  (2 inputs)   (3 hidden neurons)   (1 output)
```

Every arrow is a weight. "Fully connected" means every input connects to every hidden neuron, and every hidden neuron connects to every output. That's a *lot* of arrows — which is exactly why we'll switch to matrix math to keep track of them all.

---

## Architecture: input, hidden, and output layers

An MLP has three kinds of layers:

- **Input layer** — not really "neurons," just the raw feature values $\mathbf{x}$. If each example has 784 numbers (like a 28×28 image flattened), the input layer has 784 slots.
- **Hidden layer(s)** — the middle layers that do the actual transforming. "Hidden" because their outputs aren't the final answer and aren't the raw input; they're intermediate representations. You choose how many hidden layers and how many neurons each has. These are **hyperparameters**.
- **Output layer** — produces the final prediction. Its size matches the task: 1 neuron for binary yes/no, 10 neurons for classifying digits 0–9, etc.

We describe an MLP's shape by listing the sizes: e.g. **784 → 128 → 64 → 10** means 784 inputs, two hidden layers of 128 and 64 neurons, and 10 outputs.

**Terminology note:** the number of neurons in a layer is its **width**; the number of layers is the **depth**. A network is "deep" when it has many layers — hence *deep learning*.

---

## The universal approximation theorem (intuition)

Here's a jaw-dropping mathematical fact:

> **Universal Approximation Theorem:** An MLP with a *single hidden layer* containing *enough* neurons, using a non-linear activation, can approximate *any* continuous function to any desired accuracy.

Read that again. Any smooth function — a spiral, a wiggly curve, a complicated decision surface — can be matched arbitrarily well by just one hidden layer if it's wide enough. The network is a **universal function approximator**.

**Intuition for why:** each hidden neuron with a non-linear activation creates one "bump" or "step" in the output. Add enough bumps of different heights, positions, and widths, and you can trace out any curve — the same way enough LEGO bricks can approximate any shape. More neurons = finer bricks = closer fit.

Two crucial caveats that keep this from being magic:

1. **It doesn't tell you how to *find* the weights.** The theorem says a good setting of weights *exists*; it doesn't say training will find it. That's what backpropagation (next lesson) is for.
2. **"Enough neurons" can be astronomically many.** A single wide layer might need billions of neurons for something a **deep** (many-layer) network captures with far fewer. Depth is exponentially more *efficient* than width for many functions. That's the practical reason we build deep networks instead of one enormous flat one.

So the theorem is a reassurance ("yes, the right network exists") not a recipe. It tells us MLPs are expressive enough; the rest of the track is about training them.

---

## Why non-linearity is essential

There's one non-negotiable ingredient: the **non-linear activation** between layers. Without it, stacking layers is pointless. Here's the proof, and it's worth seeing.

Suppose we skip activations, so each layer is just a linear map: layer 1 computes $\mathbf{h} = W_1\mathbf{x} + \mathbf{b}_1$, layer 2 computes $\mathbf{y} = W_2\mathbf{h} + \mathbf{b}_2$. Substitute:

$$\mathbf{y} = W_2(W_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (W_2 W_1)\mathbf{x} + (W_2\mathbf{b}_1 + \mathbf{b}_2)$$

Let $W' = W_2 W_1$ and $\mathbf{b}' = W_2\mathbf{b}_1 + \mathbf{b}_2$. Then $\mathbf{y} = W'\mathbf{x} + \mathbf{b}'$ — a **single** linear layer! Two stacked linear layers collapse into one. No matter how many linear layers you stack, they're equivalent to *one* line/plane. You'd be right back to the perceptron's limitation and XOR would remain unsolvable.

**The activation breaks this collapse.** By bending the signal non-linearly between layers, we prevent the multiplications from merging, so each layer adds genuinely new expressive power. This is *the* reason activation functions (Lesson 4) exist. In English: *without a non-linear squish between layers, depth buys you nothing.*

---

## Forward pass: how data flows through the network

The **forward pass** (or forward propagation) is the process of feeding an input through the network to get a prediction. It's just "apply each layer in turn."

For a 2-layer MLP (one hidden layer, one output layer), given input $\mathbf{x}$:

**Step 1 — hidden layer:**
$$\mathbf{z}^{(1)} = W_1\mathbf{x} + \mathbf{b}_1, \qquad \mathbf{h} = \sigma(\mathbf{z}^{(1)})$$

*In English:* multiply the input by the first weight matrix, add the first bias vector to get the pre-activations $\mathbf{z}^{(1)}$, then apply the activation elementwise to get the hidden representation $\mathbf{h}$.

**Step 2 — output layer:**
$$\mathbf{z}^{(2)} = W_2\mathbf{h} + \mathbf{b}_2, \qquad \hat{\mathbf{y}} = \phi(\mathbf{z}^{(2)})$$

*In English:* take the hidden vector, multiply by the second weight matrix, add the second bias, then apply an output activation $\phi$ (which might be sigmoid for probabilities, softmax for classes, or nothing for regression).

That's it. Data flows **left to right**, layer by layer, each layer being "linear transform, then squish." Everything downstream — loss computation, backprop — sits on top of this simple pipeline.

---

## Matrix formulation of a layer

The compact heart of it all:

$$\mathbf{h} = \sigma(W\mathbf{x} + \mathbf{b})$$

Let's nail down the shapes, because getting shapes right is 90% of debugging neural nets.

Say a layer takes $n$ inputs and produces $m$ outputs (has $m$ neurons):

- $\mathbf{x}$ is a vector of length $n$ — the input.
- $W$ is an $m \times n$ **matrix** — row $i$ holds the weights of neuron $i$. So $W$ has one row per output neuron and one column per input.
- $\mathbf{b}$ is a vector of length $m$ — one bias per neuron.
- $W\mathbf{x}$ is a vector of length $m$ — the matrix-vector product does *all m* dot products at once.
- $\mathbf{b}$ is added elementwise, giving the pre-activations $\mathbf{z}$ (length $m$).
- $\sigma$ is applied elementwise, giving the output $\mathbf{h}$ (length $m$).

**Why matrices?** Because a layer is just "$m$ perceptrons, each doing a dot product." Stacking those $m$ weight vectors as the rows of a matrix lets one matrix multiply compute all $m$ neurons simultaneously — and computers (especially GPUs) are extremely fast at matrix multiplication. The single line $\mathbf{h} = \sigma(W\mathbf{x}+\mathbf{b})$ *is* a whole layer.

### Batching many examples at once

In practice we push a whole **batch** of examples through together. Stack $N$ examples as rows of a matrix $X$ (shape $N \times n$). Then:

$$H = \sigma(X W^\top + \mathbf{b})$$

where $W^\top$ is $n \times m$, so $X W^\top$ is $N \times m$ — one row of hidden activations per example. The bias $\mathbf{b}$ is **broadcast** (added to every row). This is why you'll see `X @ W.T + b` in code. Processing a batch at once is far more efficient than one example at a time.

---

## The role of hidden layers

What are hidden layers actually *doing*? A beautiful way to think about it: **each hidden layer learns a new representation of the data — a set of features — that makes the next layer's job easier.**

- The **first** hidden layer combines raw inputs into simple patterns. On images, first-layer neurons often learn to detect edges and blobs.
- The **second** layer combines those simple patterns into more complex ones — corners, textures, shapes.
- **Deeper** layers combine those into high-level concepts — "wheel," "eye," "the sentiment is negative."

This is **representation learning**: instead of hand-engineering features (the old way), the network *discovers* useful features automatically, layer by layer, because that's what minimizes the loss. Each hidden layer is a stepping stone that reshapes the data so that, by the final layer, the classes are simple to separate — often literally linearly separable in that final learned space.

For XOR specifically: the hidden layer transforms the four non-separable points into a new 2D space where they *become* linearly separable, and the output neuron then draws one line. The hidden layer "untangles" the problem.

---

## Implementing a 2-layer MLP in pure numpy

Let's build a 2-layer MLP and use it to solve **XOR** — the problem a single perceptron couldn't touch. We'll only do the **forward pass** here (we'll add learning/backprop next lesson), but we'll hand-pick weights that we *know* solve XOR, to prove the architecture works.

```python
import numpy as np

def sigmoid(z):
    # Smooth 'squish' to (0,1). We use this instead of the step function.
    return 1.0 / (1.0 + np.exp(-z))

# ---- Network: 2 inputs -> 2 hidden neurons -> 1 output ----
# Hand-chosen weights that solve XOR.
# Hidden neuron 1 acts like OR, hidden neuron 2 acts like AND.
W1 = np.array([[20.0, 20.0],     # hidden neuron 1 weights
               [20.0, 20.0]])    # hidden neuron 2 weights
b1 = np.array([-10.0,            # OR-ish: fires if at least one input on
               -30.0])           # AND-ish: fires only if both inputs on

W2 = np.array([[20.0, -20.0]])   # output: (OR) minus (AND) = XOR
b2 = np.array([-10.0])

def forward(x):
    x = np.asarray(x, dtype=float)
    z1 = W1 @ x + b1        # pre-activations of hidden layer, shape (2,)
    h  = sigmoid(z1)        # hidden activations, shape (2,)
    z2 = W2 @ h + b2        # output pre-activation, shape (1,)
    y  = sigmoid(z2)        # final output in (0,1)
    return y[0]

print("XOR via a 2-layer MLP:")
for x in [[0,0],[0,1],[1,0],[1,1]]:
    out = forward(x)
    print(f"  input {x} -> {out:.3f}  (rounded {round(out)})")
```

Run it, and you'll see outputs near 0 for `[0,0]` and `[1,1]`, and near 1 for `[0,1]` and `[1,0]` — **XOR solved!** The hidden layer created an "OR" feature and an "AND" feature; the output neuron computed OR − AND, which is exactly XOR. This is the concrete payoff of depth.

> **Why the big weights (20, −30)?** Large weights make the sigmoid behave almost like a sharp step function, giving clean 0/1-ish outputs. With learned weights they'd be gentler, but the same logic holds.

---

## A random-initialized MLP with batched forward pass

Here's a more realistic, general-purpose forward pass: random weights, batched input, clean shapes. This is the template you'll extend in the next lesson.

```python
import numpy as np

rng = np.random.default_rng(42)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# Network shape: n_in -> n_hidden -> n_out
n_in, n_hidden, n_out = 3, 5, 2

# Initialize weights small and random; biases at zero.
# Shapes are chosen so batched matmuls line up.
W1 = rng.normal(0, 0.5, size=(n_in, n_hidden))   # (3, 5)
b1 = np.zeros(n_hidden)                           # (5,)
W2 = rng.normal(0, 0.5, size=(n_hidden, n_out))   # (5, 2)
b2 = np.zeros(n_out)                              # (2,)

def forward(X):
    # X has shape (batch, n_in). We use X @ W so shapes chain naturally.
    Z1 = X @ W1 + b1        # (batch, 5)
    H  = sigmoid(Z1)        # (batch, 5)
    Z2 = H @ W2 + b2        # (batch, 2)
    Y  = sigmoid(Z2)        # (batch, 2)
    return Y

# A batch of 4 examples, each with 3 features
X = rng.normal(size=(4, 3))
out = forward(X)
print("input shape :", X.shape)
print("output shape:", out.shape)   # (4, 2)
print(out)
```

Notice how the shapes chain: `(4,3) @ (3,5) → (4,5) @ (5,2) → (4,2)`. Reading shapes like this is a skill — when a network throws a shape error, tracing this chain is how you find the bug. The rule: the inner dimensions must match, and the outer dimensions become the result's shape.

---

## Counting parameters

The **number of parameters** is the total count of weights and biases the network must learn. It's a key measure of model size (GPT-3 has 175 *billion* parameters).

For one fully-connected layer taking $n$ inputs to $m$ outputs:

$$\text{params} = \underbrace{n \times m}_{\text{weights}} + \underbrace{m}_{\text{biases}}$$

*In English:* every input connects to every output (that's $n\times m$ weights), plus one bias per output neuron ($m$ biases).

**Worked example — a 784 → 128 → 10 MLP** (like a digit classifier):

| Layer | Weights | Biases | Total |
|---|---|---|---|
| Input → Hidden (784→128) | $784 \times 128 = 100352$ | $128$ | $100480$ |
| Hidden → Output (128→10) | $128 \times 10 = 1280$ | $10$ | $1290$ |
| **Whole network** | | | **101770** |

So this small network already has ~102k parameters — each one a knob training must tune. Let's compute it in code:

```python
def count_params(layer_sizes):
    """layer_sizes e.g. [784, 128, 10] -> total parameters."""
    total = 0
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        weights = n_in * n_out
        biases = n_out
        total += weights + biases
        print(f"  {n_in:4d} -> {n_out:4d}: {weights + biases:>8d} params")
    return total

print("784 -> 128 -> 10 network:")
print("TOTAL:", count_params([784, 128, 10]))

print("\n784 -> 256 -> 128 -> 10 network:")
print("TOTAL:", count_params([784, 256, 128, 10]))
```

**Why parameter count matters:** more parameters = more capacity to fit complex patterns, but also more memory, slower training, and more risk of **overfitting** (memorizing the training data instead of learning general rules). At IOAI you constantly trade off "big enough to learn the task" against "small enough to train in the time/compute you have." Being able to estimate parameters in your head is a genuinely useful competition skill.

---

## Why this matters for IOAI

- **The MLP is the default first model.** Faced with tabular data or as the "head" on top of a CNN/transformer, an MLP is often what you reach for at IOAI. You'll write these constantly.
- **Shape reasoning is a survival skill.** Half of all deep-learning bugs are shape mismatches. The `X @ W + b` chain from this lesson is the mental model that lets you debug them fast under time pressure.
- **Width vs depth is a real decision.** The universal approximation theorem plus the "depth is more efficient" insight guides how you size a network. IOAI problems reward knowing that a deeper net often beats a wider one for the same parameter budget.
- **Parameter counting sizes your compute.** Knowing roughly how big your model is tells you whether it'll fit in memory and train in time — directly relevant when the competition gives you a Colab GPU and a deadline.
- **Everything builds on this.** Transformers are MLPs plus attention; CNNs are MLPs with weight-sharing. Solidify the MLP and the advanced tracks become variations on a theme.

---

## Sort MLP Concepts

```widget
{
  "type": "concept-sort",
  "title": "Width or Depth? Which Property?",
  "categories": [
    { "name": "Relates to Width (neurons/layer)", "color": "#5B5BD6" },
    { "name": "Relates to Depth (num layers)", "color": "#F97316" }
  ],
  "items": [
    { "text": "More neurons in a single hidden layer", "category": "Relates to Width (neurons/layer)" },
    { "text": "Hierarchical feature learning", "category": "Relates to Depth (num layers)" },
    { "text": "Universal approximation theorem guarantee", "category": "Relates to Width (neurons/layer)" },
    { "text": "ResNet has 50 layers (deep)", "category": "Relates to Depth (num layers)" },
    { "text": "1024-unit hidden layer vs 512-unit", "category": "Relates to Width (neurons/layer)" },
    { "text": "Requires gradient flow across many steps", "category": "Relates to Depth (num layers)" }
  ]
}
```

---

## Summary

- An **MLP** stacks layers of neurons; every neuron connects to all neurons in the next layer (**fully connected**).
- Layers are **input → hidden(s) → output**; **width** = neurons per layer, **depth** = number of layers.
- Each layer computes $\mathbf{h} = \sigma(W\mathbf{x}+\mathbf{b})$ — a linear transform followed by a non-linear activation.
- **Non-linearity is essential**: without it, stacked layers collapse into a single linear layer.
- The **universal approximation theorem** says one wide-enough hidden layer can approximate any continuous function — but **depth** achieves the same far more efficiently.
- The **forward pass** feeds data left-to-right through the layers to produce a prediction.
- A 2-layer MLP solves **XOR** because the hidden layer learns features that make the problem linearly separable.
- **Parameters** = weights + biases; a fully-connected $n\to m$ layer has $nm + m$ of them.
