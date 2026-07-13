---
title: The Perceptron
track: nn
order: 1
estimatedTime: 40
difficulty: intermediate
---

# The Perceptron

Every deep neural network you will ever meet — the models behind ChatGPT, image classifiers, self-driving cars — is built from one tiny idea repeated millions of times. That idea is the **perceptron**: a single artificial neuron.

If you understand this one lesson deeply, everything that comes later (multi-layer networks, backpropagation, CNNs, transformers) is just "more of the same, stacked cleverly." So let's take our time and really get it.

> **Big picture:** A perceptron takes some numbers in, multiplies each by a weight, adds them up, and outputs a decision. That's the whole thing. The magic is that the weights can be *learned* from data.

---

## A little history: Rosenblatt, 1958

In 1958, a psychologist named **Frank Rosenblatt** built a machine called the **Mark I Perceptron**. It was not software running on a laptop — it was a room-sized machine with a camera-like grid of 400 light sensors, wired to electrical circuits, with weights stored as the settings of little motor-driven potentiometers (adjustable knobs).

Rosenblatt's goal was ambitious: he wanted a machine that could **learn to recognize patterns** the way a brain does, instead of being programmed with explicit rules. He showed it images, it made guesses, and it adjusted its own knobs when it was wrong.

The press went wild. The New York Times reported that the Navy expected the perceptron to be "the embryo of an electronic computer that [would] be able to walk, talk, see, write, reproduce itself and be conscious of its existence."

That was, to put it kindly, an overstatement. But the core idea — **a learning machine that adjusts numerical weights from examples** — is exactly what powers modern AI. Rosenblatt was 60 years early, but he was right.

Later in this lesson we'll also meet the perceptron's famous **limitation** (the XOR problem), which nearly killed the whole field in the 1970s. Understanding that limit is what motivates *multi-layer* networks in the next lesson.

---

## The neuron analogy

The perceptron is loosely inspired by a **biological neuron** — a brain cell.

A real neuron works roughly like this:

1. It receives electrical signals from thousands of other neurons through connections called **dendrites**.
2. Some connections are strong, some are weak. A strong connection passes on more signal.
3. The neuron **adds up** all the incoming signals.
4. If the total exceeds some **threshold**, the neuron "fires" — it sends its own signal down its **axon** to other neurons. If not, it stays quiet.

The artificial perceptron copies this structure with numbers:

| Biology | Perceptron |
|---|---|
| Incoming signals from other neurons | Input numbers $x_1, x_2, \dots, x_n$ |
| Strength of each connection | Weights $w_1, w_2, \dots, w_n$ |
| Adding up the signals | Weighted sum $\sum w_i x_i$ |
| Firing threshold | Bias $b$ and threshold |
| Fire / don't fire | Output $\hat{y}$ (1 or 0) |

Don't take the brain analogy too literally — real neurons are far more complicated, and modern deep learning has drifted very far from biology. But as a *first picture*, "inputs, weighted connections, a threshold, an output" is exactly right.

---

## Mathematical formulation

Here is the perceptron in one line of math:

$$\hat{y} = \sigma(\mathbf{w} \cdot \mathbf{x} + b)$$

Let's unpack every single symbol, because each one matters.

- $\mathbf{x} = (x_1, x_2, \dots, x_n)$ is the **input vector** — the numbers describing one example. If we're classifying a flower, $x_1$ might be petal length and $x_2$ petal width.
- $\mathbf{w} = (w_1, w_2, \dots, w_n)$ is the **weight vector** — one weight per input. Big positive weight = "this input strongly pushes toward output 1." Big negative weight = "this input pushes toward output 0."
- $\mathbf{w} \cdot \mathbf{x}$ is the **dot product**: $w_1 x_1 + w_2 x_2 + \dots + w_n x_n$. In plain English: *multiply each input by its weight, then add them all up.*
- $b$ is the **bias** — a single number added on top. It shifts the threshold. Think of it as how "eager" the neuron is to fire even when all inputs are zero.
- $\sigma$ (the Greek letter sigma) is the **activation function**. For the classic perceptron it's a **step function**: output 1 if the total is positive, 0 otherwise.
- $\hat{y}$ (read "y-hat") is the **prediction** — the neuron's output.

The quantity inside the activation, $z = \mathbf{w} \cdot \mathbf{x} + b$, is called the **pre-activation** or **logit**. We'll use the letter $z$ for it a lot.

**In one English sentence:** *Multiply each input by its weight, add them up, add the bias, and if the result is above zero, say 1 — otherwise say 0.*

---

## The step activation function

The classic perceptron uses the **Heaviside step function**:

$$
\sigma(z) =
\begin{cases}
1 & \text{if } z \geq 0 \\
0 & \text{if } z < 0
\end{cases}
$$

**What this says in English:** if the weighted sum plus bias is zero or positive, the neuron fires (outputs 1). If it's negative, the neuron stays off (outputs 0). There's no in-between — it's a hard, all-or-nothing switch.

Here it is drawn as ASCII:

```
 output
   1 |          ________________
     |         |
     |         |
   0 |_________|________________  z (pre-activation)
              z=0
```

The step function is simple and intuitive, but it has one fatal flaw that we'll hit later: **it's flat everywhere**, so its slope (derivative) is zero. That means it gives us no information about *which direction* to nudge the weights. This is exactly why later networks switch to smooth activations like sigmoid and ReLU (Lesson 4). For now, though, the step function is perfect for building intuition.

```python
import numpy as np

def step(z):
    # Returns 1.0 where z >= 0, else 0.0. Works on arrays too.
    return (z >= 0).astype(float)

print(step(np.array([-2.0, -0.1, 0.0, 0.5, 3.0])))
# [0. 0. 1. 1. 1.]
```

---

## What does a perceptron actually compute? A concrete example

Imagine you're deciding whether to go outside for a run. You care about two things:

- $x_1$ = is it sunny? (1 = yes, 0 = no)
- $x_2$ = did you sleep well? (1 = yes, 0 = no)

Suppose your personal weights are $w_1 = 2$ (sun matters a lot) and $w_2 = 1$ (sleep matters a bit), with bias $b = -2$.

The perceptron computes $z = 2 x_1 + 1 x_2 - 2$, then outputs $\text{step}(z)$.

| Sunny $x_1$ | Slept $x_2$ | $z = 2x_1 + x_2 - 2$ | Output (run?) |
|---|---|---|---|
| 0 | 0 | $-2$ | 0 (no) |
| 0 | 1 | $-1$ | 0 (no) |
| 1 | 0 | $0$ | 1 (yes) |
| 1 | 1 | $1$ | 1 (yes) |

So this particular perceptron encodes the rule: *"I'll run whenever it's sunny, regardless of sleep."* By changing the weights and bias we'd get different rules. **Learning** means finding the weights that reproduce the behavior we want from examples.

---

## The perceptron as a line (the decision boundary)

Here's the geometric picture that makes everything click.

The perceptron outputs 1 when $\mathbf{w} \cdot \mathbf{x} + b \geq 0$ and 0 when it's negative. The **boundary** between these two regions is the set of points where

$$\mathbf{w} \cdot \mathbf{x} + b = 0.$$

In 2D (two inputs), that equation $w_1 x_1 + w_2 x_2 + b = 0$ is the equation of a **straight line**. In 3D it's a flat plane. In higher dimensions it's called a **hyperplane** — but it's always "flat."

So a single perceptron draws **one straight line** and says "everything on this side is class 1, everything on that side is class 0."

```
 x2
  |   0   0   |  1   1
  |  0   0    |   1    1
  |    0   0  |  1   1
  |___________|______________ x1
              ^
        decision boundary (a line)
```

- The weight vector $\mathbf{w}$ points **perpendicular** to the line, toward the "class 1" side.
- The bias $b$ controls how far the line sits from the origin.

This geometric fact is the whole story of the perceptron's power *and* its weakness. It can perfectly separate data that a straight line can separate (**linearly separable** data). It is helpless against data that needs a curve or multiple lines. Hold that thought — it comes back with XOR.

---

## The perceptron learning rule

Now the exciting part: how does the perceptron *learn* the right weights? Rosenblatt's rule is beautifully simple and it's the ancestor of all modern training.

The idea: **show the perceptron an example, let it guess, and if it's wrong, nudge the weights to make the right answer more likely next time.**

For each training example $(\mathbf{x}, y)$ where $y$ is the true label:

1. Compute the prediction: $\hat{y} = \text{step}(\mathbf{w} \cdot \mathbf{x} + b)$.
2. Compute the **error**: $e = y - \hat{y}$. This is $0$ if correct, $+1$ if we said 0 but should've said 1, and $-1$ if we said 1 but should've said 0.
3. Update each weight and the bias:

$$w_i \leftarrow w_i + \eta \, e \, x_i \qquad b \leftarrow b + \eta \, e$$

Here $\eta$ (the Greek letter "eta") is the **learning rate** — a small positive number like 0.1 that controls how big each nudge is.

**Reading the update rule in English:** *If we got it right ($e = 0$), don't change anything. If we should have fired but didn't ($e = +1$), increase the weights of the inputs that were active, so next time the sum is bigger and more likely to cross the threshold. If we fired but shouldn't have ($e = -1$), decrease those weights.*

Notice the multiplication by $x_i$: weights only change for inputs that were actually "on." An input of zero contributes nothing to the mistake, so its weight isn't adjusted. That's sensible — you fix the connections that were responsible.

> **The Perceptron Convergence Theorem (1962):** If the data *is* linearly separable, this rule is mathematically **guaranteed** to find a separating line in a finite number of steps. That was a genuinely remarkable result. The catch — as we'll see — is the phrase "if the data is linearly separable."

---

## Full numpy implementation from scratch

Let's build a perceptron and train it to learn the **AND** function. The AND function outputs 1 only when both inputs are 1.

```python
import numpy as np

# ---- Training data: the AND function ----
# Each row of X is one example (x1, x2). y is the desired output.
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], dtype=float)
y = np.array([0, 0, 0, 1], dtype=float)  # AND: only 1&1 -> 1

# ---- Initialize weights and bias ----
# Start at zero. (For a single perceptron this is fine.)
w = np.zeros(2)
b = 0.0
lr = 0.1          # learning rate (eta)
epochs = 20       # how many times we sweep through the whole dataset

def step(z):
    return 1.0 if z >= 0 else 0.0

# ---- Training loop ----
for epoch in range(epochs):
    errors = 0
    for xi, target in zip(X, y):
        z = np.dot(w, xi) + b          # pre-activation
        pred = step(z)                 # prediction (0 or 1)
        error = target - pred          # -1, 0, or +1
        # Perceptron update rule:
        w += lr * error * xi
        b += lr * error
        errors += int(error != 0)
    print(f"epoch {epoch+1:2d} | w = {w}, b = {b:.2f} | mistakes = {errors}")
    if errors == 0:
        print("Converged! No mistakes left.")
        break

# ---- Test the trained perceptron ----
print("\nFinal predictions:")
for xi, target in zip(X, y):
    pred = step(np.dot(w, xi) + b)
    print(f"  input {xi} -> predicted {pred:.0f}, true {target:.0f}")
```

If you run this, you'll watch the weights climb from zero and the number of mistakes drop to zero, usually within a handful of epochs. The perceptron *discovers* the AND rule entirely from the four examples — nobody told it "output 1 only when both are 1."

**Key implementation notes:**

- One **epoch** = one full pass over all training examples.
- We update after *every* example (this is called **online** or **stochastic** learning). You could instead accumulate updates over the whole dataset (batch learning).
- We stop early once there are no mistakes — the convergence theorem promises this will happen for separable data like AND.

---

## Vectorizing the prediction

Looping example-by-example is clear but slow. numpy lets us predict on the *entire* dataset at once with matrix math — a habit worth building now because every real network does this.

```python
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
w = np.array([0.2, 0.2])
b = -0.3

# X @ w does the dot product for every row simultaneously.
Z = X @ w + b          # shape (4,) — one pre-activation per example
preds = (Z >= 0).astype(int)
print("Z    :", Z)
print("preds:", preds)
```

`X @ w` is matrix-vector multiplication: each row of $X$ is dotted with $w$. This one line replaces the Python loop and runs far faster, especially for thousands of examples. Get comfortable with `@` — it's everywhere from here on.

---

## The XOR problem: where one perceptron fails

Now for the plot twist that shook AI in 1969.

Consider the **XOR** ("exclusive or") function. It outputs 1 when the inputs are *different* and 0 when they're the *same*:

| $x_1$ | $x_2$ | XOR |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Let's plot these four points. Mark the 1s with `●` and the 0s with `○`:

```
 x2
  1 |  ●        ○
    |
    |
  0 |  ○        ●
    |___________________ x1
       0        1
```

Look carefully. The two `●` points (the 1s) sit at *opposite corners*. The two `○` points (the 0s) sit at the *other two* opposite corners.

**Question:** Can you draw a single straight line that puts both `●` on one side and both `○` on the other?

Try it. You can't. No matter how you angle a line, you can never separate the diagonals with one cut. XOR is **not linearly separable** — and a single perceptron can *only* draw a straight line. Therefore **a single perceptron can never learn XOR.** Run the training loop on XOR and it will loop forever, never reaching zero mistakes.

In 1969, Marvin Minsky and Seymour Papert published a book called *Perceptrons* that proved this limitation rigorously. The result was so discouraging that funding dried up and the field entered the first "**AI winter**." The dream seemed dead.

---

## The escape: stack perceptrons into layers

Here's the resolution — and it's the bridge to the entire rest of this course.

A single perceptron draws one line. But what if we use **several** perceptrons and feed their outputs into *another* perceptron? Each first-layer perceptron draws its own line, and the second layer *combines* those lines. With combinations of lines you can carve out **any** region, including the two diagonal corners of XOR.

Concretely, XOR can be written using simpler functions the perceptron *can* learn:

$$\text{XOR}(x_1, x_2) = \text{AND}\big(\,\text{OR}(x_1, x_2),\ \text{NAND}(x_1, x_2)\,\big)$$

- OR fires if either input is 1 (linearly separable ✓).
- NAND fires unless both are 1 (linearly separable ✓).
- ANDing those two gives XOR.

So two perceptrons in a hidden layer, feeding one output perceptron, solves XOR. This stack is called a **Multi-Layer Perceptron (MLP)** — and it's exactly what Lesson 2 is about. The "AI winter" ended when people realized you could train these stacks with **backpropagation** (Lesson 3).

> **The moral of the story:** A single neuron is a straight-line classifier — powerful but limited. Depth (layers) is what buys you the ability to model curves, corners, and arbitrarily complex patterns.

---

## Visualizing the decision boundary in code

Let's actually *see* the line a trained perceptron draws. We'll train on a simple separable dataset and print an ASCII map of its decisions across a grid.

```python
import numpy as np

# A linearly separable toy dataset: points above the line y=x are class 1.
rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, size=(200, 2))
y = (X[:, 1] > X[:, 0]).astype(float)   # class 1 if x2 > x1

w = np.zeros(2)
b = 0.0
lr = 0.1

def step(z):
    return (z >= 0).astype(float)

# Train
for epoch in range(50):
    z = X @ w + b
    preds = step(z)
    errors = y - preds
    # Batch update: sum the nudges over all examples
    w += lr * (errors @ X) / len(X)
    b += lr * errors.mean()

# Draw the decision boundary as an ASCII grid
print("Decision map ('#'=class1, '.'=class0):")
for j in range(20, -1, -1):        # rows: x2 from +1 down to -1
    x2 = -1 + 2 * j / 20
    row = ""
    for i in range(41):            # cols: x1 from -1 to +1
        x1 = -1 + 2 * i / 40
        pred = step(np.array([w[0]*x1 + w[1]*x2 + b]))[0]
        row += "#" if pred == 1 else "."
    print(row)
print(f"\nLearned: w = {w}, b = {b:.3f}")
```

When you run this you'll see a clean diagonal split — the perceptron rediscovered the line $x_2 = x_1$. That crisp straight edge is the visual signature of a linear classifier. Every technique in later lessons is, in some sense, about bending that edge into a curve.

---

## Why this matters for IOAI

You will probably never deploy a bare single-layer perceptron at the International Olympiad in Artificial Intelligence — modern problems need deep networks. So why spend a whole lesson here?

- **It's the atom of everything.** Every `nn.Linear` layer in PyTorch, every attention head in a transformer, is a bank of perceptron-style weighted sums. IOAI problems that involve tweaking model architectures assume you *feel* what one neuron does.
- **The learning rule is the seed of gradient descent.** "Predict, measure error, nudge weights in proportion to the error" is the exact skeleton of backpropagation and Adam. Understanding it here makes Lessons 3 and 5 far easier.
- **Linear separability is a real diagnostic.** When a model refuses to fit XOR-like data, the fix is more capacity (depth/width) or better features — a judgment call you'll make under time pressure at the competition.
- **The decision-boundary picture is your intuition tool.** IOAI questions often ask *why* a model misclassifies certain points. Being able to imagine the boundary geometry lets you reason instead of guess.

Master the perceptron, and the rest of the Neural Networks track is a series of natural extensions rather than a wall of new ideas.

---

## Summary

- The **perceptron** (Rosenblatt, 1958) is a single artificial neuron: inputs → weights → sum + bias → activation → output.
- Its formula is $\hat{y} = \sigma(\mathbf{w} \cdot \mathbf{x} + b)$, where $\sigma$ is the **step function** for the classic version.
- The pre-activation $z = \mathbf{w}\cdot\mathbf{x} + b$ defines a **line/hyperplane**; the perceptron classifies by which side of it a point lands on.
- The **learning rule** $w_i \leftarrow w_i + \eta(y-\hat{y})x_i$ nudges weights toward correctness and is *guaranteed* to converge on linearly separable data.
- A single perceptron **cannot** solve **XOR** because XOR is not linearly separable — this triggered the first AI winter.
- **Stacking** perceptrons into layers (an MLP) overcomes this, which is where we go next.
