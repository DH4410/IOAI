---
title: Activation Functions
track: nn
order: 4
estimatedTime: 35
difficulty: intermediate
---

# Activation Functions

We've said several times that neural networks need **non-linear activation functions** between layers. This lesson is all about those functions: what they are, why they matter so much, and how to pick the right one. The activation is the small "squish" applied to each neuron's output, and choosing it well is one of the highest-leverage decisions in network design.

> **Big picture:** An activation function is a simple non-linear function applied elementwise after each layer's weighted sum. It's what lets deep networks model curves, and its exact shape controls how well gradients flow during training.

---

## Why we need non-linear activations (again, but deeper)

We proved this in Lesson 2, but it's so central that let's re-state it crisply. A layer computes $\mathbf{z} = W\mathbf{x} + \mathbf{b}$ — that's a **linear** operation (a rotation, scaling, and shift). If you stack linear operations, you get... another linear operation. The whole deep network would collapse into a single line/plane, no matter how many layers.

The **activation function** $\sigma$ inserts a non-linear bend between layers: $\mathbf{h} = \sigma(\mathbf{z})$. That bend is what prevents the collapse and lets each layer add real expressive power. Without it, "deep learning" would just be "linear regression with extra steps."

**Two jobs of an activation:**
1. **Introduce non-linearity** so the network can approximate complex functions.
2. **Keep gradients flowing** so backprop can train deep stacks. As we saw in Lesson 3, the activation's *derivative* multiplies into every gradient passing through it — so the shape of that derivative decides whether gradients vanish, explode, or flow nicely.

Different activations trade off these jobs differently. Let's meet the main ones.

---

## Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**In English:** the sigmoid takes any real number and squishes it into the range $(0, 1)$. Big positive inputs → close to 1; big negative inputs → close to 0; input 0 → exactly 0.5. It's a smooth S-shaped curve.

```
   1 |            _________
     |         __/
 0.5 |      __/
     |   __/
   0 |__/________________________  x
              0
```

**Where it's used:** historically the default activation; today mostly for the **output** of a binary classifier (because $(0,1)$ reads naturally as a probability) and inside gates of LSTMs.

**The saturation problem — its fatal flaw:** look at the flat tails. When $x$ is very positive or very negative, the curve is nearly flat, so its **derivative is nearly zero**. Recall the sigmoid derivative:

$$\sigma'(x) = \sigma(x)\,(1 - \sigma(x))$$

Its maximum is only **0.25** (at $x=0$), and it's tiny everywhere else. In a deep network, backprop multiplies these small numbers layer after layer, so the gradient **vanishes** — early layers barely learn. This is the vanishing-gradient problem from Lesson 3, and it's why sigmoid fell out of favor for hidden layers.

A second, smaller issue: sigmoid outputs are always **positive** (not centered at zero), which makes gradient updates zig-zag inefficiently.

```python
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

xs = np.array([-6, -3, -1, 0, 1, 3, 6], dtype=float)
print("x       :", xs)
print("sigmoid :", np.round(sigmoid(xs), 3))
print("deriv   :", np.round(sigmoid_deriv(xs), 3))
# Notice the derivative is ~0 at the tails (x=±6) — that's saturation.
```

---

## Tanh

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

**In English:** tanh (hyperbolic tangent) is a stretched, shifted sigmoid. It squishes inputs into $(-1, 1)$ instead of $(0,1)$, and it passes through $0$ at $x=0$. Same S-shape, but **centered at zero**.

```
   1 |            _________
     |         __/
   0 |______ _/________________  x
     |    __/
  -1 |___/
```

**Advantage over sigmoid:** because its outputs are centered around zero (they can be negative), gradient updates are better-behaved — no systematic zig-zag. Its derivative $\tanh'(x) = 1 - \tanh^2(x)$ peaks at **1** (vs sigmoid's 0.25), so gradients shrink less. tanh was the preferred hidden-layer activation before ReLU.

**But** tanh still **saturates** at its tails just like sigmoid — flat regions with near-zero derivative — so deep tanh networks still suffer vanishing gradients, just less severely. Handy relationship: $\tanh(x) = 2\sigma(2x) - 1$.

```python
import numpy as np

def tanh(x):
    return np.tanh(x)   # numpy has it built in

def tanh_deriv(x):
    return 1 - np.tanh(x) ** 2

xs = np.array([-3, -1, 0, 1, 3], dtype=float)
print("tanh :", np.round(tanh(xs), 3))
print("deriv:", np.round(tanh_deriv(xs), 3))   # peaks at 1.0 when x=0
```

---

## ReLU — the modern default

$$\text{ReLU}(x) = \max(0, x)$$

**In English:** the Rectified Linear Unit is almost embarrassingly simple: if the input is positive, pass it through unchanged; if it's negative, output zero. That's it. No exponentials, no curves.

```
 output
     |                    /
     |                  /
     |                /
   0 |______________/________________  x
                    0
   (flat at 0 for x<0, then a 45° line for x>0)
```

**Why ReLU revolutionized deep learning:**

- **No saturation for positive inputs.** Its derivative is exactly **1** whenever $x > 0$. So gradients pass through *undiminished* — no vanishing on the positive side. This alone made training very deep networks (dozens of layers) practical for the first time.
- **Blazingly cheap.** Just a comparison with zero — far faster than computing $e^x$.
- **Sparse activations.** About half the neurons output zero for any input, which is efficient and can act as a mild regularizer.

Its derivative:
$$\text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x < 0 \end{cases}$$
*(undefined exactly at 0; in practice we just define it as 0 or 1 there — it doesn't matter.)*

**The dying ReLU problem:** notice the derivative is **0 for all negative inputs**. If a neuron's weights push it into always-negative territory (its pre-activation is negative for every training example), then its gradient is always zero, so it **never updates again** — it's "dead." A large learning rate or bad init can kill a big fraction of neurons this way, permanently reducing the network's capacity. This is the one real downside of plain ReLU, and it motivates the variants below.

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

xs = np.array([-3, -1, 0, 2, 5], dtype=float)
print("relu :", relu(xs))          # [0. 0. 0. 2. 5.]
print("deriv:", relu_deriv(xs))    # [0. 0. 0. 1. 1.]  -> zero for x<=0 (dying risk)
```

---

## Leaky ReLU, ELU, and GELU

These fix or refine ReLU's dead-neuron issue.

### Leaky ReLU

$$\text{LeakyReLU}(x) = \begin{cases} x & x > 0 \\ \alpha x & x \le 0 \end{cases}$$

**In English:** instead of flattening negatives to exactly zero, let a *small* slope $\alpha$ (typically 0.01) leak through. Now the derivative for negative inputs is $\alpha$ (not 0), so neurons can never fully die — there's always a trickle of gradient to revive them.

```python
import numpy as np
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)
```

### ELU (Exponential Linear Unit)

$$\text{ELU}(x) = \begin{cases} x & x > 0 \\ \alpha(e^{x} - 1) & x \le 0 \end{cases}$$

**In English:** positive side is like ReLU; negative side curves smoothly down to $-\alpha$ instead of a hard corner. This makes outputs closer to zero-mean (helping training) and gives a smooth derivative everywhere. Costs an exponential, so it's a bit slower.

### GELU (Gaussian Error Linear Unit) — the transformer favorite

$$\text{GELU}(x) = x \cdot \Phi(x)$$

where $\Phi(x)$ is the cumulative distribution function of the standard normal — the probability that a standard Gaussian is less than $x$.

**In English:** GELU multiplies the input by "how likely a standard normal value is to be below it." For big positive $x$, $\Phi(x)\approx 1$ so it acts like the identity; for big negative $x$, $\Phi(x)\approx 0$ so it zeroes out; near zero it gives a smooth, slightly curved transition that can even dip *slightly* negative. Think of it as a "soft, probabilistic ReLU."

**Why it matters:** GELU is the activation used inside **BERT, GPT, and virtually every modern transformer**. Its smoothness helps these enormous models train stably. A common fast approximation:

$$\text{GELU}(x) \approx 0.5\,x\left(1 + \tanh\!\left[\sqrt{\tfrac{2}{\pi}}\,(x + 0.044715\,x^3)\right]\right)$$

```python
import numpy as np

def gelu(x):
    # tanh approximation used in practice (e.g. GPT)
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

xs = np.array([-3, -1, 0, 1, 3], dtype=float)
print("gelu:", np.round(gelu(xs), 3))
# Note it dips slightly negative around x=-1, unlike ReLU which is exactly 0 there.
```

> **Since you'll hit the CV and NLP tracks:** when you see `nn.GELU()` in a transformer, you now know it's a smooth ReLU-like function chosen because it trains huge models gracefully.

---

## Softmax — for multi-class output

Everything above is a *hidden-layer* activation. **Softmax** is different: it's an **output** activation for **multi-class classification**, and it acts on a whole vector at once, not elementwise.

Given a vector of raw scores (logits) $\mathbf{z} = (z_1, \dots, z_K)$ for $K$ classes:

$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

**In English:** exponentiate every score (making them all positive and amplifying differences), then divide each by the total so they **sum to 1**. The result is a **probability distribution** over the classes — each output is between 0 and 1, and together they add to 100%.

Example: logits $(2.0, 1.0, 0.1)$ → softmax ≈ $(0.66, 0.24, 0.10)$. The network is saying "66% class 0, 24% class 1, 10% class 2." The biggest logit gets the biggest probability, but softly.

**The numerical stability trick:** $e^{z}$ overflows for large $z$. Because softmax is unchanged if you subtract a constant from every logit, we subtract the max first:

```python
import numpy as np

def softmax(z):
    # Subtract max for numerical stability (result is identical mathematically).
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)

logits = np.array([2.0, 1.0, 0.1])
p = softmax(logits)
print("probabilities:", np.round(p, 3))   # [0.659 0.242 0.099]
print("sum          :", p.sum())          # 1.0
```

Softmax pairs with the **cross-entropy loss** for training classifiers — you'll see this constantly. **Sigmoid vs softmax:** sigmoid is for *independent* yes/no outputs (multi-label); softmax is for *mutually exclusive* classes (pick exactly one).

---

## Choosing the right activation

A practical cheat-sheet you can lean on at IOAI:

| Situation | Use | Why |
|---|---|---|
| Hidden layers, default | **ReLU** | Fast, no vanishing on positive side, works almost always |
| Hidden layers, worried about dead neurons | **Leaky ReLU / ELU** | Small negative slope keeps neurons alive |
| Hidden layers in a transformer | **GELU** | Smooth; the field standard for attention models |
| Binary classification output | **Sigmoid** | Maps to a single probability in (0,1) |
| Multi-class (one correct class) output | **Softmax** | Probability distribution over classes |
| Multi-label (many can be correct) output | **Sigmoid** (per class) | Independent probabilities |
| Regression output (predicting a number) | **None (linear)** | Output can be any real value |

**Rules of thumb:**
- Start with **ReLU** for hidden layers. It's the boring, correct default. Only reach for variants if you diagnose dead neurons or you're building a transformer.
- **Never** use sigmoid/tanh for many hidden layers in a deep net — vanishing gradients.
- Match the **output** activation to the task, and match the **loss** to the output (softmax ↔ cross-entropy, linear ↔ MSE).

---

## Putting them side by side

```python
import numpy as np

def sigmoid(x): return 1/(1+np.exp(-x))
def tanh(x):    return np.tanh(x)
def relu(x):    return np.maximum(0, x)
def leaky(x):   return np.where(x>0, x, 0.01*x)
def gelu(x):    return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3)))

xs = np.array([-4, -2, -1, 0, 1, 2, 4], dtype=float)
print(f"{'x':>5} {'sigmoid':>8} {'tanh':>7} {'relu':>6} {'leaky':>7} {'gelu':>7}")
for x in xs:
    print(f"{x:5.1f} {sigmoid(x):8.3f} {tanh(x):7.3f} {relu(x):6.1f} {leaky(x):7.3f} {gelu(x):7.3f}")
```

Run this and read across each row. Notice: sigmoid and tanh flatten (saturate) at the extremes; ReLU zeros the negatives hard; leaky lets a sliver through; GELU curves smoothly and dips slightly negative near $x=-1$. Seeing the actual numbers side by side is the fastest way to build intuition for how each one reshapes a signal.

---

## Why this matters for IOAI

- **The default that just works.** In the heat of a competition you don't want to agonize over activations. ReLU for hidden layers, softmax for classification output — knowing the defaults lets you spend your time on the parts that actually differ.
- **Diagnosing dead training.** If a network's loss plateaus and half its neurons output zero, you'll recognize *dying ReLU* and switch to Leaky ReLU — a fix that's invisible unless you understand activations.
- **Reading modern architectures.** The CV and NLP tracks are full of GELU. Knowing what it is (a smooth ReLU) means transformer code won't be a black box.
- **Output/loss pairing.** A shocking number of bugs come from mismatching output activation and loss (e.g. applying softmax twice, or using MSE on class labels). Understanding softmax vs sigmoid vs linear prevents these.
- **Gradient flow is the through-line.** The whole reason activations matter — vanishing vs flowing gradients — connects directly to why deep networks train or don't. That intuition pays off across every model you build.

---

## Summary

- Activation functions add the **non-linearity** that lets deep networks model complex functions, and their **derivative** controls gradient flow.
- **Sigmoid** $\frac{1}{1+e^{-x}}$: output in $(0,1)$; **saturates** (derivative ≤ 0.25) → vanishing gradients. Good for binary output, bad for deep hidden layers.
- **Tanh**: zero-centered version of sigmoid, output in $(-1,1)$; still saturates but less.
- **ReLU** $\max(0,x)$: the modern default — cheap, no positive-side vanishing (derivative 1), but suffers **dying ReLU** (zero gradient for negatives).
- **Leaky ReLU / ELU**: let a small negative slope through to keep neurons alive.
- **GELU**: smooth ReLU-like function; the standard in **transformers**.
- **Softmax**: turns a vector of logits into a **probability distribution** (sums to 1) for **multi-class** output; pairs with cross-entropy.
- **Choosing:** ReLU for hidden layers by default; match the output activation (sigmoid / softmax / linear) to the task.
