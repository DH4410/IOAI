---
title: Derivatives and Gradients
track: math
order: 3
estimatedTime: 55
difficulty: intermediate
---

# Derivatives and Gradients

Every time a neural network learns, it does one thing over and over: it computes **gradients** and takes a small step. That's training. That's it. The famous "backpropagation" algorithm is, at its core, a big organized way of computing gradients. So if you understand what a gradient is and why it points the way it does, you understand the mechanism behind all of deep learning.

The good news: a gradient is built out of **derivatives**, and a derivative is a simple, intuitive idea. We'll start there — no scary limits, just a clear picture of "rate of change" — and build up step by step to gradient descent, the algorithm that trains essentially every modern AI model.

Take your time with this one. It's the most important math lesson for actually understanding how learning works.

---

## What is a derivative? Rate of change

Forget formulas for a second. A derivative answers a very practical question:

> **If I nudge the input a tiny bit, how much does the output change?**

That's it. A derivative is a **rate of change** — a sensitivity. "How fast is this changing?"

Everyday examples you already understand:
- **Speed** is a derivative. It's how fast your *position* changes as *time* passes. Going 60 km/h means "each hour, my position changes by 60 km."
- The **slope of a hill** is a derivative. It's how much your *height* changes as you walk *forward*. A steep hill has a big slope; flat ground has slope zero.

That second one — slope — is the picture to keep in your head. **A derivative is the slope of a function.**

### Slope, precisely

Imagine a function $f(x)$ drawn as a curve. Pick a point on it. The derivative at that point, written $f'(x)$ or $\frac{df}{dx}$, is the **slope of the curve right there** — how steeply it's going up or down.

- Slope **positive** → the function is going *up* as $x$ increases.
- Slope **negative** → the function is going *down* as $x$ increases.
- Slope **zero** → the function is flat right there (the top of a hill or bottom of a valley).

That last one — slope zero at the bottom of a valley — is going to be the whole key to learning. Hold onto it.

### The definition (for completeness)

Formally, the derivative is what the slope becomes as you shrink the step size $h$ down to nearly nothing:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

Read it in plain words: "take a tiny step $h$, see how much $f$ changed, divide by the step size to get a rate, and imagine the step getting infinitely small." The fraction $\frac{f(x+h)-f(x)}{h}$ is just *rise over run* — the slope between two nearby points. Shrinking $h$ makes it the slope *at* the point.

You'll rarely compute this limit by hand. Instead you'll use a few **rules** that tell you the derivative directly. Let's learn them.

---

## Derivative rules you actually need

### The power rule

For a function $f(x) = x^n$, the derivative is:

$$\frac{d}{dx}x^n = n\,x^{n-1}$$

In words: bring the exponent down in front, then subtract one from the exponent. A few examples:

- $f(x) = x^2 \Rightarrow f'(x) = 2x$
- $f(x) = x^3 \Rightarrow f'(x) = 3x^2$
- $f(x) = x^1 = x \Rightarrow f'(x) = 1$ (a straight line with slope 1)
- $f(x) = x^0 = 1 \Rightarrow f'(x) = 0$ (a constant is flat — slope zero everywhere)

That last point generalizes: **the derivative of any constant is 0.** A constant doesn't change, so its rate of change is zero. Makes sense.

### A couple more useful ones

- Constant multiplier stays put: $\frac{d}{dx}[c \cdot f(x)] = c \cdot f'(x)$. So $\frac{d}{dx}(5x^2) = 5 \cdot 2x = 10x$.
- Derivatives add up: $\frac{d}{dx}[f + g] = f' + g'$. So $\frac{d}{dx}(x^2 + x^3) = 2x + 3x^2$.
- The exponential is its own derivative: $\frac{d}{dx}e^x = e^x$. (This is why $e$ shows up everywhere.)
- Natural log: $\frac{d}{dx}\ln x = \frac{1}{x}$. (Shows up constantly in loss functions.)

Let's verify the power rule numerically. We'll compute the derivative of $x^2$ at $x = 3$. By the rule it should be $2 \times 3 = 6$.

```python
def f(x):
    return x**2

x = 3.0
h = 1e-6                          # a tiny step
slope = (f(x + h) - f(x)) / h     # rise over run
print(slope)                      # about 6.0  -> matches 2*x = 6
```

We just used the definition directly: nudge $x$ by a tiny $h$, see how much $f$ moved, divide. This is called a **numerical derivative**, and it's a lifesaver for checking your math — more on that at the end.

### The chain rule — the one backprop is built on

This is the most important rule for AI, so read it twice. The **chain rule** handles *functions inside functions*.

Suppose $y$ depends on $u$, and $u$ depends on $x$. So changing $x$ changes $u$, which changes $y$. How sensitive is $y$ to $x$? The chain rule says: **multiply the sensitivities along the chain.**

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

The intuition is wonderfully physical. Imagine three gears meshed together. Gear $x$ turns gear $u$, and gear $u$ turns gear $y$. If turning $x$ makes $u$ spin **2×** as fast, and turning $u$ makes $y$ spin **3×** as fast, then turning $x$ makes $y$ spin **6× = 2 × 3** as fast. Sensitivities multiply down the chain.

**Example.** Let $y = (x^2 + 1)^3$. This is an outer function (something cubed) wrapped around an inner function ($x^2 + 1$). Set $u = x^2 + 1$, so $y = u^3$.

- Outer: $\frac{dy}{du} = 3u^2 = 3(x^2+1)^2$
- Inner: $\frac{du}{dx} = 2x$
- Multiply: $\frac{dy}{dx} = 3(x^2+1)^2 \cdot 2x = 6x(x^2+1)^2$

```python
def f(x):
    return (x**2 + 1)**3

# chain rule says derivative is 6*x*(x^2+1)^2
def deriv_analytic(x):
    return 6*x*(x**2 + 1)**2

x = 2.0
h = 1e-6
numeric = (f(x+h) - f(x)) / h
print("numeric :", numeric)           # ~600
print("analytic:", deriv_analytic(x)) # 600.0
```

Why does AI care so much about the chain rule? Because **a neural network is functions inside functions inside functions** — layer 1 feeds layer 2 feeds layer 3, dozens deep. To find how the final loss depends on a weight buried in layer 1, you multiply sensitivities backward through every layer. That backward multiplication *is* backpropagation. The chain rule is the engine; backprop is the chain rule applied systematically to a deep stack of functions.

---

## Partial derivatives: many inputs

So far our functions had one input, $x$. But real models have *millions* of inputs (weights). A loss function might be $L(w_1, w_2, w_3, \ldots)$ depending on all of them. How do we take a derivative when there are many inputs?

Answer: one at a time. A **partial derivative** is the derivative with respect to *one* variable, **treating all the others as constants** (frozen). We swap the straight $d$ for a curly $\partial$ (called "del" or "partial") to signal "this is a partial derivative."

$$\frac{\partial f}{\partial x} \quad\text{means: derivative of } f \text{ with respect to } x, \text{ holding everything else fixed.}$$

The intuition: you're standing on a hilly landscape (height depends on your east–west position *and* your north–south position). $\frac{\partial f}{\partial x}$ asks: "if I take a tiny step *east only*, how much does my height change?" You freeze north–south and just wiggle east.

**Example.** Let $f(x, y) = x^2 + 3xy + y^2$.

To get $\frac{\partial f}{\partial x}$, treat $y$ as a constant:
- $x^2 \to 2x$
- $3xy \to 3y$ (since $y$ is a constant, $3xy$ is just "constant times $x$")
- $y^2 \to 0$ (a constant, as far as $x$ is concerned)

So $\frac{\partial f}{\partial x} = 2x + 3y$.

To get $\frac{\partial f}{\partial y}$, treat $x$ as a constant:
- $x^2 \to 0$
- $3xy \to 3x$
- $y^2 \to 2y$

So $\frac{\partial f}{\partial y} = 3x + 2y$.

```python
def f(x, y):
    return x**2 + 3*x*y + y**2

x, y, h = 1.0, 2.0, 1e-6

# partial wrt x: wiggle x only
df_dx = (f(x+h, y) - f(x, y)) / h
# partial wrt y: wiggle y only
df_dy = (f(x, y+h) - f(x, y)) / h

print("df/dx:", df_dx, " expected 2x+3y =", 2*x + 3*y)   # ~8
print("df/dy:", df_dy, " expected 3x+2y =", 3*x + 2*y)   # ~7
```

Nothing new here — a partial derivative is an ordinary derivative where you've mentally frozen the other variables. That's the only trick.

---

## The gradient: all the partials together

Now we assemble the star of the show. The **gradient** of a function is simply *the vector of all its partial derivatives*. We write it with the symbol $\nabla$ (called "nabla" or "del"):

$$\nabla f = \begin{bmatrix} \dfrac{\partial f}{\partial x_1} \\[6pt] \dfrac{\partial f}{\partial x_2} \\[4pt] \vdots \\[2pt] \dfrac{\partial f}{\partial x_n} \end{bmatrix}$$

For our $f(x, y) = x^2 + 3xy + y^2$, the gradient is:

$$\nabla f = \begin{bmatrix} 2x + 3y \\ 3x + 2y \end{bmatrix}$$

So at the point $(1, 2)$, the gradient is $\begin{bmatrix} 2(1)+3(2) \\ 3(1)+2(2) \end{bmatrix} = \begin{bmatrix} 8 \\ 7 \end{bmatrix}$.

### The one fact that makes gradients magical

Here's the property that makes the gradient the beating heart of machine learning:

> **The gradient points in the direction of steepest increase.** It's an arrow that points *uphill*, straight up the steepest slope. And its length tells you *how steep* that slope is.

Picture yourself on a hillside in fog. You can't see the whole landscape, but you can feel the ground under your feet. The gradient is the arrow saying "THIS way is the steepest way up, and here's how steep it is." Walk along the gradient and you climb fastest.

Now — for machine learning we don't want to climb. We want to *descend*. We want to make the loss (the error) as small as possible. So we go the **opposite** direction: $-\nabla f$, the **negative** gradient, points the steepest way *downhill*. That's the direction that shrinks the loss fastest. Remember this; it's the next section entirely.

```python
import numpy as np

def grad_f(x, y):
    return np.array([2*x + 3*y, 3*x + 2*y])

print(grad_f(1, 2))   # [8 7]  -> the uphill direction at (1,2)
# To go downhill (reduce f), we'd step toward -[8, 7] = [-8, -7]
```

---

## Gradient descent: walk downhill

We finally arrive at the algorithm that trains basically every AI model. The setup:

- We have a **loss function** $L$ that measures how *wrong* the model is. High loss = bad predictions. Low loss = good predictions.
- $L$ depends on the model's **weights** $w$ (there might be billions of them).
- **Goal:** find the weights that make $L$ as small as possible — the bottom of the valley.

We can't just solve for the minimum directly (the function is far too complicated). So we do something cleverly simple: **start somewhere, and repeatedly step downhill.**

That's gradient descent. Here's the rule, the single most important equation in this course:

$$w \leftarrow w - \alpha \frac{\partial L}{\partial w}$$

Or in vector form for all the weights at once:

$$\mathbf{w} \leftarrow \mathbf{w} - \alpha \, \nabla L$$

Let's decode it:
- $\nabla L$ (or $\frac{\partial L}{\partial w}$) is the gradient — the *uphill* direction of the loss.
- The **minus sign** flips it to point *downhill*. We subtract the gradient so we move toward smaller loss.
- $\alpha$ (alpha) is the **learning rate** — the size of each step. A small positive number like 0.01.
- $\leftarrow$ means "update": compute the right side, make it the new $w$.

In plain English: **look at which way is downhill, take a small step that way, repeat.** Do this thousands of times and you slide down to the bottom of the valley — the weights with low loss, i.e. a trained model.

### The learning rate: not too big, not too small

The learning rate $\alpha$ controls step size, and picking it well matters a lot:

- **Too small** → you creep downhill agonizingly slowly. Training takes forever.
- **Too big** → you overshoot the valley, bounce to the other side, maybe bounce even higher, and *diverge* (the loss blows up to infinity). Disaster.
- **Just right** → smooth, steady descent to the bottom.

Tuning the learning rate is one of the first things you do when training any model. We'll return to it in the ML track.

### Watching it work

Let's minimize a simple function, $f(w) = (w - 3)^2$. By eye, the minimum is obviously at $w = 3$ (that's where the squared term is zero). Its derivative is $f'(w) = 2(w - 3)$. Let's let gradient descent *find* $w = 3$ on its own, starting from a random spot.

```python
def f(w):
    return (w - 3)**2

def grad(w):
    return 2*(w - 3)          # derivative of (w-3)^2

w = 0.0        # start far from the answer
lr = 0.1       # learning rate

for step in range(30):
    g = grad(w)              # which way is uphill?
    w = w - lr * g           # step downhill
    if step % 5 == 0:
        print(f"step {step:2d}: w = {w:.4f}, loss = {f(w):.4f}")

print("final w:", round(w, 4))   # very close to 3.0
```

Run it and watch `w` march from 0 toward 3, and the loss shrink toward 0. **No one told the algorithm the answer was 3.** It found it by feeling the slope and stepping downhill, over and over. That is machine learning in miniature. A real model does the exact same loop — just with millions of weights and a loss computed over real data.

### Why this matters for AI: this IS training

When you hear "the model is training," here's what's literally happening:

1. Feed data through the model, get predictions.
2. Compute the loss (how wrong the predictions are).
3. Compute the gradient of the loss with respect to every weight — **this is backpropagation**, the chain rule applied backward through all the layers.
4. Update every weight by stepping downhill: $\mathbf{w} \leftarrow \mathbf{w} - \alpha \nabla L$.
5. Repeat millions of times.

Steps 3 and 4 are the derivative and gradient material from this whole lesson. Backprop = computing gradients (chain rule). The update = gradient descent. There is no other magic. The reason models can learn anything at all is this humble loop of "feel the slope, step downhill."

---

## Numerical gradient checking: a real-world superpower

When you code up a gradient by hand (deriving it with the rules), it's *easy* to make a mistake — a sign error, a forgotten factor. Here's the trick professionals use to catch those bugs: compare your hand-derived gradient against a **numerical** one computed straight from the definition.

The numerical gradient uses the "nudge and measure" idea, but a slightly better version called the **central difference** — nudge both up *and* down and use the two-sided slope. It's more accurate:

$$\frac{\partial f}{\partial x} \approx \frac{f(x + h) - f(x - h)}{2h}$$

If your analytic gradient matches this numerical one, you can trust your math. If they disagree, you have a bug. This "gradient check" has saved countless researchers from silently broken models.

```python
import numpy as np

def f(v):
    x, y = v
    return x**2 + 3*x*y + y**2

def analytic_grad(v):
    x, y = v
    return np.array([2*x + 3*y, 3*x + 2*y])

def numerical_grad(f, v, h=1e-5):
    grad = np.zeros_like(v, dtype=float)
    for i in range(len(v)):
        step = np.zeros_like(v, dtype=float)
        step[i] = h
        grad[i] = (f(v + step) - f(v - step)) / (2*h)   # central difference
    return grad

point = np.array([1.0, 2.0])
print("analytic :", analytic_grad(point))              # [8. 7.]
print("numerical:", numerical_grad(f, point))          # [8. 7.]

# Are they close? (Never test floats with ==; test closeness.)
diff = np.max(np.abs(analytic_grad(point) - numerical_grad(f, point)))
print("max difference:", diff)                         # tiny, ~1e-9
print("gradient check passed:", diff < 1e-5)           # True
```

The numerical gradient is too *slow* to use for actual training (it needs two function evaluations per weight, and models have billions of weights — that's why we use backprop instead). But as a **checking tool** during development, it's gold. Learn this pattern; you'll use it for real.

---

## Sort Calculus Concepts

```widget
{
  "type": "concept-sort",
  "title": "Derivative or Gradient? Match the Description",
  "categories": [
    { "name": "Derivative (1D)", "color": "#5B5BD6" },
    { "name": "Gradient (multi-D)", "color": "#F97316" }
  ],
  "items": [
    { "text": "df/dx — one input, one output", "category": "Derivative (1D)" },
    { "text": "∇f = [∂f/∂x₁, ∂f/∂x₂, ...] — vector of partial derivatives", "category": "Gradient (multi-D)" },
    { "text": "Tells you the slope at a point on a curve", "category": "Derivative (1D)" },
    { "text": "Points in the direction of steepest ascent", "category": "Gradient (multi-D)" },
    { "text": "Chain rule: d/dx f(g(x)) = f'(g(x))·g'(x)", "category": "Derivative (1D)" },
    { "text": "Used in backpropagation for all model weights", "category": "Gradient (multi-D)" }
  ]
}
```

---

## Practice Questions

**Quick check:** The loss function is L(w) = (w - 3)². What is dL/dw, and what weight value minimizes it?
> dL/dw = 2(w - 3). Set to zero: 2(w - 3) = 0 → **w = 3** minimizes the loss. The gradient is zero at the minimum, and gradient descent would converge there.

**Quick check:** During backpropagation through a sigmoid activation, what problem arises and why?
> The sigmoid derivative σ'(x) = σ(x)(1-σ(x)) ≤ 0.25. Multiplying many such values together (through deep layers) makes the gradient exponentially small — the **vanishing gradient** problem. This is why ReLU (derivative = 1 for x > 0) is preferred in deep networks.

**Quick check:** Gradient descent updates weights as w ← w − α·∇L. If α is too large, what happens?
> The steps overshoot the minimum — the loss bounces up and down and may diverge to infinity (NaN). The right α makes steady progress toward the minimum without overshooting.

**Quick check:** Apply the chain rule: if L = (a - y)² and a = w·x, what is dL/dw?
> By chain rule: dL/dw = (dL/da)·(da/dw) = 2(a-y)·x = **2(wx - y)·x**. This is the gradient of MSE with respect to a weight — the core update rule for linear regression.

---

## Summary

- A **derivative** is a *rate of change* — the slope of a function. Positive slope = going up, negative = going down, zero = flat (a peak or valley bottom).
- Key rules: **power rule** $\frac{d}{dx}x^n = nx^{n-1}$, the derivative of a constant is 0, and the **chain rule** $\frac{dy}{dx} = \frac{dy}{du}\frac{du}{dx}$ (multiply sensitivities through nested functions). The chain rule is what backpropagation is built on.
- A **partial derivative** $\frac{\partial f}{\partial x}$ differentiates with respect to one variable while freezing the others.
- The **gradient** $\nabla f$ is the vector of all partial derivatives. It points in the direction of **steepest increase**, and its length is the steepness. The *negative* gradient points steepest downhill.
- **Gradient descent** trains models by stepping downhill on the loss: $\mathbf{w} \leftarrow \mathbf{w} - \alpha\nabla L$. The **learning rate** $\alpha$ sets the step size — too big diverges, too small crawls.
- **Training a neural network = gradient descent + backpropagation.** Backprop computes the gradients (chain rule), the update rule steps downhill. This loop is the entire mechanism behind how AI learns.
- **Gradient checking** compares your hand-derived gradient to a numerical one (central difference) to catch bugs — a real practical superpower.
