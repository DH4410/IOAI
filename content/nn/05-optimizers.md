---
title: "Optimizers: SGD, Adam, AdamW"
track: nn
order: 5
estimatedTime: 45
difficulty: intermediate
---

# Optimizers: SGD, Adam, AdamW

Backpropagation gives us the gradient — the direction that lowers the loss. An **optimizer** is the rule that decides *how to actually step* using that gradient. The naive rule "subtract the gradient times a learning rate" works, but it's slow and fragile. Modern optimizers like **Adam** make training dramatically faster and more robust — they're the reason you can train a big network in an afternoon instead of a week.

> **Big picture:** The optimizer turns gradients into weight updates. Better optimizers use the *history* of past gradients to move smarter — building momentum in consistent directions and adapting the step size per-parameter.

---

## The setting: gradient descent on a landscape

Picture the loss as a **landscape** — hills and valleys over the space of all weights. Training means walking downhill to find a low valley (small loss). The gradient at your feet tells you which way is steepest uphill; you step the opposite way.

$$w \leftarrow w - \eta \nabla L$$

Here $\nabla L$ (read "grad L") is the gradient — the vector of all $\frac{\partial L}{\partial w}$ — and $\eta$ is the **learning rate** (step size). This is **gradient descent**. Everything in this lesson is a smarter version of this one line.

The landscape is treacherous: long narrow valleys, flat plateaus, and saddle points. Plain gradient descent struggles with all of them. The optimizers below are tricks to navigate this terrain better.

---

## Vanilla SGD and why it's slow

**Stochastic Gradient Descent (SGD)** is gradient descent where each step uses the gradient from a small random **mini-batch** of examples rather than the whole dataset.

- "Gradient descent" (full-batch) computes the exact gradient over *all* data each step — accurate but slow, and needs all data in memory.
- **SGD** uses a mini-batch (say 32 or 256 examples) — the gradient is a noisy *estimate*, but you get many more, cheaper steps. The noise even helps escape bad spots.

$$w \leftarrow w - \eta \nabla L_{\text{batch}}$$

**Why plain SGD is slow and annoying:**

1. **One learning rate for everything.** Every weight takes the same-sized step. But some weights need big moves, others tiny ones — a single $\eta$ can't suit all.
2. **It zig-zags in ravines.** In a long narrow valley (steep across, gentle along), the gradient mostly points across the valley, so SGD bounces side to side while creeping slowly toward the bottom. Painfully inefficient.
3. **It stalls on plateaus.** Where the gradient is tiny, steps become tiny, and progress crawls.
4. **Learning-rate sensitivity.** Too big → it overshoots and diverges; too small → it takes forever. Finding the sweet spot is fiddly.

The next ideas — momentum and adaptive rates — each fix one of these problems, and Adam combines both.

```python
import numpy as np

def sgd_step(w, grad, lr=0.1):
    return w - lr * grad
```

---

## Momentum: build up speed in consistent directions

**Idea:** instead of stepping purely on the *current* gradient, keep a running "velocity" that accumulates past gradients — like a heavy ball rolling downhill. It builds speed in directions that consistently point downhill and smooths out the zig-zag.

$$v \leftarrow \beta v + \nabla L$$
$$w \leftarrow w - \eta v$$

**In English:** $v$ is the velocity. Each step, we keep a fraction $\beta$ (typically 0.9) of the old velocity and add the new gradient. Then we step by $-\eta v$ instead of $-\eta\nabla L$.

**Why this helps:**
- In a **consistent** direction (the gradient keeps pointing the same way), velocity *accumulates* and we accelerate — plateaus get crossed faster.
- In a **zig-zagging** direction (gradient flips back and forth), the opposing contributions **cancel out** in the running average, damping the oscillation.

The physical analogy: a ball with momentum rolls straight through small bumps and doesn't ricochet off the valley walls the way a memoryless walker does. With $\beta=0.9$, the velocity is roughly an average over the last ~10 gradients.

```python
import numpy as np

def momentum_init(shape):
    return np.zeros(shape)   # velocity starts at 0

def momentum_step(w, grad, v, lr=0.1, beta=0.9):
    v = beta * v + grad      # accumulate velocity
    w = w - lr * v           # step along velocity
    return w, v
```

---

## RMSProp: adaptive per-parameter learning rates

Momentum fixed the zig-zag; **RMSProp** fixes the "one learning rate for all weights" problem. It gives **each weight its own effective learning rate**, based on how big that weight's gradients have recently been.

$$s \leftarrow \beta s + (1-\beta)(\nabla L)^2$$
$$w \leftarrow w - \frac{\eta}{\sqrt{s} + \epsilon}\nabla L$$

**In English:** $s$ tracks a running average of the *squared* gradient for each weight (its recent "typical size"). We then divide the update by $\sqrt{s}$. So:
- Weights with **large, volatile** gradients get divided by a big number → **smaller** steps (calm them down).
- Weights with **small, steady** gradients get divided by a small number → **larger** steps (speed them up).

The tiny $\epsilon$ (like $10^{-8}$) just prevents division by zero. The result: every parameter automatically gets a step size suited to it, which is a huge deal — you no longer need one perfect global learning rate.

**Why square then square-root?** Squaring makes the measure of "size" ignore the sign (we care about magnitude of wiggle, not direction). The square-root brings it back to the same units as the gradient so the division is dimensionally sensible. In effect, $\sqrt{s}$ is the recent **root-mean-square** of the gradient — hence "RMS"Prop.

---

## Adam = Momentum + RMSProp

**Adam** (Adaptive Moment Estimation) is the most popular optimizer in deep learning, and it's simply the two good ideas combined: **momentum** (a running average of the gradient) *and* **RMSProp** (a running average of the squared gradient). It's the default you'll reach for 90% of the time.

Here is the full algorithm. For each parameter, keep two running averages $m$ and $v$, both starting at 0:

**Step 1 — momentum term (first moment, the mean of gradients):**
$$m \leftarrow \beta_1 m + (1-\beta_1)\nabla L$$
*In English:* an exponential moving average of the gradient — this is the momentum "velocity." $\beta_1$ is typically 0.9.

**Step 2 — RMSProp term (second moment, the mean of squared gradients):**
$$v \leftarrow \beta_2 v + (1-\beta_2)(\nabla L)^2$$
*In English:* an exponential moving average of the squared gradient — the per-parameter "typical size." $\beta_2$ is typically 0.999.

**Step 3 — bias correction** (important early on):
$$\hat{m} = \frac{m}{1-\beta_1^t}, \qquad \hat{v} = \frac{v}{1-\beta_2^t}$$
*In English:* because $m$ and $v$ start at zero, they're biased toward zero for the first several steps. Dividing by $(1-\beta^t)$ (where $t$ is the step number) corrects this so early updates aren't artificially tiny. As $t$ grows, $\beta^t \to 0$ and the correction fades away.

**Step 4 — the update:**
$$w \leftarrow w - \alpha\frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$$
*In English:* step in the direction of the momentum-smoothed gradient $\hat m$, but scale it per-parameter by dividing by $\sqrt{\hat v}$ — exactly RMSProp's adaptive rate. $\alpha$ is the base learning rate (often 0.001), $\epsilon \approx 10^{-8}$.

**Why Adam is so good:** it gets momentum's acceleration *and* smoothing, plus RMSProp's per-parameter adaptivity, plus bias correction for a clean start. In practice it converges fast, tolerates a wide range of learning rates, and needs little tuning — which is exactly what you want under competition pressure.

The default hyperparameters — $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$, $\alpha=0.001$ — work astonishingly often. Memorize them.

---

## numpy implementation of Adam

Let's implement Adam from scratch and use it to minimize a simple function so you can watch it work. We'll minimize $f(w) = w_1^2 + 10 w_2^2$ — a classic "narrow valley" where plain SGD zig-zags but Adam glides.

```python
import numpy as np

# The function to minimize and its gradient.
# f(w) = w0^2 + 10*w1^2  (a stretched bowl; minimum at (0,0))
def grad_f(w):
    return np.array([2 * w[0], 20 * w[1]])

# ---- Adam state ----
w = np.array([5.0, 5.0])      # starting point
m = np.zeros(2)               # first moment (momentum)
v = np.zeros(2)               # second moment (RMSProp)

alpha = 0.5                   # base learning rate
beta1, beta2 = 0.9, 0.999
eps = 1e-8

print(f"start: w = {w}, f = {w[0]**2 + 10*w[1]**2:.4f}")
for t in range(1, 51):
    g = grad_f(w)                          # gradient at current w

    m = beta1 * m + (1 - beta1) * g        # update momentum
    v = beta2 * v + (1 - beta2) * g**2     # update squared-grad average

    m_hat = m / (1 - beta1**t)             # bias correction
    v_hat = v / (1 - beta2**t)

    w = w - alpha * m_hat / (np.sqrt(v_hat) + eps)   # the Adam step

    if t % 10 == 0:
        f = w[0]**2 + 10*w[1]**2
        print(f"step {t:2d}: w = [{w[0]:7.4f}, {w[1]:7.4f}], f = {f:.6f}")

print(f"\nfinal: w = {w}  (true minimum is [0, 0])")
```

Run it and watch $w$ march smoothly to $(0,0)$. Because the second coordinate has a 10× steeper gradient, plain SGD would bounce wildly in $w_2$; Adam's per-parameter scaling automatically shrinks the $w_2$ step and enlarges the $w_1$ step, so both coordinates converge together. That automatic balancing is the everyday magic of Adam.

**Match the code to the math:**

| Code | Math |
|---|---|
| `m = beta1*m + (1-beta1)*g` | $m \leftarrow \beta_1 m + (1-\beta_1)\nabla L$ |
| `v = beta2*v + (1-beta2)*g**2` | $v \leftarrow \beta_2 v + (1-\beta_2)(\nabla L)^2$ |
| `m_hat = m/(1-beta1**t)` | $\hat m = m/(1-\beta_1^t)$ |
| `w -= alpha*m_hat/(sqrt(v_hat)+eps)` | $w \leftarrow w - \alpha\,\hat m/(\sqrt{\hat v}+\epsilon)$ |

---

## AdamW: decoupled weight decay

**Weight decay** is a regularization trick: gently pull weights toward zero each step so they don't grow too large (large weights often mean overfitting). The classic way is to add a penalty $\frac{\lambda}{2}\lVert w\rVert^2$ to the loss, whose gradient is $\lambda w$ — so it just adds $\lambda w$ to the gradient.

Here's the subtle problem: in Adam, that added $\lambda w$ term goes *through* the adaptive $\sqrt{\hat v}$ scaling. So weights with big gradients get *less* decay and weights with small gradients get *more* — the regularization gets tangled up with the adaptive rates in a way nobody intended. It makes weight decay behave inconsistently.

**AdamW** fixes this by **decoupling** weight decay from the gradient update. It does the normal Adam step, then *separately* shrinks the weights:

$$w \leftarrow w - \alpha\frac{\hat{m}}{\sqrt{\hat{v}}+\epsilon} - \alpha\lambda w$$

**In English:** first take the ordinary Adam step, then, as a *separate* operation, pull every weight a little toward zero by $\alpha\lambda w$. The decay is now clean and uniform — it doesn't get distorted by the adaptive scaling.

```python
# AdamW: same as Adam, plus a decoupled decay term applied to w directly.
def adamw_step(w, m, v, g, t, alpha=1e-3, beta1=0.9, beta2=0.999,
               eps=1e-8, wd=0.01):
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    # Adam step, then separate weight decay (this is the "W"):
    w = w - alpha * m_hat / (np.sqrt(v_hat) + eps) - alpha * wd * w
    return w, m, v
```

> **This matters right now:** AdamW is the standard optimizer for training and fine-tuning transformers (BERT, GPT, ViT). When you fine-tune a model in the CV or NLP track, you'll almost certainly use `torch.optim.AdamW`. It's Adam done right for large models.

---

## Learning rate schedules: warmup + cosine decay

The learning rate doesn't have to be constant. A **schedule** changes $\alpha$ over the course of training, and the right schedule can meaningfully improve results. The combo used in essentially every modern large-model recipe — and highly relevant at **IOAI** — is **warmup followed by cosine decay**.

**Warmup:** for the first few hundred/thousand steps, *ramp the learning rate up* from ~0 to its target value, linearly. Why? At the very start the weights are random and gradients can be wild; a big step immediately could blow up training. Warmup eases in gently until things stabilize.

**Cosine decay:** after warmup, *smoothly decrease* the learning rate along a cosine curve down toward ~0 by the end of training. Early on, big steps make fast progress; later, small steps let the model settle precisely into a good minimum. The cosine shape decays slowly at first, then faster, then levels off — empirically excellent.

$$\alpha_t = \alpha_{\max}\cdot\frac{1}{2}\left(1 + \cos\!\left(\pi\frac{t - t_{\text{warmup}}}{T - t_{\text{warmup}}}\right)\right)$$

*In English:* after warmup, the cosine term slides from 1 (at the start of decay) down to 0 (at the end), so $\alpha$ glides from $\alpha_{\max}$ to 0.

```python
import numpy as np

def lr_schedule(t, warmup=100, total=1000, max_lr=1e-3):
    if t < warmup:
        # linear warmup from 0 up to max_lr
        return max_lr * t / warmup
    # cosine decay from max_lr down to 0
    progress = (t - warmup) / (total - warmup)
    return max_lr * 0.5 * (1 + np.cos(np.pi * progress))

# Print the shape of the schedule
for t in [0, 50, 100, 300, 600, 900, 1000]:
    print(f"step {t:4d}: lr = {lr_schedule(t):.6f}")
# You'll see lr ramp up to ~1e-3 by step 100, then curve smoothly back to ~0.
```

Run it and read the numbers: the rate climbs during warmup, peaks, then eases back down. This warmup+cosine pattern is the default for training transformers and comes up constantly in competition-grade training recipes. Knowing it exists — and that it helps — can be the difference between a model that converges and one that diverges at IOAI.

---

## Quick comparison

| Optimizer | Key idea | Strength | Watch out |
|---|---|---|---|
| **SGD** | step down the gradient | simple, well-understood, great final generalization | slow, zig-zags, one LR for all |
| **SGD + Momentum** | accumulate velocity | accelerates, damps oscillation | still one global LR |
| **RMSProp** | per-parameter adaptive LR | handles varied gradient scales | no momentum |
| **Adam** | momentum + adaptive LR + bias correction | fast, robust, little tuning | can generalize slightly worse than tuned SGD |
| **AdamW** | Adam + decoupled weight decay | the standard for transformers | pick a sensible weight-decay value |

**Practical advice:** default to **Adam** (or **AdamW** if you're using weight decay / training a transformer) with $\alpha=10^{-3}$. If you have time to tune and want the best possible generalization on vision tasks, well-tuned **SGD+momentum** with a schedule can edge it out — but that's a luxury. Under time pressure, Adam is the safe, strong choice.

---

## Why this matters for IOAI

- **Adam/AdamW is your default weapon.** Nearly every model you train at IOAI will use it. Knowing the four-line update and the standard hyperparameters (0.9, 0.999, 1e-8, 1e-3) means you set it up correctly without googling.
- **Diagnosing training trouble.** Loss exploding? Lower the LR or add warmup. Loss stuck? Maybe the LR is too small or you need Adam's adaptivity. Understanding *why* each optimizer behaves as it does turns guesswork into diagnosis.
- **Schedules are competition-standard.** Warmup + cosine decay is used in the exact recipes IOAI problems are based on. Recognizing and applying it can directly boost your score.
- **AdamW for fine-tuning.** The CV and NLP tracks fine-tune big pretrained models; AdamW with weight decay is the standard, and now you know *why* it's "W" and not plain Adam.
- **Compute is limited.** A better optimizer/schedule means your model reaches a good solution in fewer steps — precious when you have a shared GPU and a ticking clock.

---

## Summary

- An **optimizer** converts gradients into weight updates. Base rule: $w \leftarrow w - \eta\nabla L$ (gradient descent).
- **SGD** uses noisy mini-batch gradients: cheap steps but slow, zig-zaggy, and one learning rate for all weights.
- **Momentum** ($v \leftarrow \beta v + \nabla L$) accumulates a velocity to accelerate and damp oscillations.
- **RMSProp** ($s \leftarrow \beta s + (1-\beta)(\nabla L)^2$) gives each weight an adaptive step size by dividing by $\sqrt{s}$.
- **Adam** = momentum + RMSProp + bias correction; update $w \leftarrow w - \alpha\frac{\hat m}{\sqrt{\hat v}+\epsilon}$. Defaults: $\beta_1=0.9,\ \beta_2=0.999,\ \epsilon=10^{-8},\ \alpha=10^{-3}$.
- **AdamW** decouples weight decay from the adaptive update — the standard for transformers.
- **Warmup + cosine decay** schedules the learning rate: ramp up to avoid early instability, then smoothly decay to settle into a good minimum. Ubiquitous in modern training.
