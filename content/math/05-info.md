---
title: Entropy and Cross-Entropy
track: math
order: 5
estimatedTime: 45
difficulty: intermediate
---

# Entropy and Cross-Entropy

Here's a question that sounds philosophical but has a precise mathematical answer: **how do you measure surprise?** How do you measure information? How much do you *learn* when you find out the result of a coin flip versus finding out the result of a rigged coin that always lands heads?

The answers to these questions form **information theory**, invented by Claude Shannon in 1948. And it turns out these ideas are not just elegant — they're the mathematical foundation of how we train classifiers. The loss function used to train almost every classification model in the world, **cross-entropy**, comes straight out of this lesson. When you finish, you'll understand *why* that loss has the shape it does, instead of it being a magic formula from a library.

Let's build it up from the most basic idea: surprise.

---

## Information = surprise

Start with intuition. Which of these tells you more?

- "The sun rose this morning." → You already knew this. Zero surprise. Learned nothing.
- "It snowed in the Sahara today." → Shocking! Very surprising. You learned a lot.

The pattern: **rare events are surprising and carry a lot of information; common events are unsurprising and carry little.** Information and surprise are the same thing, and both are tied to how *improbable* an event is.

So we want a formula where:
- If $p$ (the probability of the event) is close to 1 → information near 0 (no surprise).
- If $p$ is close to 0 → information is huge (big surprise).

The function that does this beautifully is the **negative logarithm**. The information content (or "surprisal") of an event with probability $p$ is:

$$I(x) = -\log_2 p(x)$$

Let's sanity-check it against our intuition:
- $p = 1$ (certain): $-\log_2 1 = -0 = 0$. Zero surprise. Correct!
- $p = 0.5$ (a coin flip): $-\log_2 0.5 = -(-1) = 1$. Exactly 1 unit of information.
- $p = 0.01$ (rare): $-\log_2 0.01 \approx 6.64$. Lots of information. Correct!

The unit here, when we use $\log_2$ (log base 2), is the **bit**. One bit is the information in a single fair coin flip — the answer to one yes/no question. This is *the* bit, the same one in "bits and bytes." A fair coin flip carries exactly one bit; that's not a coincidence, it's the definition.

```python
import numpy as np

def surprisal_bits(p):
    return -np.log2(p)

print(surprisal_bits(1.0))    # 0.0   -> certain, no surprise
print(surprisal_bits(0.5))    # 1.0   -> one fair coin flip = 1 bit
print(surprisal_bits(0.01))   # ~6.64 -> rare event, very informative
```

### Why the logarithm?

Two reasons the log is the *right* choice, not an arbitrary one:

1. **It matches surprise.** As $p$ goes from 1 down to 0, $-\log p$ climbs smoothly from 0 up to infinity. Rare = surprising, exactly as we wanted.
2. **Information adds up.** If you flip two independent coins, the probability of a specific pair is $0.5 \times 0.5 = 0.25$ (probabilities multiply). But intuitively you learned *two* bits — one per coin. The log turns that multiplication into addition: $-\log_2(0.25) = -\log_2(0.5) - \log_2(0.5) = 1 + 1 = 2$ bits. The log is the unique function that makes independent information *add*. That additivity is exactly why it's the natural measure.

---

## Entropy: average surprise

Surprisal measures one specific outcome. **Entropy** measures the *whole distribution* — it's the **average surprise** you expect across all possible outcomes. In other words, it's the average amount of information per event, or how *uncertain* the distribution is overall.

To get an average, weight each outcome's surprisal by how often it happens (its probability) and sum:

$$H(p) = -\sum_i p_i \log_2 p_i$$

Read it as: for each outcome $i$, take its surprisal $-\log_2 p_i$, weight it by its probability $p_i$, and add them all up. That's an expected value — the expected surprise.

### What entropy measures: uncertainty

Entropy is highest when you're *most uncertain* and lowest when you're *most certain*. Let's see it clearly with a coin.

**A fair coin** ($p = 0.5$ each side):
$$H = -(0.5\log_2 0.5 + 0.5\log_2 0.5) = -(0.5 \cdot (-1) + 0.5 \cdot (-1)) = 1 \text{ bit}$$
Maximum uncertainty for a coin — you truly can't predict it. Entropy = 1 bit.

**A totally rigged coin** ($p = 1$ heads, $0$ tails):
$$H = -(1 \cdot \log_2 1 + 0 \cdot \log_2 0) = -(0 + 0) = 0 \text{ bits}$$
(We take $0 \cdot \log 0 = 0$ by convention.) Zero uncertainty — you *know* it's heads. Entropy = 0. You learn nothing from the flip.

**A biased coin** ($p = 0.9$ heads): entropy comes out to about 0.47 bits — between the two. Somewhat predictable, so less than the fair coin's full bit, but not zero.

```python
import numpy as np

def entropy_bits(probs):
    probs = np.array(probs)
    probs = probs[probs > 0]              # skip zeros (0*log0 = 0)
    return -np.sum(probs * np.log2(probs))

print(entropy_bits([0.5, 0.5]))    # 1.0   -> fair coin, max uncertainty
print(entropy_bits([1.0, 0.0]))    # 0.0   -> rigged coin, no uncertainty
print(entropy_bits([0.9, 0.1]))    # ~0.47 -> biased, in between
print(entropy_bits([0.25]*4))      # 2.0   -> 4 equal outcomes = 2 bits
```

Notice the last one: 4 equally-likely outcomes have entropy 2 bits. Makes sense — you'd need 2 yes/no questions to pin down one of four things. **Entropy tells you the minimum number of bits needed, on average, to describe outcomes from a distribution.** A predictable distribution is cheap to describe (low entropy); a chaotic one is expensive (high entropy).

> **The big takeaway:** entropy measures uncertainty. Peaked, confident distributions have low entropy. Flat, spread-out distributions have high entropy. Maximum entropy = maximum ignorance = the uniform distribution.

---

## Cross-entropy: comparing two distributions

Now the piece that makes this all matter for AI. Cross-entropy compares **two** distributions: the *true* distribution $p$ (reality) and a *predicted* distribution $q$ (your model's guess).

$$H(p, q) = -\sum_i p_i \log q_i$$

Look carefully at the difference from plain entropy. Entropy was $-\sum p_i \log p_i$ — using $p$ everywhere. Cross-entropy is $-\sum p_i \log q_i$ — it weights by the *true* probabilities $p_i$ but takes the surprisal $-\log q_i$ from the *model's* probabilities $q_i$.

The interpretation: **cross-entropy is the average surprise you actually experience when reality follows $p$ but you were expecting $q$.** If your predictions $q$ match reality $p$ perfectly, cross-entropy is as low as it can be (it equals the entropy of $p$). The more your predictions are *wrong* — the more confidently you bet on things that don't happen — the higher the cross-entropy climbs. It punishes confident mistakes brutally.

Here's the key property that makes it a great loss function:

- Predict the truth confidently and correctly → **low** cross-entropy. 
- Predict confidently and *wrongly* → **huge** cross-entropy (approaching infinity as you put probability near 0 on the thing that actually happens).

That asymmetry — mild reward for being right, savage penalty for being confidently wrong — is exactly what you want to push a model toward honest, calibrated predictions.

---

## Cross-entropy as a loss function

This is the payoff. Cross-entropy is the loss function used to train nearly every classification model. Here's how it works in practice.

In classification, the *true* distribution $p$ is usually a **one-hot** vector — all the probability on the correct class, zero on the rest. If the true label is "cat" out of {cat, dog, bird}, then $p = [1, 0, 0]$.

The model outputs a predicted distribution $q$, like $q = [0.7, 0.2, 0.1]$ (70% cat, 20% dog, 10% bird).

Cross-entropy: $H(p, q) = -\sum_i p_i \log q_i$. But since $p$ is one-hot (all zeros except a 1 on the true class), *every term vanishes except the true-class term*. The whole sum collapses to just:

$$\text{loss} = -\log q_{\text{true class}}$$

Beautifully simple! The cross-entropy loss for one example is just **the negative log of the probability the model assigned to the correct answer.** Let's feel out its behavior:

- Model says 0.99 for the true class → loss $= -\log(0.99) \approx 0.01$. Tiny. Great job.
- Model says 0.5 for the true class → loss $= -\log(0.5) \approx 0.69$. Meh, uncertain.
- Model says 0.01 for the true class → loss $= -\log(0.01) \approx 4.6$. Huge! You were confidently wrong.

The loss shoots toward infinity as the model's probability on the right answer approaches 0. That steep penalty is what drives learning: being confidently wrong hurts a *lot*, so gradient descent works hard to fix it.

```python
import numpy as np

def cross_entropy(true_onehot, predicted_probs):
    predicted_probs = np.clip(predicted_probs, 1e-12, 1.0)   # avoid log(0)
    return -np.sum(true_onehot * np.log(predicted_probs))

true = np.array([1, 0, 0])          # the true class is index 0 (cat)

good = np.array([0.9, 0.05, 0.05])  # confident and correct
meh  = np.array([0.4, 0.3, 0.3])    # unsure
bad  = np.array([0.01, 0.495, 0.495])  # confident but WRONG

print("good:", round(cross_entropy(true, good), 3))   # ~0.105
print("meh :", round(cross_entropy(true, meh), 3))    # ~0.916
print("bad :", round(cross_entropy(true, bad), 3))    # ~4.605
```

Notice we `clip` the probabilities away from exactly 0 before taking the log. That's a crucial practical detail: $\log(0) = -\infty$ would crash your training. Real ML libraries handle this internally (and more cleverly), but the danger is real.

### Log loss / binary cross-entropy

For **binary** classification (two classes, like spam/not-spam), cross-entropy has a special two-term form called **log loss** or **binary cross-entropy**. If $y$ is the true label (0 or 1) and $\hat{y}$ is the model's predicted probability of class 1:

$$\text{loss} = -[\,y\log\hat{y} + (1 - y)\log(1 - \hat{y})\,]$$

It looks like two pieces, but only one is ever active: if $y = 1$, the second term dies and you get $-\log\hat{y}$; if $y = 0$, the first term dies and you get $-\log(1 - \hat{y})$. Either way it's "negative log of the probability you gave to the true answer" — exactly the same idea as before, just written for the two-class case. You'll meet this again when we train logistic regression.

```python
import numpy as np

def log_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

y_true = np.array([1, 0, 1, 1, 0])
good_preds = np.array([0.9, 0.1, 0.8, 0.95, 0.05])   # mostly right
bad_preds  = np.array([0.1, 0.9, 0.2, 0.05, 0.95])   # mostly wrong

print("good log loss:", round(log_loss(y_true, good_preds), 3))   # low
print("bad  log loss:", round(log_loss(y_true, bad_preds), 3))    # high
```

Over a whole dataset you **average** the per-example losses. That average log loss is precisely the number gradient descent minimizes when training a classifier. The whole pipeline — model outputs probabilities, cross-entropy scores them, gradient descent shrinks the loss — is grounded in this one idea from information theory.

---

## KL divergence: the extra cost of being wrong

There's one more concept worth knowing, because it appears all over modern AI (VAEs, diffusion models, reinforcement learning from human feedback, knowledge distillation). It's the **KL divergence**, written $D_{KL}(p \| q)$.

KL divergence measures how *different* two distributions are — specifically, the **extra** surprise you suffer by using the wrong distribution $q$ when reality is really $p$. It's just cross-entropy minus entropy:

$$D_{KL}(p \| q) = H(p, q) - H(p) = \sum_i p_i \log\frac{p_i}{q_i}$$

Think of it this way:
- $H(p)$ — the surprise you'd have *unavoidably*, even with a perfect model (the true uncertainty in reality).
- $H(p, q)$ — the surprise you *actually* suffer using your imperfect model $q$.
- $D_{KL}$ — the **wasted** surprise, the penalty purely for your model being wrong. The gap between your model and perfection.

Key properties:
- $D_{KL}(p \| q) \ge 0$ always. It's zero **only** when $q = p$ exactly (a perfect model wastes nothing).
- It's **not symmetric**: $D_{KL}(p \| q) \ne D_{KL}(q \| p)$ in general. "How surprised is $p$ by $q$" differs from "how surprised is $q$ by $p$." So it's a "divergence," not a true distance.

Here's a neat connection: since $H(p)$ (the entropy of the true labels) is a fixed constant that doesn't depend on your model, **minimizing cross-entropy is exactly the same as minimizing KL divergence.** When you train a classifier by minimizing cross-entropy, you are secretly pulling the model's distribution $q$ as close as possible to the true distribution $p$. That's a lovely unifying idea: training = shrinking the KL divergence between your model and reality.

```python
import numpy as np

def kl_divergence(p, q):
    p, q = np.array(p, dtype=float), np.array(q, dtype=float)
    q = np.clip(q, 1e-12, 1.0)
    mask = p > 0                           # 0*log0 = 0
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))

p = np.array([0.5, 0.5])
print(kl_divergence(p, [0.5, 0.5]))   # 0.0   -> identical, no waste
print(kl_divergence(p, [0.9, 0.1]))   # >0    -> q is different from p
print(kl_divergence(p, [0.99, 0.01])) # even bigger -> q is very wrong
```

---

## Sort the Information Theory Concepts

```widget
{
  "type": "concept-sort",
  "title": "Entropy, Cross-Entropy, or KL Divergence?",
  "categories": [
    { "name": "Entropy H(p)", "color": "#5B5BD6" },
    { "name": "Cross-Entropy H(p,q)", "color": "#F97316" },
    { "name": "KL Divergence D_KL(p||q)", "color": "#22C55E" }
  ],
  "items": [
    { "text": "Measures uncertainty of a single distribution", "category": "Entropy H(p)" },
    { "text": "The standard classification loss function", "category": "Cross-Entropy H(p,q)" },
    { "text": "Always ≥ 0, equals 0 when distributions match", "category": "KL Divergence D_KL(p||q)" },
    { "text": "Fair coin flip = 1 bit", "category": "Entropy H(p)" },
    { "text": "-log(probability of correct class)", "category": "Cross-Entropy H(p,q)" },
    { "text": "H(p,q) - H(p): the extra cost of being wrong", "category": "KL Divergence D_KL(p||q)" }
  ]
}
```

---

## Summary

- **Information / surprisal** of an event is $-\log_2 p$. Rare events (small $p$) are surprising and carry lots of information; certain events ($p = 1$) carry none. Using $\log_2$, the unit is the **bit** — one fair coin flip = 1 bit. The log is chosen because it makes independent information *add*.
- **Entropy** $H(p) = -\sum_i p_i \log p_i$ is the *average* surprise of a distribution — a measure of **uncertainty**. A fair coin has entropy 1 bit (max uncertainty); a rigged coin has 0 (no uncertainty). Flat distributions have high entropy, peaked ones low.
- **Cross-entropy** $H(p, q) = -\sum_i p_i \log q_i$ compares a true distribution $p$ to a predicted one $q$. It's the average surprise when reality is $p$ but you expected $q$ — low when predictions match reality, huge when you're confidently wrong.
- As a **loss function** with one-hot labels, cross-entropy collapses to $-\log q_{\text{true class}}$: the negative log-probability of the correct answer. This is the standard classification loss. **Binary cross-entropy / log loss** is the two-class version. Always guard against $\log(0)$ by clipping.
- **KL divergence** $D_{KL}(p\|q) = H(p,q) - H(p) \ge 0$ measures the *extra* surprise from using the wrong distribution — how far your model is from reality. Minimizing cross-entropy is equivalent to minimizing KL divergence, so training a classifier pulls its distribution toward the truth.
