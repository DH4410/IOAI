---
title: Probability Basics
track: math
order: 4
estimatedTime: 50
difficulty: intermediate
---

# Probability Basics

Every modern AI model speaks the language of probability. When a classifier says "I'm 87% sure this is a cat," that 87% is a probability. When a language model picks the next word, it's sampling from a probability distribution over the whole vocabulary. When a spam filter flags an email, it's computing the probability that the email is spam given the words in it. Probability is the *native tongue* of machine learning.

The great thing is that the basics are intuitive — they match how you already reason about uncertainty in daily life ("it'll probably rain," "there's no chance," "it's a coin flip"). We're just going to make that reasoning precise and give it notation. By the end you'll understand conditional probability, Bayes' theorem, and the handful of distributions that show up again and again in AI.

---

## Sample space, events, probability

Let's build the vocabulary from the ground up.

An **experiment** is anything with an uncertain outcome — rolling a die, flipping a coin, checking tomorrow's weather.

The **sample space** (written $\Omega$, Greek capital omega) is the set of *all possible outcomes*. For a single die roll, $\Omega = \{1, 2, 3, 4, 5, 6\}$. For a coin flip, $\Omega = \{\text{heads}, \text{tails}\}$.

An **event** is any outcome or group of outcomes you care about. "Rolling an even number" is an event: $\{2, 4, 6\}$. "Rolling a 3" is an event: $\{3\}$.

The **probability** of an event, $P(A)$, is a number between 0 and 1 saying how likely it is:
- $P(A) = 0$ → impossible, never happens.
- $P(A) = 1$ → certain, always happens.
- $P(A) = 0.5$ → happens half the time.

For a **fair** setup where every outcome is equally likely, probability is just counting:

$$P(A) = \frac{\text{number of outcomes in } A}{\text{total number of outcomes}}$$

Rolling an even number on a fair die: $P(\text{even}) = \frac{3}{6} = 0.5$. Three even faces out of six total.

```python
# Estimate a probability by simulation ("Monte Carlo")
import numpy as np
rng = np.random.default_rng(0)

rolls = rng.integers(1, 7, size=100_000)   # 100k fair die rolls
p_even = np.mean(rolls % 2 == 0)           # fraction that are even
print(p_even)                              # about 0.5
```

That code demonstrates a deep idea: **probability is the long-run fraction.** Roll a die a hundred thousand times, and the fraction of evens settles near the true probability 0.5. Simulating to estimate a probability like this is called a **Monte Carlo** method, and it's used all over AI when the math is too hard to do exactly.

### Two basic rules

1. **Probabilities are between 0 and 1:** $0 \le P(A) \le 1$.
2. **Everything sums to 1:** if you add up the probabilities of all possible outcomes, you get exactly 1. Something must happen. $P(1) + P(2) + \cdots + P(6) = 1$.

This second rule matters enormously in AI: a classifier's output probabilities over all classes must **sum to 1**. That's a hard constraint, and it's exactly why we use the softmax function (coming in a later lesson) to squash raw scores into a valid probability distribution.

---

## Combining events: AND, OR, NOT

Real questions involve multiple events. "What's the chance it rains AND I forget my umbrella?" Let's get precise.

### NOT: the complement

The probability that $A$ does *not* happen is whatever's left over from 1:

$$P(\text{not } A) = 1 - P(A)$$

If there's a 30% chance of rain, there's a 70% chance of no rain. Simple, and surprisingly useful: often the easiest way to compute "at least one" is to compute "none" and subtract from 1.

### AND: joint probability

$P(A \text{ and } B)$ — also written $P(A \cap B)$ or $P(A, B)$ — is the probability that *both* happen. How you compute it depends on whether the events are **independent**.

Two events are **independent** if one happening tells you nothing about the other (coin flips, separate dice). For independent events, you **multiply**:

$$P(A \text{ and } B) = P(A) \times P(B) \qquad \text{(only if independent!)}$$

Two heads in a row: $P(\text{H}) \times P(\text{H}) = 0.5 \times 0.5 = 0.25$.

> **Warning:** the multiply rule *only* works for independent events. If the events are related (rain and umbrella-forgetting aren't independent — you're more careful when it looks like rain), you need the general rule with conditional probability below.

### OR: union

$P(A \text{ or } B)$ — written $P(A \cup B)$ — is the probability that *at least one* happens. The rule:

$$P(A \text{ or } B) = P(A) + P(B) - P(A \text{ and } B)$$

Why subtract the "and" part? Because when you add $P(A)$ and $P(B)$, you've counted the overlap (where both happen) *twice*. You subtract it once to fix the double-count. Picture two overlapping circles: adding both circles double-counts the lens in the middle, so you remove one copy.

If the events **can't both happen** (mutually exclusive, like rolling a 2 or a 5 on one die), the overlap is zero and it simplifies to just $P(A) + P(B)$. Probability of a 2 or a 5: $\frac{1}{6} + \frac{1}{6} = \frac{2}{6}$.

```python
# P(A or B) with a die: A = "even", B = "greater than 4"
# A = {2,4,6}, B = {5,6}, overlap = {6}
pA = 3/6      # even
pB = 2/6      # >4  (5 and 6)
pAB = 1/6     # both (only 6)
print(pA + pB - pAB)   # 4/6 = 0.666...  -> {2,4,5,6}
```

---

## Conditional probability: P(A | B)

This is where probability gets powerful for AI. **Conditional probability** asks: given that $B$ already happened, what's the chance of $A$? We write it $P(A \mid B)$, read "the probability of $A$ *given* $B$."

The bar $\mid$ means "given" or "assuming." You're updating your beliefs based on new information.

The formula:

$$P(A \mid B) = \frac{P(A \text{ and } B)}{P(B)}$$

The intuition: learning that $B$ happened *shrinks your world* down to only the cases where $B$ is true. Inside that smaller world, you ask what fraction also has $A$. You take the "both happen" slice and divide by the "$B$ happened" slice — you're renormalizing to the new, smaller universe.

**Example.** Roll a fair die. What's $P(\text{it's a } 2 \mid \text{it's even})$? Someone tells you the roll was even. Now the only possibilities are $\{2, 4, 6\}$ — a world of 3 outcomes. Among those, one is a 2. So $P(2 \mid \text{even}) = \frac{1}{3}$. Knowing "even" changed the answer from $\frac{1}{6}$ (no info) to $\frac{1}{3}$ (with info). That's the whole point: **information updates probability.**

```python
import numpy as np
rng = np.random.default_rng(0)

rolls = rng.integers(1, 7, size=200_000)
even = rolls % 2 == 0
# Among even rolls, what fraction are exactly 2?
p_two_given_even = np.mean(rolls[even] == 2)
print(p_two_given_even)   # about 0.333
```

### The general AND rule

Rearranging the conditional formula gives the general way to compute "and" (works whether or not events are independent):

$$P(A \text{ and } B) = P(A \mid B)\, P(B)$$

And if $A$ and $B$ happen to be independent, then $P(A \mid B) = P(A)$ (knowing $B$ tells you nothing about $A$), and this collapses back to the simple multiply rule. Nice and consistent.

### Why conditional probability matters for AI

*Everything* a supervised model does is conditional probability. A classifier computes $P(\text{class} \mid \text{input})$ — "given this image, what's the probability it's a cat?" A language model computes $P(\text{next word} \mid \text{previous words})$ — that's literally the definition of a language model. Spam filters compute $P(\text{spam} \mid \text{words in email})$. When you understand $P(A \mid B)$, you understand what the output of a model *is*.

---

## Bayes' theorem: flipping the condition

Sometimes you know $P(B \mid A)$ but you actually want $P(A \mid B)$ — the *reverse*. **Bayes' theorem** is the bridge that flips a conditional around:

$$P(A \mid B) = \frac{P(B \mid A)\, P(A)}{P(B)}$$

This one formula is so important it has philosophies named after it ("Bayesian" reasoning). It tells you how to **update your beliefs when new evidence arrives.** The names of the pieces:

- $P(A)$ — the **prior**: what you believed about $A$ *before* seeing evidence.
- $P(B \mid A)$ — the **likelihood**: how probable the evidence $B$ is *if* $A$ were true.
- $P(A \mid B)$ — the **posterior**: your *updated* belief about $A$ *after* seeing $B$.
- $P(B)$ — the **evidence**: how probable $B$ is overall (a normalizing factor).

### The classic medical-test example (the one that fools everyone)

A disease affects **1%** of people. A test is **99% accurate** — meaning if you have the disease it's positive 99% of the time, and if you don't it's negative 99% of the time (1% false positives). You test **positive**. What's the chance you actually have the disease?

Almost everyone blurts "99%!" The real answer is about **50%**. Here's why, via Bayes.

- $P(\text{disease}) = 0.01$ (prior — the disease is rare)
- $P(\text{positive} \mid \text{disease}) = 0.99$ (likelihood — the test works)
- $P(\text{positive} \mid \text{no disease}) = 0.01$ (false positive rate)

First find $P(\text{positive})$ overall — positives come from two sources, true positives and false positives:

$$P(\text{pos}) = \underbrace{0.99 \times 0.01}_{\text{sick and positive}} + \underbrace{0.01 \times 0.99}_{\text{healthy but false positive}} = 0.0099 + 0.0099 = 0.0198$$

Now Bayes:

$$P(\text{disease} \mid \text{pos}) = \frac{0.99 \times 0.01}{0.0198} = \frac{0.0099}{0.0198} = 0.5$$

Only **50%**! The intuition: the disease is *so rare* that even a good test produces about as many false alarms (from the huge healthy population) as true positives (from the tiny sick population). The rare prior drags the answer way down. This is why doctors retest, and why understanding the base rate matters. Bayes forces you to account for the prior instead of trusting the test blindly.

```python
p_disease = 0.01
p_pos_given_disease = 0.99
p_pos_given_healthy = 0.01

p_pos = p_pos_given_disease * p_disease + p_pos_given_healthy * (1 - p_disease)
p_disease_given_pos = (p_pos_given_disease * p_disease) / p_pos
print(round(p_disease_given_pos, 3))   # 0.5
```

Bayes' theorem underpins spam filters (naive Bayes classifiers), medical AI, and the entire "Bayesian" school of machine learning. It's the mathematical rule for learning from evidence.

---

## Random variables and distributions

A **random variable** is a number whose value comes from a random process. The result of a die roll, the height of a random person, the pixel brightness at a random spot — all random variables. We usually name them with capital letters like $X$.

A **probability distribution** describes *how the probability is spread out* over the possible values of a random variable — which values are likely, which are rare. Let's meet the handful of distributions you'll see constantly in AI.

### Uniform distribution: everything equally likely

Every outcome has the same probability. A fair die is uniform over $\{1, \ldots, 6\}$. A continuous uniform over $[0, 1]$ means any decimal between 0 and 1 is equally likely. It's the "no preference / maximum ignorance" distribution.

```python
import numpy as np
rng = np.random.default_rng(0)
samples = rng.uniform(0, 1, size=5)
print(samples)   # 5 random decimals, each equally likely to be anywhere in [0,1]
```

### Bernoulli distribution: a single yes/no

A **Bernoulli** random variable is a single trial with two outcomes: 1 (success) with probability $p$, and 0 (failure) with probability $1 - p$. A coin flip is Bernoulli with $p = 0.5$. A biased coin, or "will this user click the ad?", is Bernoulli with some other $p$.

This is *the* distribution behind **binary classification**. When a model outputs "80% cat," it's describing a Bernoulli distribution: cat with probability 0.8, not-cat with probability 0.2.

```python
import numpy as np
rng = np.random.default_rng(0)
p = 0.8
flips = (rng.random(10) < p).astype(int)   # 1 with prob 0.8, else 0
print(flips)
```

### Gaussian (normal) distribution: the famous bell curve

The **Gaussian**, or **normal** distribution, is the bell-shaped curve you've seen everywhere. It's described by two numbers:
- $\mu$ (mu) — the **mean**, the center of the bell.
- $\sigma$ (sigma) — the **standard deviation**, how wide the bell is.

Its formula (you don't need to memorize it, just recognize it):

$$p(x) = \frac{1}{\sigma\sqrt{2\pi}}\, e^{-\frac{(x - \mu)^2}{2\sigma^2}}$$

Values near the mean are most likely; the further out you go, the rarer. About 68% of the probability sits within one $\sigma$ of the mean, 95% within two, 99.7% within three (the "68-95-99.7 rule").

Why is the Gaussian *everywhere* in AI? A few reasons: neural network weights are usually initialized from a Gaussian, noise in data is often modeled as Gaussian, and — the deep reason — the **Central Limit Theorem** says that when you add up lots of small independent random effects, the result tends toward a Gaussian *no matter what the individual pieces looked like*. Since real-world quantities are usually sums of many little influences, bell curves show up all over nature and data. (We'll dig into this in the statistics lesson.)

```python
import numpy as np
rng = np.random.default_rng(0)
samples = rng.normal(loc=0.0, scale=1.0, size=100_000)   # mean 0, std 1
print("mean:", round(samples.mean(), 3))   # ~0
print("std :", round(samples.std(), 3))    # ~1
# Fraction within 1 std of the mean:
print(np.mean(np.abs(samples) < 1))        # ~0.68
```

---

## Expected value and variance

Two numbers summarize any distribution: where it's centered, and how spread out it is.

### Expected value: the average outcome

The **expected value** $E[X]$ (also called the mean, $\mu$) is the long-run average — what you'd get if you repeated the experiment forever and averaged. For a discrete random variable, weight each value by its probability and add:

$$E[X] = \sum_i x_i \, P(x_i)$$

For a fair die: $E[X] = 1(\tfrac{1}{6}) + 2(\tfrac{1}{6}) + \cdots + 6(\tfrac{1}{6}) = \frac{21}{6} = 3.5$. Notice the expected value (3.5) isn't even a possible roll — it's the *average* you'd converge to over many rolls, not a value you'll actually see on any single roll. Expected value is a balance point, not a prediction.

```python
import numpy as np
values = np.array([1, 2, 3, 4, 5, 6])
probs = np.full(6, 1/6)
print(np.sum(values * probs))   # 3.5
```

### Variance: how spread out

The **variance** $\text{Var}(X)$ measures how far values typically land from the mean. It's the expected value of the *squared* distance from the mean:

$$\text{Var}(X) = E[(X - \mu)^2]$$

We square the distances so that being above and below the mean both count as "spread" (they don't cancel), and so big deviations are punished more. Because variance is in *squared* units, we often take its square root to get the **standard deviation** $\sigma = \sqrt{\text{Var}(X)}$, which is back in the original, interpretable units.

- Small variance → values cluster tightly around the mean (a narrow bell).
- Large variance → values scatter widely (a wide bell).

```python
import numpy as np
rng = np.random.default_rng(0)

tight = rng.normal(0, 1, 10000)    # std 1
wide  = rng.normal(0, 5, 10000)    # std 5
print("tight variance:", round(tight.var(), 2))   # ~1
print("wide  variance:", round(wide.var(), 2))    # ~25
print("tight std:", round(tight.std(), 2))        # ~1
print("wide  std:", round(wide.std(), 2))         # ~5
```

Variance and standard deviation matter throughout ML: they tell you how noisy your data is, how confident a model is, and how spread out predictions are. They're the tools you'll use in the very next lesson to standardize features.

---

## Why every neural net output is a probability

Let's zoom out and see how this all lands in real AI.

A classifier's final layer produces raw scores (called *logits*) — just arbitrary numbers. But we want probabilities: values in $[0, 1]$ that sum to 1 over the classes. So we pass the logits through **softmax**, which exponentiates and normalizes them into a valid probability distribution. The model's output is then a genuine probability distribution over the classes — obeying the "sums to 1" rule we started with.

Because the output is a probability, we can:
- **Report confidence** ("87% cat").
- **Compare to the truth** using a loss built from probability theory — **cross-entropy** (the topic of the next lesson), which is grounded in exactly the log-probability ideas here.
- **Sample** from it (how language models generate varied text — they sample the next word from the predicted distribution).

So the arc is: probability describes uncertainty → models output probability distributions → we train them with probability-based losses. Every concept in this lesson — conditional probability, distributions, expected value — is load-bearing for how modern AI actually works.

---

## Sort These Concepts

```widget
{
  "type": "concept-sort",
  "title": "Bayes' Theorem: Match the Term",
  "categories": [
    { "name": "Prior P(A)", "color": "#5B5BD6" },
    { "name": "Likelihood P(B|A)", "color": "#F97316" },
    { "name": "Posterior P(A|B)", "color": "#22C55E" }
  ],
  "items": [
    { "text": "Your belief BEFORE seeing evidence", "category": "Prior P(A)" },
    { "text": "Your belief AFTER seeing evidence", "category": "Posterior P(A|B)" },
    { "text": "How probable the evidence is IF hypothesis is true", "category": "Likelihood P(B|A)" },
    { "text": "1% disease prevalence in the population", "category": "Prior P(A)" },
    { "text": "Test positive rate given you have the disease (99%)", "category": "Likelihood P(B|A)" },
    { "text": "50% chance of disease given positive test", "category": "Posterior P(A|B)" }
  ]
}
```

---

## Summary

- The **sample space** is all possible outcomes; an **event** is a subset; **probability** $P(A) \in [0, 1]$ measures likelihood. Over all outcomes, probabilities **sum to 1**.
- **NOT:** $P(\text{not }A) = 1 - P(A)$. **AND (independent):** $P(A)P(B)$. **OR:** $P(A) + P(B) - P(A \text{ and } B)$ (subtract the double-counted overlap).
- **Conditional probability** $P(A \mid B) = \frac{P(A \text{ and } B)}{P(B)}$ updates a probability given new information. Almost every model computes a conditional like $P(\text{class} \mid \text{input})$.
- **Bayes' theorem** $P(A \mid B) = \frac{P(B \mid A)P(A)}{P(B)}$ flips a conditional and formalizes learning from evidence. The rare-disease example shows why the prior (base rate) matters — a positive test can still mean only 50%.
- Key **distributions**: **uniform** (all equal), **Bernoulli** (single yes/no with probability $p$ — the basis of binary classification), and **Gaussian** (the bell curve, defined by mean $\mu$ and std $\sigma$, everywhere thanks to the Central Limit Theorem).
- **Expected value** $E[X] = \sum x_i P(x_i)$ is the long-run average (the center); **variance** $E[(X-\mu)^2]$ measures spread, and its square root is the **standard deviation**.
- Neural nets output **probability distributions** (via softmax), which is why they can report confidence and be trained with probability-based losses like cross-entropy.
