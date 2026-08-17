---
title: Statistics for ML
track: math
order: 6
estimatedTime: 40
difficulty: intermediate
---

# Statistics for ML

Machine learning is, at its core, applied statistics with a lot of computing power. Before you ever train a model, you need to *understand your data* — where it's centered, how spread out it is, how features relate to each other, and whether patterns you see are real or just noise. That's statistics. And after you train a model, statistics tells you whether it actually learned something or just memorized — the difference between a model that works and one that embarrasses you on the leaderboard.

This lesson pulls together the statistical tools you'll use constantly: summary numbers (mean, median, variance), how to measure relationships (correlation), the ever-present normal distribution and why it's everywhere, standardization, a taste of hypothesis testing, and the single most important idea in practical ML — the **train/validation/test split** and what **overfitting** really is, seen through a statistical lens.

---

## Summarizing data: center and spread

When you get a pile of numbers, the first thing you do is summarize it. Two questions: **where is it centered?** and **how spread out is it?**

### Measures of center

**Mean** (average): add everything up, divide by the count.

$$\bar{x} = \frac{1}{n}\sum_i x_i$$

The mean is the "balance point" of the data. It's the most common summary, but it has a weakness: it gets **yanked around by outliers**. One billionaire in a room of ten people makes the *average* wealth in millions, even though nine people are broke.

**Median**: sort the numbers and take the middle one. Half the data is below, half above. The median *ignores* how extreme the outliers are — it only cares about order — so it's **robust** to outliers. That billionaire barely moves the median.

**Mode**: the most frequently occurring value. Useful for categories ("the most common label"), less so for continuous numbers.

> **Rule of thumb:** if your data has outliers or is skewed (like income, house prices, or city populations), the **median** describes the "typical" value more honestly than the mean. If the data is roughly symmetric and clean, the mean is fine and carries more information.

```python
import numpy as np

data = np.array([1, 2, 2, 3, 4, 5, 100])   # note the outlier 100
print("mean  :", np.mean(data))     # 16.7 -> dragged up by the 100
print("median:", np.median(data))   # 3.0  -> unfazed by the outlier
# mode: the value that appears most often
vals, counts = np.unique(data, return_counts=True)
print("mode  :", vals[np.argmax(counts)])   # 2
```

Look at that gap: mean 16.7 vs median 3. One outlier made the mean useless as a description of a "typical" value. This is why data scientists always look at both.

### Measures of spread

**Variance** measures how far, on average, points sit from the mean (squared):

$$\text{Var}(x) = \frac{1}{n}\sum_i (x_i - \bar{x})^2$$

**Standard deviation** is the square root of variance, $\sigma = \sqrt{\text{Var}}$. We prefer it for interpretation because it's back in the *original units*. If your data is heights in cm, the variance is in "cm squared" (meaningless), but the standard deviation is in cm (meaningful — "heights vary by about 7 cm").

- Small std → data clusters tightly around the mean.
- Large std → data is spread out, noisy.

```python
import numpy as np

tight = np.array([49, 50, 50, 51, 50])   # all near 50
wide  = np.array([10, 90, 50, 30, 70])   # all over the place
print("tight: mean", tight.mean(), "std", round(tight.std(), 2))   # std ~0.6
print("wide : mean", wide.mean(),  "std", round(wide.std(),  2))   # std ~28
```

Both might have a mean near 50, but they're wildly different datasets. The standard deviation is what tells them apart. **Center alone never describes data — you always need spread too.**

---

## Correlation: how two things move together

Often you want to know: **when one variable goes up, does the other go up too?** Do taller people weigh more? Does studying more raise test scores? That's **correlation**.

The **correlation coefficient** (Pearson's $r$) is a single number between $-1$ and $+1$ that measures the *linear* relationship between two variables:

- $r = +1$ → perfect positive relationship. When $x$ goes up, $y$ goes up, in a perfectly straight line.
- $r = 0$ → no linear relationship. Knowing $x$ tells you nothing (linearly) about $y$.
- $r = -1$ → perfect negative relationship. When $x$ goes up, $y$ goes *down*, straight line.

The formula standardizes both variables and averages the product of their deviations:

$$r = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_i (x_i - \bar{x})^2}\,\sqrt{\sum_i (y_i - \bar{y})^2}}$$

You'll notice this looks a lot like a **cosine similarity** (from the dot-product lesson) between the mean-centered variables — and it is! Correlation is essentially the cosine of the angle between two centered data vectors. Nice connection.

```python
import numpy as np
rng = np.random.default_rng(0)

x = rng.normal(0, 1, 500)
y_pos = 2*x + rng.normal(0, 0.5, 500)   # strongly follows x
y_neg = -3*x + rng.normal(0, 0.5, 500)  # strongly opposes x
y_none = rng.normal(0, 1, 500)          # unrelated to x

print("positive:", round(np.corrcoef(x, y_pos)[0, 1], 2))    # near +1
print("negative:", round(np.corrcoef(x, y_neg)[0, 1], 2))    # near -1
print("none    :", round(np.corrcoef(x, y_none)[0, 1], 2))   # near 0
```

### The warning everyone must hear: correlation is not causation

Just because two things move together does **not** mean one causes the other. Ice cream sales and drowning deaths are correlated — but ice cream doesn't cause drowning. A hidden third factor (hot weather) drives both. In ML this bites constantly: a feature might correlate with your target for a silly reason, and a model that leans on it will fail in the real world. Always ask *why* a correlation exists before trusting it.

Also, correlation only catches **linear** relationships. Two variables can have a perfect relationship (like $y = x^2$) and still show correlation near zero, because the relationship isn't a straight line. A correlation of 0 means "no *linear* relationship," not "no relationship at all."

### Why correlation matters for AI

Correlation is your first tool for **feature selection**. Features strongly correlated with the target are promising predictors. Features strongly correlated *with each other* are redundant — they carry overlapping information, which can confuse some models (this is called *multicollinearity*). Before building anything fancy, computing a correlation matrix tells you a lot about which features matter and which duplicate each other.

---

## The normal distribution and why it's everywhere

We met the Gaussian (normal) bell curve in the probability lesson. Here's *why* it's so central to statistics and ML — and it's one of the most beautiful results in all of mathematics.

### The Central Limit Theorem (CLT)

The **Central Limit Theorem** says something almost magical:

> When you add up (or average) many independent random things, the result follows a **normal distribution — no matter what shape the individual things had.**

Read that again. The individual pieces can be wildly non-normal — uniform, skewed, weird, whatever. But *sum enough of them* and the total is a bell curve. The bell curve emerges from chaos, automatically, as long as you're adding up many small independent effects.

Let's *see* it happen. We'll take dice rolls (a flat, uniform distribution — nothing like a bell), average a bunch of them together, and repeat:

```python
import numpy as np
rng = np.random.default_rng(0)

# One die roll is UNIFORM (flat) — not a bell curve at all.
# But average 30 rolls together, many times, and watch what emerges:
means = [rng.integers(1, 7, size=30).mean() for _ in range(10000)]
means = np.array(means)

print("mean of the averages:", round(means.mean(), 2))   # ~3.5 (the die's mean)
print("std  of the averages:", round(means.std(), 3))    # small — averaging reduces spread
# If you histogrammed `means`, you'd see a beautiful bell curve,
# even though a single die roll is perfectly flat.
```

Run it. The averages form a bell curve even though a die roll is flat. That's the CLT in action.

### Why the CLT makes the Gaussian appear everywhere

Real-world quantities are usually the sum of *many* small independent influences:
- A person's **height** = genetics (many genes, each a small effect) + nutrition + sleep + ... → bell curve.
- **Measurement error** = many tiny random disturbances added up → bell curve.
- The **noise** in almost any dataset = the accumulation of countless small effects → bell curve.

Because so much of the world is "many small things added up," the Gaussian is the default distribution nature reaches for. That's why ML assumes it constantly: we initialize neural network weights from Gaussians, we model noise as Gaussian, we assume residuals are Gaussian in linear regression. The CLT is the reason those assumptions are so often reasonable.

---

## Z-scores and standardization

Here's a practical problem. Suppose one feature is "age" (values 0–100) and another is "income" (values 0–500,000). If you feed these raw into a model, the income feature — just because its numbers are *bigger* — will dominate distance and gradient calculations, drowning out age. The features are on totally different scales. We need to put them on equal footing.

The fix is **standardization**: convert every value into a **z-score**, which measures *how many standard deviations it is from the mean*:

$$z = \frac{x - \mu}{\sigma}$$

Subtract the mean (so the data is centered at 0), then divide by the standard deviation (so the spread is exactly 1). After standardizing, *every* feature has mean 0 and standard deviation 1 — they're all on the same scale, and no feature bullies the others just for having big numbers.

A z-score is also interpretable on its own: $z = 2$ means "this value is 2 standard deviations above average" — unusually high. $z = -1.5$ means "1.5 std below average." For normal data, $|z| > 3$ is quite rare (remember the 68-95-99.7 rule), so z-scores are also a handy way to flag outliers.

```python
import numpy as np

age    = np.array([20, 25, 30, 45, 60, 22, 38])
income = np.array([30000, 45000, 50000, 90000, 120000, 35000, 70000])

def standardize(x):
    return (x - x.mean()) / x.std()

age_z    = standardize(age)
income_z = standardize(income)

print("age z-scores   :", np.round(age_z, 2))
print("income z-scores:", np.round(income_z, 2))
print("age    -> mean:", round(age_z.mean(), 6), "std:", round(age_z.std(), 3))
print("income -> mean:", round(income_z.mean(), 6), "std:", round(income_z.std(), 3))
```

Both features now have mean ~0 and std ~1. They're comparable. **Standardization is one of the most common preprocessing steps in all of ML** — you'll use scikit-learn's `StandardScaler` for exactly this. Distance-based models (KNN, SVM), gradient descent, and neural nets all train faster and behave better on standardized data.

> **Critical detail for later:** you compute the mean and std from the **training data only**, then apply those *same* numbers to the validation and test data. Peeking at test data to compute the scaling is a form of cheating (data leakage) that we'll return to.

---

## Hypothesis testing (the concept)

Sometimes you see a difference in data and need to ask: **is this real, or could it just be random luck?** That's what **hypothesis testing** is for. You don't need the heavy machinery for IOAI, but you should understand the idea, because it's how we decide whether a result *means* something.

The setup:
- The **null hypothesis** ($H_0$) is the boring default: "there's no real effect; any difference is just chance."
- The **alternative hypothesis** ($H_1$) is what you suspect: "there IS a real effect."

You then compute a **p-value**: the probability of seeing data *at least as extreme* as what you got, *if the null hypothesis were true* (i.e., if it were all just luck). 

- **Small p-value** (traditionally < 0.05) → "this would be really unlikely by pure chance, so I doubt the null. The effect is probably real." We say the result is *statistically significant*.
- **Large p-value** → "eh, this could easily happen by chance. No strong evidence of an effect."

The intuition with a coin: if a coin lands heads 6 times out of 10, is it rigged? Probably not — that happens by luck all the time (big p-value, not significant). But 60 heads out of 100? Now it'd be very strange for a fair coin (small p-value) — you'd start to believe it's biased.

```python
import numpy as np
rng = np.random.default_rng(0)

# Is a coin that got 60/100 heads suspicious? Simulate a FAIR coin 100k times
# and see how often pure luck gives 60+ heads out of 100.
fair_trials = rng.binomial(n=100, p=0.5, size=100_000)
p_value = np.mean(fair_trials >= 60)
print("p-value for 60+/100 heads from a fair coin:", round(p_value, 3))
# Around 0.03 — fairly unlikely by chance, so we'd start to doubt fairness.
```

> **The famous pitfall:** a p-value below 0.05 does **not** mean "95% chance the effect is real," and statistical significance does **not** mean the effect is *large* or *important*. With enough data, even a tiny, useless difference becomes "significant." Significance is about *"is it probably not zero,"* not *"does it matter."* Keep those separate.

In ML, this mindset guards you against fooling yourself: is model B *really* better than model A, or did it just get lucky on this particular test set? That skepticism is exactly what cross-validation (below) helps answer.

---

## Train / validation / test splits — and why

Now the most important practical idea in all of applied ML. When you build a model, you must answer: **will it work on data it has never seen?** That's the *only* thing that matters — a model that only works on data it already saw is useless. To find out, you **hold out** data.

You split your dataset into three parts:

| Split | Purpose | Does the model see it during training? |
|---|---|---|
| **Training set** (~60-80%) | The model learns its parameters from this. | Yes — it's what the model fits to. |
| **Validation set** (~10-20%) | You tune choices (learning rate, tree depth, which model) by checking performance here. | No — used only to *check*, not to fit. |
| **Test set** (~10-20%) | The final, one-time honest estimate of real-world performance. | No — touched *once*, at the very end. |

The logic is like studying for an exam:
- **Training set** = the practice problems you study from.
- **Validation set** = a practice exam you use to decide what to study more.
- **Test set** = the *real* exam, which you must not peek at beforehand, or your grade means nothing.

> **The golden rule: never let your model learn from the test set.** Not for training, not for tuning, not for computing scaling factors — nothing. The moment your model has "seen" the test set in any way, the test score becomes a lie. It stops measuring generalization and starts measuring memorization. This leaking of test information into training is called **data leakage**, and it's the number-one way people fool themselves (and lose competitions).

```python
import numpy as np
from sklearn.model_selection import train_test_split

X = np.arange(100).reshape(100, 1)   # 100 fake examples
y = np.arange(100)

# First split off the test set (held for the very end)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Then split the rest into train and validation
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=0)

print("train:", len(X_train))   # 60
print("val  :", len(X_val))     # 20
print("test :", len(X_test))    # 20
```

---

## Overfitting from a statistical viewpoint

Finally, let's understand **overfitting** the way a statistician sees it — it'll make the whole ML track click.

Every dataset is really two things mixed together:
1. **Signal** — the true, general pattern you want to learn (e.g., "bigger houses cost more").
2. **Noise** — random junk specific to *this* sample (measurement errors, coincidences, one weird house).

A good model learns the **signal** and *ignores* the noise. **Overfitting** is when a model, in its eagerness to fit the training data perfectly, starts learning the **noise** too — memorizing the exact quirks of the training sample that won't repeat in new data.

Here's the tell-tale statistical signature:
- An overfit model has **low training error** (it nailed the training data, noise and all) but **high validation/test error** (the noise it memorized doesn't help on new data — it actually hurts).
- The **gap** between training error and validation error is the fingerprint of overfitting. Small gap = healthy. Big gap = the model memorized noise.

The opposite failure is **underfitting**: the model is too simple to even capture the signal, so it does badly on *both* training and validation. High error everywhere.

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(0)
X = np.linspace(0, 1, 20).reshape(-1, 1)
y = np.sin(2*np.pi*X).ravel() + rng.normal(0, 0.2, 20)   # signal + noise

X_train, X_val = X[:14], X[14:]
y_train, y_val = y[:14], y[14:]

for degree in [1, 4, 15]:
    poly = PolynomialFeatures(degree)
    Xt = poly.fit_transform(X_train)
    Xv = poly.transform(X_val)
    model = LinearRegression().fit(Xt, y_train)
    train_err = mean_squared_error(y_train, model.predict(Xt))
    val_err   = mean_squared_error(y_val,   model.predict(Xv))
    print(f"degree {degree:2d}: train MSE {train_err:.3f}, val MSE {val_err:.3f}")
```

Run it and watch the pattern:
- **Degree 1** (a straight line) — underfits. High error on both. Too simple for the wavy signal.
- **Degree 4** — about right. Low error on both, small gap. It found the signal.
- **Degree 15** — overfits. Tiny training error but *large* validation error. It bent itself through every noisy point in the training data, and those bends are garbage on new data.

That train-vs-validation gap is the thing you watch during *every* model you ever train. It's why we split data in the first place: without a validation set, you'd never know your beautiful low training error was actually a symptom of memorizing noise. The entire next track (Classical ML) revolves around finding that "degree 4" sweet spot — enough capacity to catch the signal, not so much that it memorizes noise.

---

## Sort These Statistical Concepts

```widget
{
  "type": "concept-sort",
  "title": "Mean or Median? Which to Use?",
  "categories": [
    { "name": "Use Mean", "color": "#5B5BD6" },
    { "name": "Use Median", "color": "#F97316" }
  ],
  "items": [
    { "text": "Symmetric data with no outliers (test scores)", "category": "Use Mean" },
    { "text": "House prices (a few billionaire mansions skew data)", "category": "Use Median" },
    { "text": "Standardization (mean=0, std=1 scaling)", "category": "Use Mean" },
    { "text": "Income distribution in a country", "category": "Use Median" },
    { "text": "Normally distributed noise", "category": "Use Mean" },
    { "text": "Outlier-robust evaluation metric", "category": "Use Median" }
  ]
}
```

---

## Summary

- **Center:** the **mean** is the average (sensitive to outliers); the **median** is the middle value (robust to outliers); the **mode** is the most common value. Use the median for skewed/outlier-heavy data.
- **Spread:** **variance** is the average squared distance from the mean; **standard deviation** (its square root) is in the original units and is the go-to measure of spread. Center without spread never describes data.
- **Correlation** ($r$, from $-1$ to $+1$) measures the *linear* relationship between two variables — but **correlation is not causation**, and $r = 0$ only rules out *linear* relationships. It's a first tool for feature selection.
- The **Central Limit Theorem** says sums/averages of many independent random things become **normal** regardless of the originals' shape — which is *why* the Gaussian bell curve appears all over data and ML.
- **Standardization** turns values into **z-scores** $z = \frac{x - \mu}{\sigma}$, giving every feature mean 0 and std 1 so no feature dominates by scale. Compute the scaling on training data only.
- **Hypothesis testing** and the **p-value** ask "could this be just chance?" A small p-value means the result is unlikely by luck (statistically significant) — but significance is not the same as importance.
- Split data into **train / validation / test**. The **golden rule**: never let the model learn from the test set (avoid **data leakage**). 
- **Overfitting** = learning noise instead of signal: low training error but high validation error (a big *gap*). **Underfitting** = too simple, high error everywhere. Watching the train-vs-validation gap is how you diagnose both.
