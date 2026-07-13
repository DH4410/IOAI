---
title: What is Machine Learning?
track: ml
order: 1
estimatedTime: 30
difficulty: beginner
---

# What is Machine Learning?

Machine learning is one of the most exciting and rapidly growing fields in computer science. Instead of writing explicit rules for a program to follow, you give it data and let it figure out the rules itself. This lesson will give you a solid mental model of what ML is, why it works, and how practitioners actually think about problems.

---

## Why Machine Learning?

Imagine you are asked to write a program that tells you whether an email is spam. You might start with rules like:

- If the email contains "FREE MONEY", mark it as spam
- If the sender is unknown, mark it as spam
- If the subject is in ALL CAPS, mark it as spam

This works for a while. But spammers adapt. Soon they write "Fr3e M0ney" and your rules break. You spend your days adding more and more rules, playing whack-a-mole with clever spammers.

Machine learning takes a different approach: instead of writing rules, you show the program thousands of examples of spam and non-spam email, and let it learn the patterns on its own.

This is the core idea: **learn from data, not from hand-crafted rules**.

---

## The Three Types of Machine Learning

### 1. Supervised Learning

In supervised learning, every training example has a **label** — the correct answer. The model learns to map inputs to outputs.

| Input (Features) | Output (Label) |
|---|---|
| House size, location, age | Price ($) |
| Email text | Spam / Not Spam |
| Image of a digit | 0–9 |
| Patient measurements | Disease / No Disease |

The word "supervised" comes from the idea that a teacher (the labels) is supervising the learning process.

Supervised learning splits into two sub-problems:

- **Classification**: The output is a category (spam vs. not spam, cat vs. dog)
- **Regression**: The output is a number (house price, temperature tomorrow)

### 2. Unsupervised Learning

In unsupervised learning, there are **no labels**. The model tries to find hidden structure in the data on its own.

Common tasks:
- **Clustering**: Group customers by purchasing behavior
- **Dimensionality Reduction**: Compress 1000-feature data into 2 features for visualization
- **Anomaly Detection**: Find unusual transactions in a bank dataset

### 3. Reinforcement Learning

In reinforcement learning, an **agent** learns by interacting with an **environment**. It takes actions, receives rewards (positive or negative), and learns to maximize total reward over time.

Examples:
- A robot learning to walk
- AlphaGo learning to play Go
- A trading algorithm learning when to buy and sell

> **Note for IOAI:** The classical ML track focuses on supervised and unsupervised learning. You will encounter reinforcement learning in advanced modules. For now, supervised learning is where most competition problems live.

---

## Features and Labels

Before building any model, you need to understand your data.

**Features** (also called attributes, inputs, or X): the information you give to the model.

**Labels** (also called targets, outputs, or y): what you want to predict.

```python
import numpy as np

# Think of a dataset as a table
# Each row is one example (a house)
# Each column in X is a feature

X = np.array([
    [1400, 3, 10],   # 1400 sq ft, 3 bedrooms, 10 years old
    [1800, 4, 5],
    [1200, 2, 20],
    [2200, 4, 2],
    [900,  2, 30],
])

# y is the label — the house price in $1000s
y = np.array([250, 320, 180, 400, 140])

print("Number of examples:", X.shape[0])
print("Number of features:", X.shape[1])
print("Features:\n", X)
print("Labels:", y)
```

---

## A Motivating Example: Predicting House Prices

Let's make this concrete. Suppose you are a real estate agent in a new city. You have data on 500 houses that sold last year, including:
- Square footage
- Number of bedrooms
- Age of the house
- Distance from city center
- Neighborhood crime rate

And you know the final sale price of each house.

You want to build a model that, given a new house's features, predicts its price.

This is a classic **regression** problem. Here is the full ML workflow in one picture:

```
Raw Data → Clean & Prepare → Build Model → Train → Evaluate → Deploy
```

Let's walk through each step.

---

## The General ML Workflow

### Step 1: Collect and Understand Data

Before touching a model, understand your data. Ask:
- How many examples do I have? (More is better)
- Are there missing values?
- What are the ranges of each feature?
- Is the data balanced (equal class representation for classification)?

```python
import numpy as np

# Simulated house price data
np.random.seed(42)
n = 100

sqft = np.random.randint(800, 3000, n)
bedrooms = np.random.randint(1, 6, n)
age = np.random.randint(0, 50, n)

# Price roughly depends on sqft and bedrooms, with some noise
price = 50 + 0.12 * sqft + 15 * bedrooms - 0.5 * age + np.random.randn(n) * 20

print("Dataset shape: {} examples, {} features".format(n, 3))
print("Price range: ${:.0f}k – ${:.0f}k".format(price.min(), price.max()))
print("Average sqft:", sqft.mean())
```

### Step 2: Split into Train / Validation / Test Sets

This is one of the most important concepts in ML. You need to evaluate your model on data it has **never seen**. Why?

Because a model can memorize the training data and perform perfectly on it without learning anything useful. This is called **overfitting** (covered deeply in Lesson 9).

```
All Data
├── Training Set    (~70%)  ← Model learns from this
├── Validation Set  (~15%)  ← You tune the model using this
└── Test Set        (~15%)  ← Only touched at the very end
```

> **Important rule:** Never look at the test set until you are completely done. Looking at it to make decisions about your model is "data leakage" — it gives you an overly optimistic view of performance.

```python
from sklearn.model_selection import train_test_split
import numpy as np

np.random.seed(42)
n = 200

# Create feature matrix
X = np.column_stack([
    np.random.randint(800, 3000, n),   # sqft
    np.random.randint(1, 6, n),         # bedrooms
    np.random.randint(0, 50, n)         # age
])
y = 50 + 0.12 * X[:,0] + 15 * X[:,1] - 0.5 * X[:,2] + np.random.randn(n) * 20

# First split: separate out the test set
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# Second split: separate train and validation from the remaining
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.176, random_state=42  # 0.176 * 0.85 ≈ 0.15
)

print(f"Training examples:   {len(X_train)}")
print(f"Validation examples: {len(X_val)}")
print(f"Test examples:       {len(X_test)}")
```

### Step 3: Build and Train a Model

Now you choose a model and train it. Training means adjusting the model's internal parameters so it makes accurate predictions on the training data.

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

np.random.seed(42)
n = 200
X = np.column_stack([
    np.random.randint(800, 3000, n),
    np.random.randint(1, 6, n),
    np.random.randint(0, 50, n)
])
y = 50 + 0.12 * X[:,0] + 15 * X[:,1] - 0.5 * X[:,2] + np.random.randn(n) * 20

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model — just two lines!
model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained successfully.")
print("Learned weights:", model.coef_)
print("Learned bias:", model.intercept_)
```

### Step 4: Evaluate

After training, you measure how well the model performs on data it has not seen.

```python
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error: ${rmse:.1f}k")
print(f"R² Score: {r2:.3f}")
```

### Step 5: Iterate

If performance is not good enough, you go back and:
- Collect more data
- Engineer better features
- Try a different model
- Tune hyperparameters

This cycle repeats until you are satisfied with performance on the validation set.

### Step 6: Deploy

Once satisfied, you test on the test set one final time to get an honest estimate of real-world performance, then deploy.

---

## Manual Model vs. Sklearn Approach

To really understand what sklearn is doing, let's first build a simple "model" by hand, then compare to sklearn.

### The Problem

We have one feature (house size in sqft) and want to predict price. We will try to find the best weight `w` and bias `b` such that:

```
price ≈ w * sqft + b
```

### Manual Approach: Trying Different Weights

```python
import numpy as np

# Simple 1D dataset
sqft = np.array([800, 1000, 1200, 1500, 1800, 2000, 2200, 2500])
price = np.array([120, 145, 168, 205, 240, 270, 295, 335])  # in $1000s

def predict(w, b, x):
    return w * x + b

def mse_loss(w, b, x, y):
    predictions = predict(w, b, x)
    errors = predictions - y
    return np.mean(errors ** 2)

# Try different weights and biases
print("Manual search:")
print(f"{'w':>8} {'b':>8} {'MSE':>12}")
print("-" * 30)
for w in [0.08, 0.10, 0.12, 0.14, 0.16]:
    for b in [20, 30, 40, 50]:
        mse = mse_loss(w, b, sqft, price)
        print(f"{w:>8.2f} {b:>8} {mse:>12.2f}")
```

This is tedious. We are guessing. Sklearn uses a mathematical formula (or gradient descent) to find the optimal `w` and `b` automatically.

### Sklearn Approach

```python
from sklearn.linear_model import LinearRegression
import numpy as np

sqft = np.array([800, 1000, 1200, 1500, 1800, 2000, 2200, 2500])
price = np.array([120, 145, 168, 205, 240, 270, 295, 335])

# sklearn expects X to be a 2D matrix, so reshape
X = sqft.reshape(-1, 1)  # shape: (8, 1)

model = LinearRegression()
model.fit(X, price)

print(f"Optimal weight: {model.coef_[0]:.4f}")
print(f"Optimal bias:   {model.intercept_:.4f}")

# Predict the price of a 1700 sqft house
new_house = np.array([[1700]])
predicted = model.predict(new_house)
print(f"\nPredicted price for 1700 sqft house: ${predicted[0]:.1f}k")

# Compare predictions vs actuals
predictions = model.predict(X)
for sq, actual, pred in zip(sqft, price, predictions):
    print(f"  {sq} sqft → actual: ${actual}k, predicted: ${pred:.1f}k")
```

---

## Classification vs. Regression

Let's be clear about the two main types of supervised learning:

### Classification

The output is one of several **discrete categories**.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate a binary classification dataset
X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    random_state=42
)

print("Classification dataset:")
print(f"  Features shape: {X.shape}")
print(f"  Labels: {np.unique(y)} (0 = class A, 1 = class B)")
print(f"  Class 0 count: {(y == 0).sum()}")
print(f"  Class 1 count: {(y == 1).sum()}")
```

### Regression

The output is a **continuous number**.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# Generate a regression dataset
X, y = make_regression(
    n_samples=200,
    n_features=3,
    noise=15,
    random_state=42
)

print("Regression dataset:")
print(f"  Features shape: {X.shape}")
print(f"  Target range: [{y.min():.1f}, {y.max():.1f}]")
print(f"  Target mean: {y.mean():.1f}")
```

---

## Examples from IOAI Competitions

IOAI problems often look like real-world ML tasks. Here are some patterns you will see:

| Problem Type | Task | Model Type |
|---|---|---|
| Digit Recognition | Given a 28×28 pixel image, predict which digit (0-9) it is | Classification |
| Sentiment Analysis | Given a movie review, predict positive or negative | Classification |
| Temperature Forecast | Given weather history, predict tomorrow's temperature | Regression |
| Disease Prediction | Given patient vitals, predict diagnosis | Classification |
| Sales Forecasting | Given historical sales, predict next month | Regression |
| Anomaly Detection | Find unusual patterns in network traffic | Unsupervised |

---

## How to Think Like an ML Practitioner

The best ML practitioners have a structured way of thinking. Here is a checklist for every problem:

### Before Writing Any Code

1. **What am I predicting?** (regression or classification?)
2. **What is my evaluation metric?** (accuracy, MSE, F1, AUC?)
3. **How much data do I have?** (100 examples vs. 1 million changes everything)
4. **What are the features?** (numeric, categorical, text, images?)
5. **Are there obvious patterns?** (look at the data first!)

### The Practitioner's Hierarchy

When starting a new problem, work through this hierarchy:

```
1. Establish a baseline (even random guessing gives a number)
2. Try the simplest possible model (linear/logistic regression)
3. Look at errors — which examples does the model get wrong? Why?
4. Engineer better features
5. Try more powerful models (trees, ensembles)
6. Tune hyperparameters
7. Ensemble multiple models (if really needed)
```

> **Competition tip:** A simple model with good features beats a complex model with bad features almost every time. Always start simple.

---

## Why Does ML Actually Work?

This is a deep question with a simple intuition: **the world has structure**.

Data is not random. House prices really do depend on size and location. Spam really does have patterns. Cats really do look different from dogs.

Machine learning works because:
1. **Real patterns exist** in data
2. **We have enough examples** for the model to find them
3. **New examples resemble old ones** — the future looks like the past

When one of these assumptions breaks, ML struggles. This is why ML models can fail in novel situations they have never seen before.

---

## Putting It All Together: Full ML Pipeline

Here is a complete, runnable example showing the full workflow on a house price problem:

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ── 1. Generate Data ──────────────────────────────────────────────
np.random.seed(42)
n = 300

sqft     = np.random.randint(600, 3500, n).astype(float)
bedrooms = np.random.randint(1, 7, n).astype(float)
age      = np.random.randint(0, 60, n).astype(float)
distance = np.random.uniform(0.5, 30, n)  # miles from city center

price = (
    50
    + 0.15 * sqft
    + 20  * bedrooms
    - 1.2 * age
    - 3.0 * distance
    + np.random.randn(n) * 25
)

X = np.column_stack([sqft, bedrooms, age, distance])
feature_names = ["sqft", "bedrooms", "age", "distance"]

# ── 2. Split Data ─────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, price, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ── 3. Scale Features ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit on train only!
X_test_scaled  = scaler.transform(X_test)        # only transform test

# ── 4. Train Model ────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ── 5. Evaluate ───────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\nTest RMSE: ${rmse:.1f}k")
print(f"Test R²:   {r2:.3f}")

print("\nLearned feature importance (by coefficient magnitude):")
for name, coef in zip(feature_names, model.coef_):
    print(f"  {name:>10}: {coef:+.2f}")

# ── 6. Make a Prediction ──────────────────────────────────────────
new_house = np.array([[2000, 3, 10, 5.0]])  # 2000 sqft, 3 bed, 10yr, 5mi
new_house_scaled = scaler.transform(new_house)
predicted_price = model.predict(new_house_scaled)[0]
print(f"\nPredicted price for new house: ${predicted_price:.1f}k")
```

---

## Summary

| Concept | Key Idea |
|---|---|
| Machine Learning | Learn patterns from data instead of coding rules |
| Supervised Learning | Learn from labeled examples |
| Unsupervised Learning | Find structure without labels |
| Classification | Predict a category |
| Regression | Predict a number |
| Features | The inputs to your model |
| Labels | The outputs you want to predict |
| Train/Val/Test split | Evaluate fairly on unseen data |
| ML Workflow | Data → Clean → Split → Train → Evaluate → Iterate → Deploy |

You now have the foundation. In the next lesson, we will dive deep into **Linear Regression** — the simplest and most interpretable supervised learning algorithm — and understand exactly how models learn from data.
