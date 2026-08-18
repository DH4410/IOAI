---
title: Overfitting and Regularization
track: ml
order: 9
estimatedTime: 40
difficulty: intermediate
---

# Overfitting and Regularization

A model that scores 99% on training data but 65% on test data is not a good model — it has memorized the training set rather than learning the actual pattern. This is overfitting, and it is the most common problem in ML competitions.

---

## 1. What Is Overfitting?

When a model is too complex for the amount of data, it starts to memorize every detail — including random noise that is specific to the training set. It gets great training scores but performs poorly on new examples.

```
Underfit:  Train=60%  Test=58%   → model too simple, misses the pattern
Good fit:  Train=91%  Test=88%   → model generalizes well
Overfit:   Train=99%  Test=64%   → model memorizing noise
```

The interactive chart below lets you see this directly. Drag the slider from simple to complex and watch the gap open up.

```widget
{
  "type": "overfit-curve",
  "title": "Drag to see how complexity affects train vs validation accuracy"
}
```

**Quick check:** Your model has 99% training accuracy and 68% test accuracy. What is this and how do you fix it?
> Overfitting. Fix: simplify the model, add regularization, get more data, or use dropout.

---

## 2. Bias-Variance Tradeoff

This is the classic framing of the overfitting problem:

- **Bias** = how wrong the model is on average (underfitting = high bias)
- **Variance** = how much the model's output changes with different training sets (overfitting = high variance)

You cannot eliminate both at once. Making the model more complex reduces bias but increases variance. The goal is to find the sweet spot.

---

## 3. Regularization: L1 and L2

Regularization adds a penalty to the loss function to discourage large weights. Large weights usually mean the model is fitting noise.

**L2 regularization (Ridge):** Penalizes the sum of squared weights. Pushes all weights toward zero but rarely to exactly zero.

```
Loss_total = Loss_original + λ * sum(w²)
```

**L1 regularization (Lasso):** Penalizes the sum of absolute values. Can push some weights all the way to zero (feature selection).

```
Loss_total = Loss_original + λ * sum(|w|)
```

In scikit-learn:

```python
from sklearn.linear_model import Ridge, Lasso, LogisticRegression

# L2 regularization on linear regression
ridge = Ridge(alpha=1.0)   # alpha = λ
ridge.fit(X_train, y_train)

# L1 regularization
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Logistic regression uses C = 1/λ (inverse!)
# Smaller C = more regularization
lr = LogisticRegression(C=0.1, penalty='l2')
lr.fit(X_train, y_train)
```

The `alpha` / `C` parameter controls how strong the regularization is. Tune it with cross-validation.

---

## 4. Cross-Validation

Instead of a single train/test split, k-fold cross-validation splits the data into k parts and trains k separate models, each time using a different fold as the validation set.

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

scores = cross_val_score(Ridge(alpha=1.0), X, y, cv=5, scoring='r2')
print(f'CV R²: {scores.mean():.3f} ± {scores.std():.3f}')
```

This gives a much more reliable estimate of how well the model generalizes. Use it to compare models and tune hyperparameters.

---

## 5. Early Stopping

For gradient-based models (neural networks, gradient boosting), you can stop training when the validation loss stops improving:

```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=1000,
    validation_fraction=0.1,
    n_iter_no_change=20,   # stop if no improvement for 20 rounds
    tol=1e-4,
)
model.fit(X_train, y_train)
print(f'Stopped at {model.n_estimators_} estimators')
```

Early stopping is one of the most effective regularization techniques because it prevents the model from training too long.

---

## 6. Practical Checklist

When your model is overfitting in a competition:

1. **More data** — augmentation, generate synthetic examples
2. **Simpler model** — fewer trees, smaller network, shallower depth
3. **L1 / L2 regularization** — tune the strength
4. **Dropout** (neural networks) — randomly disable neurons during training
5. **Cross-validation** — ensure your validation score is reliable
6. **Early stopping** — stop before overfitting

```python
# Quick diagnostic: compare train vs validation score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_tr, y_tr)

print(f'Train: {accuracy_score(y_tr, model.predict(X_tr)):.3f}')
print(f'Val:   {accuracy_score(y_val, model.predict(X_val)):.3f}')
# Big gap = overfitting
```

---

## Diagnose the Training Problem

```widget
{
  "type": "concept-sort",
  "title": "Is This Overfitting, Underfitting, or Good Fit?",
  "categories": [
    { "name": "Overfitting", "color": "#EF4444" },
    { "name": "Underfitting", "color": "#F97316" },
    { "name": "Good fit", "color": "#22C55E" }
  ],
  "items": [
    { "text": "Train accuracy 99%, validation accuracy 62%", "category": "Overfitting" },
    { "text": "Train accuracy 71%, validation accuracy 70%", "category": "Underfitting" },
    { "text": "Train accuracy 89%, validation accuracy 87%", "category": "Good fit" },
    { "text": "Train loss keeps falling but validation loss starts rising", "category": "Overfitting" },
    { "text": "Both train and validation loss are high and plateau early", "category": "Underfitting" },
    { "text": "Train loss 0.12, validation loss 0.65 — huge gap", "category": "Overfitting" },
    { "text": "Adding more layers makes both train and val loss worse", "category": "Underfitting" },
    { "text": "CV score 0.91, test leaderboard score 0.90 — consistent", "category": "Good fit" }
  ]
}
```

---

## Summary

| Problem | Sign | Fix |
|---|---|---|
| Overfitting | Train >> Val | Regularize, simplify, more data |
| Underfitting | Train ≈ Val, both low | More complex model, more features |
| Good fit | Train ≈ Val, both high | Keep it! |

The overfit-underfit balance is a constant tension in ML. In competitions, overfitting is the more common problem — cross-validate everything and do not look at the test set until the very end.
