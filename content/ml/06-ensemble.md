---
title: Random Forests and Gradient Boosting
track: ml
order: 6
estimatedTime: 40
difficulty: intermediate
---

# Random Forests and Gradient Boosting

A single decision tree is easy to understand but not very accurate. Ensemble methods fix this by combining many trees. They are the most powerful tool for tabular data in competitions.

---

## 1. The Main Idea

When you ask many slightly different models to make predictions and average them, the errors cancel out and the average is more accurate than any single model.

This only works if the models make **different** errors. If all models make the same mistake, averaging changes nothing.

**Random Forest** and **Gradient Boosting** are two ways to build diverse trees that complement each other.

---

## 2. Random Forests

A Random Forest trains many decision trees, each on a different random subset of the data and a random subset of the features. Then it takes a majority vote (classification) or average (regression).

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(
    n_estimators=100,    # number of trees
    max_depth=None,      # trees can grow fully (forest resists overfitting)
    random_state=42
)
rf.fit(X_train, y_train)
print(f'Accuracy: {rf.score(X_test, y_test):.2%}')
```

**Quick check:** Why does a Random Forest resist overfitting even when each tree is fully grown?
> Each tree overfits to a slightly different random sample. Their errors are different. When you average them, the overfit errors cancel out.

Feature importance works the same way as a single tree:

```python
importances = rf.feature_importances_
print(dict(zip(['sepal len', 'sepal wid', 'petal len', 'petal wid'], importances.round(3))))
```

---

## 3. Gradient Boosting

Gradient boosting builds trees **sequentially**. Each new tree focuses on fixing the mistakes of all the previous trees.

1. Train a small tree on the data
2. Look at where it was wrong (the residuals)
3. Train the next tree on those residuals
4. Repeat 100-1000 times
5. Final prediction = sum of all trees

This process corrects errors step by step, which is why it often outperforms Random Forests.

---

## 4. XGBoost and LightGBM

In practice, people use optimized libraries rather than scikit-learn's `GradientBoostingClassifier`. The two most common in competitions:

**XGBoost** - very popular, well-tested:

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)
model.fit(X_train, y_train)
print(f'Accuracy: {model.score(X_test, y_test):.2%}')
```

**LightGBM** - faster on large datasets:

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    verbose=-1
)
model.fit(X_train, y_train)
print(f'Accuracy: {model.score(X_test, y_test):.2%}')
```

Both follow the scikit-learn API (`.fit`, `.predict`, `.score`).

---

## 5. Key Hyperparameters

These matter most for gradient boosting:

| Parameter | What it does | Typical values |
|---|---|---|
| `n_estimators` | Number of trees | 100-1000 |
| `max_depth` | Tree depth | 3-6 |
| `learning_rate` | Step size (smaller = more trees needed) | 0.01-0.3 |
| `subsample` | Fraction of data per tree | 0.6-1.0 |

The key tradeoff: lower `learning_rate` with higher `n_estimators` usually gives better results but is slower.

```python
# A solid default setup
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
    verbosity=0
)
```

**Quick check:** You increase `n_estimators` from 100 to 500 and training time goes up a lot but accuracy barely improves. What could help?
> Lower `learning_rate` (e.g., 0.01) so more trees are actually needed. Or use early stopping to find the right number of trees automatically.

---

## 6. Early Stopping

Instead of guessing `n_estimators`, use early stopping: stop training when the validation score stops improving.

```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

model = xgb.XGBClassifier(
    n_estimators=2000,     # set high - early stopping will stop before this
    max_depth=4,
    learning_rate=0.05,
    verbosity=0
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=20,    # stop if no improvement for 20 rounds
    verbose=False
)
print(f'Best round: {model.best_iteration}')
print(f'Val accuracy: {model.score(X_val, y_val):.2%}')
```

---

## 7. Random Forest vs. Gradient Boosting

| | Random Forest | Gradient Boosting (XGB/LGB) |
|---|---|---|
| Training | Trees in parallel | Trees in sequence |
| Speed | Faster to train | Slower (sequential) |
| Accuracy | Usually slightly lower | Usually higher |
| Overfitting | Hard to overfit | Easier to overfit without tuning |
| Tuning needed | Minimal | More (lr, depth, estimators) |

**Which to use?** In competitions: try both. XGBoost/LightGBM usually win on tabular data when tuned. Random Forest is a great quick baseline.

---

## Summary

- **Random Forest:** many trees on random subsets, take a vote. Easy to use, hard to overfit.
- **Gradient Boosting:** trees one at a time, each fixes previous errors. More accurate but needs tuning.
- Use `xgboost` or `lightgbm` in practice (both use sklearn API).
- Key parameters: `n_estimators`, `max_depth`, `learning_rate`.
- Use early stopping to find the right `n_estimators` automatically.
