---
title: Random Forests and Gradient Boosting
track: ml
order: 6
estimatedTime: 50
difficulty: intermediate
---

# Random Forests and Gradient Boosting

Ensemble methods combine multiple models to produce a prediction better than any individual model. They are the dominant approach in classical ML competitions, frequently appearing at the top of Kaggle leaderboards. This lesson covers the two most important ensemble families: **Random Forests** (bagging) and **Gradient Boosting** (boosting).

---

## The Wisdom of Crowds

Imagine asking 1000 people to guess the weight of a cow at a county fair. Most individual guesses will be wrong, but the **average** of all guesses tends to be remarkably accurate. This is the wisdom of crowds.

The same principle applies to machine learning:
- Individual models make different kinds of errors
- When you average their predictions, the errors tend to cancel out
- The ensemble is more accurate than any single model

This only works if the models are **diverse** — making different mistakes. If all models make the same mistakes, averaging does not help.

---

## Bagging vs. Boosting

There are two main strategies for building diverse ensembles:

| Approach | Key Idea | Model Dependency |
|---|---|---|
| **Bagging** | Train models in parallel on different bootstrap samples | Independent |
| **Boosting** | Train models sequentially, each correcting the errors of the previous | Sequential |

---

## Bootstrap Sampling

A **bootstrap sample** is a sample drawn with replacement from the training data, of the same size as the original training set.

On average, about 63.2% of the original examples appear in each bootstrap sample (some appear multiple times, others not at all).

The examples that do NOT appear are called **out-of-bag (OOB)** samples — they can be used as a free validation set!

```python
import numpy as np

def bootstrap_sample(X, y, random_state=None):
    """Create one bootstrap sample from (X, y)."""
    rng = np.random.RandomState(random_state)
    n = len(y)
    indices = rng.randint(0, n, size=n)   # sample WITH replacement
    return X[indices], y[indices], indices

np.random.seed(42)
n = 10
X = np.arange(n).reshape(-1, 1)
y = np.arange(n)

# Create bootstrap sample
X_boot, y_boot, idx = bootstrap_sample(X, y, random_state=7)

print("Original indices: ", np.arange(n))
print("Bootstrap indices:", idx)
print("In-bag:           ", np.sort(np.unique(idx)))
print("Out-of-bag (OOB): ", np.setdiff1d(np.arange(n), idx))

# Verify: ~63.2% appear in each bootstrap sample
n_trials = 10000
fraction_in_bag = []
for trial in range(n_trials):
    boot_idx = np.random.randint(0, n, size=n)
    fraction_in_bag.append(len(np.unique(boot_idx)) / n)

print(f"\nAverage fraction in bag over {n_trials} trials: {np.mean(fraction_in_bag):.4f}")
print(f"(Theory: 1 - (1 - 1/n)^n → 1 - 1/e ≈ {1 - np.exp(-1):.4f})")
```

---

## Random Forests: Bagging + Feature Randomness

A **Random Forest** combines two ideas:
1. **Bagging**: Train each tree on a different bootstrap sample
2. **Feature randomness**: At each split, consider only a random subset of features (typically √p for classification, p/3 for regression)

The feature randomness is crucial — it makes the trees less correlated with each other, which improves the ensemble.

```
For each of n_estimators trees:
  1. Draw a bootstrap sample
  2. Grow a decision tree where each split:
     - Considers only max_features random features
     - Picks the best split among those features
Predict: majority vote (classification) or mean (regression)
```

```python
import numpy as np

class SimpleRandomForest:
    """Random Forest classifier — educational implementation."""

    def __init__(self, n_estimators=100, max_depth=None,
                 max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self.max_features = max_features
        self.rng          = np.random.RandomState(random_state)
        self.trees        = []
        self.feature_subsets = []

    def _get_n_features(self, p):
        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(p)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(p)))
        elif isinstance(self.max_features, int):
            return self.max_features
        return p

    def fit(self, X, y):
        from sklearn.tree import DecisionTreeClassifier
        n, p = X.shape
        n_feat = self._get_n_features(p)

        for i in range(self.n_estimators):
            # Bootstrap sample
            boot_idx = self.rng.randint(0, n, size=n)
            X_boot   = X[boot_idx]
            y_boot   = y[boot_idx]

            # Random feature subset for this tree
            feat_idx = self.rng.choice(p, size=n_feat, replace=False)
            self.feature_subsets.append(feat_idx)

            # Train a tree on the bootstrap sample with selected features
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                          random_state=self.rng.randint(0, 10000))
            tree.fit(X_boot[:, feat_idx], y_boot)
            self.trees.append(tree)

        return self

    def predict(self, X):
        # Collect predictions from all trees
        all_preds = np.zeros((len(X), self.n_estimators), dtype=int)
        for i, (tree, feat_idx) in enumerate(zip(self.trees, self.feature_subsets)):
            all_preds[:, i] = tree.predict(X[:, feat_idx])
        # Majority vote
        from scipy import stats
        return stats.mode(all_preds, axis=1)[0].flatten()

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# Compare our implementation with sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

np.random.seed(42)
X, y = make_classification(n_samples=500, n_features=10, n_informative=6, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

our_rf = SimpleRandomForest(n_estimators=50, max_depth=5, random_state=42)
our_rf.fit(X_train, y_train)

sk_rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
sk_rf.fit(X_train, y_train)

print(f"Our RF test accuracy:    {our_rf.score(X_test, y_test):.4f}")
print(f"Sklearn RF test accuracy: {sk_rf.score(X_test, y_test):.4f}")
```

---

## Sklearn RandomForestClassifier

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,         # let trees grow fully
    max_features='sqrt',    # use sqrt(n_features) features per split
    min_samples_leaf=1,
    bootstrap=True,
    oob_score=True,         # compute out-of-bag score
    n_jobs=-1,              # use all CPU cores
    random_state=42
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(f"Test accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"OOB accuracy:   {rf.oob_score_:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=data.target_names)}")
```

---

## Feature Importance from Random Forests

Random forests give more reliable feature importance than a single decision tree because the importance is averaged across many trees.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X, y = data.data, data.target

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X, y)

importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

print("Top 10 features by importance (with std across trees):")
print(f"{'Rank':>4} {'Feature':>30} {'Importance':>12} {'Std':>8}")
print("-" * 58)
for rank in range(10):
    i = indices[rank]
    bar = "#" * int(importances[i] * 60)
    print(f"{rank+1:>4} {data.feature_names[i]:>30} "
          f"{importances[i]:>12.4f} ±{std[i]:.4f}  {bar}")
```

---

## Out-of-Bag (OOB) Error

Since each tree is trained on a bootstrap sample (~63% of data), the other ~37% are "out-of-bag" for that tree. We can use these as a validation set — for free, without needing to set aside data.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

np.random.seed(42)
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# OOB error vs. number of trees
print(f"{'n_trees':>8} {'OOB acc':>10} {'Test acc':>10}")
print("-" * 32)
for n in [10, 25, 50, 100, 200, 500]:
    rf = RandomForestClassifier(n_estimators=n, oob_score=True, random_state=42)
    rf.fit(X_train, y_train)
    print(f"{n:>8} {rf.oob_score_:>10.4f} {rf.score(X_test, y_test):>10.4f}")

print("\nOOB score is a free, nearly unbiased estimate of test accuracy!")
print("It tends to converge as n_trees increases.")
```

---

## Gradient Boosting: The Sequential Learner

Boosting takes a completely different approach. Instead of training trees independently, it trains them **sequentially**. Each new tree focuses on the examples that the previous trees got wrong.

The key insight: **fit each new tree to the residuals of the current ensemble**.

**Algorithm:**

```
1. Start with a simple prediction (e.g., the mean of y for regression)
2. Compute the residuals: e = y - current_prediction
3. Train a shallow tree to predict the residuals
4. Update: new_prediction = current_prediction + learning_rate * tree_prediction
5. Repeat from step 2
```

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

class SimpleGradientBoosting:
    """Gradient boosting for regression — educational implementation."""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.max_depth     = max_depth
        self.trees         = []
        self.init_pred     = None

    def fit(self, X, y):
        # Initial prediction: the mean
        self.init_pred = y.mean()
        current_pred   = np.full(len(y), self.init_pred)

        for i in range(self.n_estimators):
            # Residuals = negative gradient of MSE = y - current_prediction
            residuals = y - current_pred

            # Fit a shallow tree to the residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)

            # Update prediction
            current_pred += self.learning_rate * tree.predict(X)

        return self

    def predict(self, X):
        pred = np.full(X.shape[0], self.init_pred)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return pred

    def loss_history(self, X, y):
        """Compute MSE after each tree is added."""
        pred  = np.full(len(y), self.init_pred)
        losses = [np.mean((y - pred)**2)]
        for tree in self.trees:
            pred  += self.learning_rate * tree.predict(X)
            losses.append(np.mean((y - pred)**2))
        return losses


# Test on regression problem
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

np.random.seed(42)
X, y = make_regression(n_samples=500, n_features=10, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gb = SimpleGradientBoosting(n_estimators=200, learning_rate=0.1, max_depth=3)
gb.fit(X_train, y_train)

y_pred = gb.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"Simple Gradient Boosting R²: {r2:.4f}")

# Show loss decreasing
losses = gb.loss_history(X_train, y_train)
print("\nTraining MSE at various stages:")
for n in [0, 10, 50, 100, 199]:
    print(f"  After {n:>3} trees: MSE = {losses[n]:.2f}")
```

---

## Sklearn's GradientBoostingClassifier

```python
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,       # stochastic GBM: use 80% of data per tree
    random_state=42
)
gb.fit(X_train, y_train)

print(f"GradientBoostingClassifier test accuracy: {gb.score(X_test, y_test):.4f}")
print(f"n_estimators used: {gb.n_estimators_}")

# Staged predictions: performance after each tree
test_scores = [accuracy_score(y_test, pred)
               for pred in gb.staged_predict(X_test)]

print("\nAccuracy after N trees:")
for n in [1, 5, 10, 25, 50, 100, 200]:
    print(f"  {n:>3} trees: {test_scores[n-1]:.4f}")
```

---

## XGBoost and LightGBM

In practice, XGBoost and LightGBM are the go-to gradient boosting libraries. They are dramatically faster than sklearn's implementation and often achieve better performance through:

- Second-order gradient approximations
- Built-in regularization (L1, L2, tree complexity)
- Histogram-based splitting (LightGBM is especially fast)
- Native handling of missing values (XGBoost)
- Parallel tree construction

```python
# XGBoost example
try:
    import xgboost as xgb
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import numpy as np

    np.random.seed(42)
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )

    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    print(f"XGBoost test accuracy: {xgb_model.score(X_test, y_test):.4f}")

except ImportError:
    print("XGBoost not installed. Install with: pip install xgboost")
    print("The sklearn GradientBoostingClassifier is equivalent for learning purposes.")

# LightGBM example
try:
    import lightgbm as lgb
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import numpy as np

    np.random.seed(42)
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lgb_model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        num_leaves=31,
        subsample=0.8,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    print(f"LightGBM test accuracy: {lgb_model.score(X_test, y_test):.4f}")

except ImportError:
    print("LightGBM not installed. Install with: pip install lightgbm")
```

---

## Key Hyperparameters That Matter

Understanding which hyperparameters to tune is essential for competitions.

### Random Forest

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

np.random.seed(42)
X, y = make_classification(n_samples=500, n_features=15, n_informative=8, random_state=42)

print("Random Forest Hyperparameter Sensitivity:")
print(f"\n{'Parameter':>20} {'Value':>10} {'CV Accuracy':>12}")
print("-" * 46)

# n_estimators: more is always better (or neutral)
for n in [10, 50, 100, 200, 500]:
    rf = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    score = cross_val_score(rf, X, y, cv=5).mean()
    print(f"{'n_estimators':>20} {n:>10} {score:>12.4f}")

print()
# max_features: controls diversity
for mf in ['sqrt', 'log2', 0.3, 0.5, 1.0]:
    rf = RandomForestClassifier(n_estimators=100, max_features=mf, random_state=42)
    score = cross_val_score(rf, X, y, cv=5).mean()
    print(f"{'max_features':>20} {str(mf):>10} {score:>12.4f}")

print()
# max_depth: unlimited usually works for RF (unlike single trees)
for d in [3, 5, 10, 20, None]:
    rf = RandomForestClassifier(n_estimators=100, max_depth=d, random_state=42)
    score = cross_val_score(rf, X, y, cv=5).mean()
    print(f"{'max_depth':>20} {str(d):>10} {score:>12.4f}")
```

### Gradient Boosting

```python
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

np.random.seed(42)
X, y = make_classification(n_samples=500, n_features=15, n_informative=8, random_state=42)

print("Gradient Boosting Hyperparameter Sensitivity:")
print(f"\n{'Parameter':>20} {'Value':>12} {'CV Accuracy':>12}")
print("-" * 48)

# learning_rate: smaller = slower but often better (need more trees)
for lr in [0.5, 0.2, 0.1, 0.05, 0.01]:
    gb = GradientBoostingClassifier(
        n_estimators=200, learning_rate=lr,
        max_depth=3, random_state=42
    )
    score = cross_val_score(gb, X, y, cv=5).mean()
    print(f"{'learning_rate':>20} {lr:>12.3f} {score:>12.4f}")

print()
# max_depth: shallow trees (2-5) work best for boosting
for d in [1, 2, 3, 4, 5, 8]:
    gb = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1,
        max_depth=d, random_state=42
    )
    score = cross_val_score(gb, X, y, cv=5).mean()
    print(f"{'max_depth':>20} {d:>12} {score:>12.4f}")
```

> **Key insight:** For gradient boosting, learning rate and n_estimators are linked: use a small learning rate (0.01–0.1) with many trees (500–2000). For random forests, more trees never hurt — just slower.

---

## Why Ensembles Win Kaggle Competitions

The mathematical reason ensembles work:

If we have M models with error variance σ² and pairwise correlation ρ between them, the ensemble variance is:

```
Var(ensemble) = (1/M) σ² + (1 - 1/M) ρ σ²
             ≈ ρ σ²  (for large M)
```

The key insight: **as long as ρ < 1 (models are not perfectly correlated), the ensemble has lower variance than any individual model**.

Random forests reduce ρ through bootstrap sampling and feature randomness.
Gradient boosting reduces bias by iteratively correcting mistakes.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, StratifiedKFold

np.random.seed(42)
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10, random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Single Tree (d=5)':   DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest (200)': RandomForestClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42),
}

print(f"{'Model':>25} {'CV Mean':>10} {'CV Std':>10}")
print("-" * 48)
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv)
    print(f"{name:>25} {scores.mean():>10.4f} ±{scores.std():>8.4f}")
```

---

## Bagging vs. Boosting: When to Use Which

| Situation | Prefer |
|---|---|
| High variance (overfitting) | Bagging (Random Forest) |
| High bias (underfitting) | Boosting (GBM, XGBoost) |
| Noisy data | Bagging (more robust to noise) |
| Clean data, tabular structure | Boosting (usually wins) |
| Need fast training | Random Forest (parallel) |
| Need best possible accuracy | Gradient Boosting |
| Need OOB validation for free | Random Forest |

---

## Summary

| Concept | Key Idea |
|---|---|
| Ensemble | Combine multiple models for better predictions |
| Bootstrap sample | Sample with replacement, ~63.2% unique examples |
| Bagging | Train in parallel on bootstrap samples, average predictions |
| Random Forest | Bagging + random feature subset per split |
| OOB error | Free validation using the ~37% out-of-bag examples |
| Gradient Boosting | Sequential: each tree fits residuals of the ensemble |
| Learning rate (GBM) | Shrinks each tree's contribution — use small values |
| n_estimators | More trees: RF is always better; GBM needs tuning |
| max_depth | RF: larger ok; GBM: shallow (2–5) usually best |

Ensemble methods are your main weapon for tabular competition data. The next lesson focuses on **evaluation metrics** — because the best model is only as good as the metric you optimize for.
