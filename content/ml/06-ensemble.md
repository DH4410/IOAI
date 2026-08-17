---
title: Random Forests and Gradient Boosting
track: ml
order: 6
estimatedTime: 55
difficulty: intermediate
---

# Random Forests and Gradient Boosting

Single decision trees overfit. Ensemble methods fix that by combining many trees. **This is the most important lesson for IOAI competitions** — gradient boosting (XGBoost, LightGBM, CatBoost) dominates tabular ML competitions worldwide.

---

## 1. The Ensemble Idea

When you combine many slightly different models, their errors tend to cancel:

```
         Tree 1  Tree 2  Tree 3  Ensemble
Sample 1  WRONG  RIGHT   RIGHT   RIGHT  ← 2/3 correct
Sample 2  RIGHT  RIGHT   WRONG   RIGHT  ← 2/3 correct  
Sample 3  RIGHT  WRONG   RIGHT   RIGHT  ← 2/3 correct
```

**Key requirement:** the errors must be different (uncorrelated). If all models make the same mistakes, averaging doesn't help.

Two strategies to create diverse trees:
- **Bagging** (Bootstrap Aggregating): train each tree on a random sample of the data → Random Forest
- **Boosting**: train each tree on the *errors* of the previous trees → Gradient Boosting

---

## 2. Random Forest: Bagging + Feature Randomness

Random Forest:
1. For each tree (say, 100 trees):
   - Bootstrap sample: sample $n$ rows **with replacement** from training data
   - At each split: only consider $\sqrt{p}$ random features (not all $p$)
2. Each tree grows to full depth (no pruning needed)
3. Final prediction: majority vote (classification) or mean (regression)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

X, y = make_classification(n_samples=2000, n_features=20, n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(
    n_estimators=200,     # more trees = more stable, diminishing returns after ~200
    max_depth=None,       # let trees grow fully — the ensemble handles overfitting
    max_features='sqrt',  # consider sqrt(n_features) at each split
    min_samples_leaf=1,
    n_jobs=-1,            # use all CPU cores
    random_state=42,
    oob_score=True        # out-of-bag score — free validation!
)
rf.fit(X_train, y_train)

print(f"OOB Score:  {rf.oob_score_:.3f}")  # no validation set needed!
print(f"Test Score: {rf.score(X_test, y_test):.3f}")
```

**Out-Of-Bag (OOB) score**: each tree was trained on a bootstrap sample, so ~37% of examples weren't used for each tree. We can use those as free validation — no need to hold out a separate val set!

---

## 3. Feature Importance: What Matters?

```python
import matplotlib.pyplot as plt

feature_names = [f'feat_{i}' for i in range(20)]
importances   = rf.feature_importances_
indices       = np.argsort(importances)[::-1]

# Top 10 features
for rank, i in enumerate(indices[:10]):
    bar = '█' * int(importances[i] * 200)
    print(f"{rank+1:2d}. {feature_names[i]:10s} {importances[i]:.4f} {bar}")
```

**Permutation importance** (more reliable than the default):
```python
from sklearn.inspection import permutation_importance

result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
for i in result.importances_mean.argsort()[::-1][:5]:
    print(f"{feature_names[i]:12s} {result.importances_mean[i]:.4f} ± {result.importances_std[i]:.4f}")
```

---

## 4. Gradient Boosting: Fix Errors, Step by Step

Random Forest trains trees **in parallel** (independent). Gradient Boosting trains trees **sequentially**:

```
Round 1: Fit tree T₁ on y            → residuals r₁ = y - T₁(x)
Round 2: Fit tree T₂ on r₁           → residuals r₂ = r₁ - T₂(x)
Round 3: Fit tree T₃ on r₂           → ...
...
Final: F(x) = T₁(x) + η·T₂(x) + η·T₃(x) + ...
```

$\eta$ (eta) = learning rate. Small η → each tree has a smaller contribution → need more trees → but usually better final result.

**Why gradient?** We're following the gradient of the loss function — that's why it's called *gradient* boosting. For MSE loss, the residuals are literally the gradient. For log loss, it's slightly different but the same concept.

---

## 5. XGBoost — The Competition Standard

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=1000,        # set high — early stopping will find the right number
    max_depth=5,              # 3-6 for most problems
    learning_rate=0.05,       # 0.01-0.1; smaller = slower but often better
    subsample=0.8,            # use 80% of data per tree (reduces overfitting)
    colsample_bytree=0.8,     # use 80% of features per tree
    reg_alpha=0.1,            # L1 regularization on leaf weights
    reg_lambda=1.0,           # L2 regularization on leaf weights
    eval_metric='logloss',
    use_label_encoder=False,
    verbosity=0,
    random_state=42,
    n_jobs=-1
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=30,  # stop if no improvement for 30 rounds
    verbose=False
)

print(f"Best iteration: {model.best_iteration}")
print(f"Val accuracy:   {model.score(X_val, y_val):.3f}")
print(f"Test accuracy:  {model.score(X_test, y_test):.3f}")
```

---

## 6. LightGBM — Faster on Large Data

LightGBM uses **leaf-wise** growth (grows the leaf with maximum gain) instead of level-wise, making it faster and often slightly better:

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=1000,
    max_depth=-1,           # -1 = no limit; control via num_leaves instead
    num_leaves=31,          # key parameter: 2^max_depth as upper bound
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_samples=20,
    verbose=-1,
    random_state=42,
    n_jobs=-1
)
callbacks = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)]
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          callbacks=callbacks)

print(f"Best iteration: {model.best_iteration_}")
print(f"Test accuracy:  {model.score(X_test, y_test):.3f}")
```

---

## 7. CatBoost — Best for Categorical Features

When your data has many categorical columns, CatBoost handles them natively (no one-hot encoding needed):

```python
from catboost import CatBoostClassifier

# Specify which columns are categorical
cat_features = [0, 2, 5]   # column indices of categorical features

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    cat_features=cat_features,
    verbose=0,
    random_seed=42
)
model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=30)
```

---

## 8. Hyperparameter Tuning with Optuna

Manual grid search is slow. **Optuna** does Bayesian optimization — it learns which regions of hyperparameter space are promising:

```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'n_estimators': 500,   # fixed; use early stopping
        'max_depth':    trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample':    trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha':    trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
        'reg_lambda':   trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
        'verbosity': 0
    }
    model = xgb.XGBClassifier(**params, random_state=42)
    scores = cross_val_score(model, X_train_full, y_train_full, cv=5, scoring='accuracy', n_jobs=-1)
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("Best params:", study.best_params)
print("Best CV accuracy:", study.best_value:.3f")
```

---

## 9. Stacking: Ensembles of Ensembles

Stacking trains a "meta-model" on the predictions of multiple base models:

```
Base models (Level 0):                    Meta-model (Level 1):
  XGBoost  ──╮                              ╭── XGBoost  ──╮
  LightGBM ──┤── predictions ──╮           ├── LightGBM ──┼── FINAL PREDICTION
  RandomForest─╯               ╰── LogReg ─╯── RF      ──╯
```

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

estimators = [
    ('xgb', xgb.XGBClassifier(n_estimators=200, verbosity=0, random_state=42)),
    ('lgb', lgb.LGBMClassifier(n_estimators=200, verbose=-1, random_state=42)),
    ('rf',  RandomForestClassifier(n_estimators=200, random_state=42)),
]
stacker = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,             # use 5-fold CV to generate meta-features
    n_jobs=-1
)
stacker.fit(X_train, y_train)
print(f"Stacked accuracy: {stacker.score(X_test, y_test):.3f}")
```

---

## 10. Sort These Techniques

```widget
{
  "type": "concept-sort",
  "title": "Bagging or Boosting?",
  "categories": [
    { "name": "Bagging", "color": "#5B5BD6" },
    { "name": "Boosting", "color": "#F97316" }
  ],
  "items": [
    { "text": "Random Forest", "category": "Bagging" },
    { "text": "XGBoost", "category": "Boosting" },
    { "text": "Trees trained in parallel", "category": "Bagging" },
    { "text": "LightGBM", "category": "Boosting" },
    { "text": "Trees trained sequentially", "category": "Boosting" },
    { "text": "Bootstrap sampling", "category": "Bagging" },
    { "text": "CatBoost", "category": "Boosting" },
    { "text": "OOB score available", "category": "Bagging" }
  ]
}
```

---

## 11. IOAI Competition Strategy

1. **Start with LightGBM or XGBoost** — they're fast and strong baselines
2. **Use early stopping** — never guess `n_estimators`
3. **Use Optuna** for hyperparameter search — 50 trials is usually enough
4. **Check feature importance** — drop features with near-zero importance (speed boost, sometimes better score)
5. **Try stacking** in the final hour — combine your best XGBoost and LightGBM models
6. **Use cross-validation** for reliable scores — single train/val splits can mislead

```python
# IOAI Quick-Start Template
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb, numpy as np

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y_train))   # out-of-fold predictions
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    model = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, num_leaves=31,
        verbose=-1, random_state=fold
    )
    callbacks = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)]
    model.fit(
        X_train[train_idx], y_train[train_idx],
        eval_set=[(X_train[val_idx], y_train[val_idx])],
        callbacks=callbacks
    )
    oof_preds[val_idx] = model.predict(X_train[val_idx])
    test_preds += model.predict(X_test) / 5

from sklearn.metrics import accuracy_score
print(f"CV OOF accuracy: {accuracy_score(y_train, oof_preds):.4f}")
```

---

## Summary

| Method | How | When to use |
|---|---|---|
| Random Forest | Parallel trees on bootstrap samples | Fast baseline, no tuning needed |
| XGBoost | Sequential trees on residuals, L1/L2 reg | Standard competition choice |
| LightGBM | Leaf-wise XGBoost, faster | Large datasets, usually faster than XGBoost |
| CatBoost | Built-in categorical handling | Data with many category columns |
| Stacking | Meta-model on base model predictions | Final 10-20% accuracy improvement |
| Optuna | Bayesian hyperparameter search | When you need the last bit of performance |

**Rule of thumb:** On tabular data, gradient boosting beats neural networks most of the time. Start there.
