---
title: Final Checklist and Competition Preparation Guide
track: ioai
order: 7
estimatedTime: 60
difficulty: advanced
---

# Final Checklist and Competition Preparation Guide

This is the practical handbook you read in the final week before IOAI. It synthesizes everything from the previous lessons into actionable checklists, quick-reference code, and the mental frameworks that differentiate students who perform well under pressure from those who do not.

---

## Section 1: The 48-Hour Pre-Competition Checklist

### Technical Preparation

The goal of the 48 hours before competition is not to learn new material — it is to verify your tools work and consolidate what you know.

**Hardware and environment:**
- [ ] Confirm access to the competition platform or environment
- [ ] Verify Python version (3.10+ recommended for IOAI)
- [ ] Confirm these libraries are available and at working versions:
  - `numpy` (array operations, linear algebra)
  - `pandas` (data loading, manipulation)
  - `scikit-learn` (models, preprocessing, evaluation)
  - `scipy` (sparse matrices for NLP, scientific computing)
- [ ] Run a quick sanity test of each library (see code below)
- [ ] If GPU access is provided: confirm CUDA drivers and torch/tensorflow work

```python
# 48-hour sanity test — run this before competition day
import numpy as np
import pandas as pd
import sklearn
import scipy

print("numpy version:", np.__version__)
print("pandas version:", pd.__version__)
print("sklearn version:", sklearn.__version__)
print("scipy version:", scipy.__version__)

# Quick functional test
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(n_estimators=50, random_state=42)
scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
assert scores.mean() > 0.90, "Sanity check failed: RandomForest on Iris should be >90%"
print(f"Sanity check passed. Iris accuracy: {scores.mean():.3f}")
```

**Code templates:**
- [ ] Have your baseline tabular pipeline code accessible (memorized or in a personal notes file)
- [ ] Have your NLP pipeline template accessible
- [ ] Have your CV feature extraction template accessible
- [ ] Verify you know how to write a submission CSV from scratch

**Mental preparation:**
- [ ] Get 8 hours of sleep both nights before competition
- [ ] Eat a full breakfast on competition day — cognitive performance degrades sharply with hunger
- [ ] Avoid new foods or caffeine amounts significantly different from your baseline
- [ ] Brief meditation or breathing exercise if that helps you (5 minutes of slow breathing reduces cortisol measurably)
- [ ] Review your notes, do not try to learn new techniques

### The Night Before

Do these in order:

1. Re-read the problem statement format from previous IOAI competitions to calibrate expectations
2. Review your pipeline templates — not to memorize, just to refresh
3. Write down on paper: the 5 most important sklearn functions you use most
4. Confirm your alarm is set and your commute/login plan is ready
5. Stop studying by 9 PM. Rest is worth more than one more hour of study at this point

---

## Section 2: Competition Day Timeline

### First 15 Minutes of Each Problem

These 15 minutes set the direction for everything that follows. Do not rush them, but do not linger either.

```
Minutes 0-3:   Read the entire problem statement once, quickly.
               Note: task type (classification/regression/NLP/CV),
               evaluation metric, dataset size, output format.

Minutes 3-7:   Read again slowly. Highlight:
               - Exact metric (f1_macro? accuracy? log_loss? RMSE?)
               - Any special instructions (handle missing values,
                 class weights specified, time-series ordering)
               - Submission format (columns, types, order)

Minutes 7-12:  Load data. Run: shape, dtypes, head, describe,
               value_counts on target.

Minutes 12-15: Write your first three lines of model code.
               By minute 15, you should have a model.fit() call
               ready to run.
```

### When to Submit

- Submit your baseline before any optimization (target: first submission by minute 45)
- Submit after every meaningful improvement that you have validated locally
- Make a "safety submission" in the final 15 minutes — a version you are confident works, even if it is not your best model

### Handling Unexpected Problem Types

If you encounter a problem type you have not prepared for:

1. **Breathe.** Everyone else is also seeing it for the first time.
2. **Identify the output type**: classification, regression, ranking, generation?
3. **Apply general ML principles**: load data, EDA, baseline, iterate
4. **Start with the simplest possible model**: even a single-feature logistic regression gives you a submitted solution
5. **Read the problem statement again**: you probably missed something useful

The correct response to an unfamiliar task is never "freeze and do nothing." It is "apply the baseline pipeline and then think about task-specific improvements."

---

## Section 3: Quick Reference — Most Important Algorithms

These are the algorithms you must know cold: what they are, when to use them, and their key hyperparameters.

### LinearRegression

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Plain OLS — no regularization
lr = LinearRegression()

# Ridge (L2) — good default for regression, prevents overfitting
ridge = Ridge(alpha=1.0)          # alpha: regularization strength

# Lasso (L1) — feature selection via sparsity
lasso = Lasso(alpha=0.01)

# Use Ridge as your default regression baseline
```

When to use: continuous target, numerical features, fast training needed.

### LogisticRegression

```python
from sklearn.linear_model import LogisticRegression

# Binary or multi-class classification
logreg = LogisticRegression(
    C=1.0,              # inverse of regularization strength (smaller C = stronger reg)
    penalty="l2",       # "l1" for sparse models, "l2" for default
    solver="lbfgs",     # "lbfgs" for small data, "saga" for L1 or large data
    max_iter=1000,      # increase if convergence warnings appear
    multi_class="auto", # handles multi-class automatically
    class_weight=None,  # use "balanced" for imbalanced classes
)
```

When to use: classification baseline, text features (TF-IDF), interpretability needed.

### RandomForestClassifier / RandomForestRegressor

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

rf = RandomForestClassifier(
    n_estimators=200,       # number of trees — more is almost always better
    max_depth=None,         # None = grow until pure; limit for less overfit
    min_samples_leaf=1,     # minimum samples per leaf
    max_features="sqrt",    # "sqrt" for classification, "1.0" for regression
    class_weight="balanced",# use when classes are imbalanced
    random_state=42,
    n_jobs=-1,              # use all CPU cores
)
```

When to use: strong baseline for tabular data, robust to feature scale, handles missing values (some implementations), good feature importances.

### GradientBoostingClassifier

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=200,       # number of boosting rounds
    learning_rate=0.05,     # smaller = more rounds needed but often better
    max_depth=4,            # tree depth — deeper = more complex interactions
    subsample=0.8,          # fraction of samples per tree (stochastic GB)
    min_samples_leaf=20,    # prevents overfitting on small datasets
    random_state=42,
)
```

When to use: highest performance on tabular data; train time is slower than RF; most sensitive to hyperparameters.

### MLPClassifier

```python
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),  # tuple specifies layer sizes
    activation="relu",             # "relu", "tanh", "logistic"
    solver="adam",                 # "adam" for general use
    alpha=0.001,                   # L2 regularization
    learning_rate="adaptive",      # adjusts LR when loss plateaus
    max_iter=500,
    random_state=42,
)
# IMPORTANT: Scale your features before using MLP
# Always pair with StandardScaler in a pipeline
```

When to use: large datasets with many features, complex non-linear relationships.

---

## Section 4: Quick Reference — Most Important Preprocessing

### Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler: mean=0, std=1 (most common)
# Use for: logistic regression, SVM, MLP, KNN
scaler = StandardScaler()

# MinMaxScaler: scale to [0, 1]
# Use for: when you need bounded values
mm_scaler = MinMaxScaler()

# RobustScaler: uses IQR instead of std
# Use for: data with many outliers
robust_scaler = RobustScaler()

# RULE: Always fit on training data only, transform train AND test
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# BETTER: Put in a Pipeline so it fits inside CV folds automatically
pipeline = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])
```

### Imputation

```python
from sklearn.impute import SimpleImputer, KNNImputer

# SimpleImputer — fast, good default
imputer_num = SimpleImputer(strategy="median")         # for numerical
imputer_cat = SimpleImputer(strategy="most_frequent")  # for categorical

# KNNImputer — uses K nearest neighbors; slower but more accurate
knn_imputer = KNNImputer(n_neighbors=5)

# Check for missing values first
print(X.isnull().sum()[X.isnull().sum() > 0])
```

### Encoding

```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder

# OneHotEncoder: for nominal categoricals (no natural order)
# Creates one binary column per category
ohe = OneHotEncoder(
    handle_unknown="ignore",  # ignore categories not seen in training
    sparse_output=False,      # return dense array (easier to work with)
    drop="first",             # drop first category to avoid multicollinearity
)

# OrdinalEncoder: for ordinal categoricals (small, medium, large)
oe = OrdinalEncoder()

# LabelEncoder: for TARGET variable only (not features)
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # converts string labels to integers

# After prediction:
# y_pred_labels = le.inverse_transform(y_pred_encoded)
```

### The Universal ColumnTransformer Template

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Identify column types
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
])

cat_pipe = Pipeline([
    ("impute",  SimpleImputer(strategy="most_frequent")),
    ("encode",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])

# Use in any pipeline
full_pipeline = Pipeline([
    ("prep",  preprocessor),
    ("model", GradientBoostingClassifier(n_estimators=200, random_state=42)),
])
```

---

## Section 5: Quick Reference — Evaluation

### Cross-Validation

```python
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, KFold,
    GridSearchCV, RandomizedSearchCV,
)

# The standard CV setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate a pipeline
scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_macro")
print(f"CV F1-macro: {scores.mean():.4f} ± {scores.std():.4f}")

# Available scoring strings
# Classification: "accuracy", "f1", "f1_macro", "f1_weighted",
#                 "roc_auc", "neg_log_loss", "precision", "recall"
# Regression:     "neg_mean_squared_error", "neg_mean_absolute_error",
#                 "r2", "neg_root_mean_squared_error"
```

### Hyperparameter Search

```python
# Grid search — exhaustive, use for small parameter spaces
param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [3, 4, 5, None],
    "model__learning_rate": [0.01, 0.05, 0.1],
}
grid_search = GridSearchCV(
    pipeline, param_grid,
    cv=cv, scoring="f1_macro",
    n_jobs=-1, verbose=1,
)
grid_search.fit(X, y)
print("Best params:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Randomized search — sample from distributions, better for large spaces
from scipy.stats import randint, uniform
param_dist = {
    "model__n_estimators": randint(50, 500),
    "model__max_depth": randint(2, 10),
    "model__learning_rate": uniform(0.01, 0.19),
}
rand_search = RandomizedSearchCV(
    pipeline, param_dist,
    n_iter=30, cv=cv, scoring="f1_macro",
    random_state=42, n_jobs=-1,
)
rand_search.fit(X, y)
```

### Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score,
    classification_report, confusion_matrix,
)

y_pred      = model.predict(X_val)
y_pred_prob = model.predict_proba(X_val)

print("Accuracy:    ", accuracy_score(y_val, y_pred))
print("F1 (macro):  ", f1_score(y_val, y_pred, average="macro"))
print("F1 (weighted)", f1_score(y_val, y_pred, average="weighted"))
print("\nClassification Report:\n", classification_report(y_val, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))

# For binary classification:
# roc_auc_score(y_val, y_pred_prob[:, 1])

# For multi-class AUC:
# roc_auc_score(y_val, y_pred_prob, multi_class="ovr", average="macro")
```

---

## Section 6: The 50-Line Pipeline That Handles Most Tabular Problems

This is the single most valuable piece of code in this entire course. Practice it until you can type it from memory in under 10 minutes.

```python
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score

# --- Load and inspect --------------------------------------------------------
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")
TARGET = "label"  # adjust
ID_COL = "id"     # adjust or remove

X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
y = train[TARGET]
X_test = test.drop(columns=[ID_COL], errors="ignore")

print(X.shape, y.value_counts(normalize=True).to_dict())

# --- Column types ------------------------------------------------------------
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

# --- Preprocessor ------------------------------------------------------------
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
    ]), num_cols),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), cat_cols),
])

# --- Models ------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in [
    ("LogReg", LogisticRegression(C=1.0, max_iter=1000)),
    ("RF",     RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ("GB",     GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                           max_depth=4, random_state=42)),
]:
    pipe = Pipeline([("prep", preprocessor), ("model", clf)])
    s = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro")
    print(f"{name:8s}  CV F1: {s.mean():.4f} ± {s.std():.4f}")

# --- Fit best model and submit -----------------------------------------------
best_pipe = Pipeline([
    ("prep",  preprocessor),
    ("model", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                          max_depth=4, random_state=42)),
])
best_pipe.fit(X, y)
preds = best_pipe.predict(X_test)

sub = pd.DataFrame({ID_COL: test[ID_COL] if ID_COL in test.columns else range(len(test)),
                    TARGET: preds})
sub.to_csv("submission.csv", index=False)
print("Submitted:", sub.shape)
```

This template handles: numerical + categorical features, missing values, scaling, multi-class classification, and produces a submission file. It covers the mechanics of approximately 85% of IOAI-style practical problems.

---

## Section 7: Feature Engineering Heuristics

Before running a grid search, consider whether any of these transformations would help your specific dataset.

### Numerical Feature Transforms

```python
import numpy as np
import pandas as pd

# Log transform for right-skewed features (income, price, count)
# Only if all values are positive
X["log_income"] = np.log1p(X["income"])  # log(1 + x) handles zeros

# Square root for count data (Poisson-like features)
X["sqrt_count"] = np.sqrt(X["count"])

# Power transform for any skew direction
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method="yeo-johnson")  # handles negative values
X_transformed = pt.fit_transform(X[num_cols])

# Binning — when continuous feature has threshold effects
X["age_bucket"] = pd.cut(X["age"], bins=[0, 25, 40, 60, 100],
                          labels=["young", "adult", "middle", "senior"])

# Z-score clipping — remove extreme outliers
for col in num_cols:
    mean, std = X[col].mean(), X[col].std()
    X[col] = X[col].clip(mean - 3*std, mean + 3*std)
```

### Interaction Features

```python
# Multiplicative interaction
X["income_x_age"] = X["income"] * X["age"]

# Ratio feature — normalize for group effects
X["income_per_family_member"] = X["income"] / (X["family_size"] + 1)

# Difference feature
X["balance_diff"] = X["current_balance"] - X["previous_balance"]

# Group statistics — encode group-level information
group_mean = X.groupby("category")["value"].transform("mean")
X["value_vs_group_mean"] = X["value"] / (group_mean + 1e-9)
```

### Date/Time Features

```python
# Parse date column and extract components
X["date"] = pd.to_datetime(X["date"])
X["year"]      = X["date"].dt.year
X["month"]     = X["date"].dt.month
X["dayofweek"] = X["date"].dt.dayofweek   # 0=Monday, 6=Sunday
X["dayofyear"] = X["date"].dt.dayofyear
X["is_weekend"]= (X["date"].dt.dayofweek >= 5).astype(int)
X["quarter"]   = X["date"].dt.quarter

# Days since a reference date
reference = pd.Timestamp("2020-01-01")
X["days_since_ref"] = (X["date"] - reference).dt.days
```

### NLP Quick Wins

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Fastest NLP baseline
nlp_baseline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                               sublinear_tf=True)),
    ("model", LogisticRegression(C=1.0, max_iter=1000)),
])
# This gets you 80-90% of peak TF-IDF performance in under 3 lines

# Text features: character-level n-grams catch morphology and typos
tfidf_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                              max_features=100000)
```

### CV Quick Wins

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

# If you have pre-computed CNN features:
X_features = np.load("image_features.npy")  # (N, feature_dim)
X_norm = normalize(X_features)               # L2 normalize
cv_model = LogisticRegression(C=1.0, max_iter=1000)
# cross_val_score(cv_model, X_norm, y, cv=5, scoring="f1_macro")

# Simple pixel features if CNN features not available:
def flatten_normalize(images):
    """
    images: (N, H, W, C) array
    Returns: (N, H*W*C) normalized array
    """
    flat = images.reshape(len(images), -1).astype(float)
    flat /= 255.0   # normalize to [0, 1]
    return flat
```

---

## Section 8: Ensembling Quick Reference

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

# Simple probability averaging ensemble
def ensemble_predict(models_and_weights, X):
    """
    Average probability predictions from multiple models.
    models_and_weights: list of (model, weight) tuples
    Returns: probability array (n_samples, n_classes)
    """
    total_weight = sum(w for _, w in models_and_weights)
    ensemble_proba = sum(
        w * m.predict_proba(X) for m, w in models_and_weights
    ) / total_weight
    return ensemble_proba

# Example usage
models_and_weights = [
    (trained_rf_pipeline, 0.3),
    (trained_gb_pipeline, 0.5),
    (trained_lr_pipeline, 0.2),
]
final_proba = ensemble_predict(models_and_weights, X_test)
final_preds = np.argmax(final_proba, axis=1)

# Calibrate ensemble weights using validation set
from sklearn.metrics import f1_score

best_weights, best_score = None, 0.0
# Example: try 11 weight combinations for a 2-model ensemble
for w1 in np.linspace(0, 1, 11):
    blended = w1 * proba_A + (1 - w1) * proba_B
    preds   = np.argmax(blended, axis=1)
    score   = f1_score(y_val, preds, average="macro")
    if score > best_score:
        best_score, best_weights = score, (w1, 1 - w1)
print(f"Best blend weights: {best_weights}, F1: {best_score:.4f}")
```

---

## Section 9: Motivation and What IOAI Participation Means

Participating in IOAI — regardless of your medal outcome — places you in a rare category of students worldwide who have engaged deeply with artificial intelligence at a competition level. This matters in several concrete ways.

### Academic Pathways

IOAI participation signals to university admissions committees (and graduate school admissions committees) that you can apply AI knowledge under pressure, with real datasets. This is distinct from classroom AI — it demonstrates the practical, problem-solving dimension of the field.

Top universities with ML/AI research programs actively seek students with olympiad backgrounds in quantitative fields. IOAI adds AI to that list alongside IMO (math), IPhO (physics), and IOI (informatics).

### Professional Pathways

The skills you developed preparing for IOAI — data preprocessing, model selection, validation, fast iteration — are directly transferable to:
- ML engineering roles (building production ML systems)
- Data science roles (analysis, predictive modeling)
- AI research (understanding existing literature, proposing new approaches)

Companies that actively hire from competitive programming and olympiad backgrounds include most major AI labs and tech companies.

### The Deeper Purpose

Beyond career outcomes, the discipline of competition-level study builds something more durable: the ability to think clearly about a problem, formulate a hypothesis, test it empirically, and update based on evidence. This is the scientific method, applied to machine learning. It is the same process that drives every meaningful ML research result.

The students who internalize this process — not just the sklearn APIs, but the underlying scientific thinking — will be equipped to contribute meaningfully to AI as the field evolves.

---

## Section 10: Reading List — Going Deeper

### Foundations

- **"The Elements of Statistical Learning"** — Hastie, Tibshirani, Friedman. Free PDF online. The theoretical foundation for everything in sklearn.
- **"Pattern Recognition and Machine Learning"** — Bishop. Probabilistic perspective; excellent on Bayesian methods and neural networks.
- **"Deep Learning"** — Goodfellow, Bengio, Courville. Free online. Comprehensive coverage of neural networks and deep learning theory.

### Practical and Competition-Focused

- **"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"** — Geron. Best practical introduction; covers sklearn comprehensively.
- **Kaggle's "Intro to Machine Learning" and "Intermediate Machine Learning" courses** — Free, practical, directly applicable to competition settings.
- **fast.ai Practical Deep Learning course** — Top-down, practical-first approach to deep learning; teaches transfer learning intuitively.

### Papers Worth Reading Before IOAI

- **"A Few Useful Things to Know About Machine Learning"** — Domingos (2012). Dense with competition-relevant insights.
- **"Random Forests"** — Breiman (2001). Original paper; builds intuition about why RF works.
- **"XGBoost: A Scalable Tree Boosting System"** — Chen & Guestrin (2016). Explains the engineering behind gradient boosting.
- **"Attention Is All You Need"** — Vaswani et al. (2017). The transformer paper; essential for 2025+ IOAI NLP tasks.

---

## Pre-Competition Knowledge Check

```widget
{
  "type": "concept-sort",
  "title": "IOAI Readiness: Which Category Does Each Skill Belong To?",
  "categories": [
    { "name": "Must do FIRST (before optimizing)", "color": "#22C55E" },
    { "name": "Critical — but easy to forget under pressure", "color": "#EF4444" },
    { "name": "Advanced — do only if time permits", "color": "#5B5BD6" }
  ],
  "items": [
    { "text": "Submit a working baseline with default LightGBM parameters", "category": "Must do FIRST (before optimizing)" },
    { "text": "Set random seeds at the top of the notebook", "category": "Critical — but easy to forget under pressure" },
    { "text": "Verify submission columns match sample_submission.csv exactly", "category": "Critical — but easy to forget under pressure" },
    { "text": "Implement the exact competition metric from the problem statement", "category": "Must do FIRST (before optimizing)" },
    { "text": "Stacking with a meta-learner on out-of-fold predictions", "category": "Advanced — do only if time permits" },
    { "text": "Pseudo-labeling high-confidence test samples", "category": "Advanced — do only if time permits" },
    { "text": "Check for NaN values before training: assert not X.isnull().any().any()", "category": "Critical — but easy to forget under pressure" },
    { "text": "Keep a backup single-model submission in case ensemble has a bug", "category": "Critical — but easy to forget under pressure" }
  ]
}
```

---

## Final Summary

You are ready for IOAI if you can answer yes to every item on this list:

**Technical:**
- [ ] I can write the 50-line tabular pipeline from memory
- [ ] I understand what StratifiedKFold does and why it matters
- [ ] I can implement TF-IDF + Logistic Regression for text classification
- [ ] I know the difference between F1-macro, F1-weighted, and accuracy
- [ ] I can build a simple ensemble of 2-3 models and calibrate the blend weights

**Mathematical:**
- [ ] I can derive the gradient of cross-entropy loss with respect to logistic regression weights
- [ ] I understand what L1 and L2 regularization do to model weights mathematically
- [ ] I can explain the bias-variance tradeoff quantitatively
- [ ] I know what backpropagation computes and why it uses the chain rule

**Competition Strategy:**
- [ ] I always submit a baseline before optimizing
- [ ] I implement the exact evaluation metric from the problem statement locally
- [ ] I know my time allocation and can stick to it under pressure
- [ ] I have a "backup submission" strategy for the final 15 minutes

If any of these items is a "no," that item is your highest-priority study goal.

Good luck at IOAI. The preparation is the point — the medal is just the receipt.
