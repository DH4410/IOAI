---
title: Building a Strong Baseline
track: ioai
order: 3
estimatedTime: 55
difficulty: advanced
---

# Building a Strong Baseline

There is a paradox at the heart of competition data science: the teams that win are rarely the ones with the most sophisticated models. They are the ones with the best baselines. A strong baseline — clean, correct, and reproducible — is the foundation on which all improvements are built. Without it, you are building on sand.

This lesson is practical. By the end, you will have a complete baseline template memorized and ready to deploy in any IOAI competition.

---

## 1. Why Baselines Win Competitions

Consider the distribution of a typical competition's leaderboard:

- **Top 5%**: Strong baselines + smart feature engineering + ensemble
- **5–20%**: Strong baselines + some feature engineering
- **20–50%**: Decent baselines with bugs or missed preprocessing steps
- **Bottom 50%**: No working baseline, overengineered solutions that did not finish

The pattern is clear: getting the basics right separates the top half from the bottom half. Getting the basics *perfect* separates the top 5% from the top 20%.

Here is what a "strong baseline" provides:

1. **A guaranteed score**: No matter what happens in the rest of the competition, you have a valid submission.
2. **A measurement anchor**: You cannot know if a change is an improvement without something to compare to.
3. **A foundation**: Every feature engineering trick and ensemble you add builds on the baseline. A buggy baseline propagates errors to everything downstream.

The most common mistake among intermediate-level students is to **skip the baseline** and jump straight to complex models. This feels productive — you are building something sophisticated. But when you run out of time with a half-finished neural network, the student with a well-tuned LightGBM baseline beats you.

---

## 2. The Baseline Template

Here is the complete baseline template. Every IOAI practical exam should begin with some version of this.

```python
# ============================================================
# IOAI COMPETITION BASELINE TEMPLATE
# ============================================================

# 0. SETUP — Always first
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import warnings
import os
import random

warnings.filterwarnings('ignore')

SEED = 42

def set_seeds(seed=SEED):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seeds()

# 1. DATA LOADING
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')
sub   = pd.read_csv('sample_submission.csv')

TARGET = 'target'            # <- Change this
ID_COL = 'id'               # <- Change this

# 2. QUICK EDA (covered in detail in next section)
print(f'Train: {train.shape}, Test: {test.shape}')
print(f'Target distribution:\n{train[TARGET].value_counts(normalize=True)}')
print(f'Missing values (train):\n{train.isnull().sum()[train.isnull().sum()>0]}')

# 3. FEATURE PREPARATION
num_cols  = train.select_dtypes(include='number').columns.tolist()
cat_cols  = train.select_dtypes(include='object').columns.tolist()
if TARGET in num_cols: num_cols.remove(TARGET)
if ID_COL  in num_cols: num_cols.remove(ID_COL)
if ID_COL  in cat_cols: cat_cols.remove(ID_COL)

feature_cols = num_cols + cat_cols

X      = train[feature_cols].copy()
y      = train[TARGET].copy()
X_test = test[feature_cols].copy()

# 4. PREPROCESSING PIPELINE
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# For the baseline: handle categoricals with ordinal encoding
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
], remainder='drop')

# 5. MODEL DEFINITION
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=SEED,
        verbose=-1,
        n_jobs=-1
    ))
])

# 6. CROSS-VALIDATION
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
print(f'CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

# 7. TRAIN ON FULL DATA AND PREDICT
model.fit(X, y)
test_preds = model.predict_proba(X_test)[:, 1]

# 8. SUBMISSION
sub[TARGET] = test_preds
sub.to_csv('submission_v1_baseline.csv', index=False)
print(f'Saved submission with {len(sub)} rows')
print(sub.head())
```

This template is approximately 60 lines of code. You should be able to write it in under 15 minutes with practice. If you cannot, practice until you can.

---

## 3. EDA Deep Dive

Exploratory Data Analysis (EDA) in a competition is not academic — it is diagnostic. You are not trying to understand every nuance of the data. You are answering specific questions that affect your modeling decisions.

### The EDA Checklist

Run through this checklist every time, in order:

#### 3.1 Shape and Data Types

```python
print(f'Train shape: {train.shape}')
print(f'Test shape: {test.shape}')
print(f'\nDtypes:\n{train.dtypes.value_counts()}')
print(f'\nFirst 5 rows:\n{train.head()}')
```

What to look for:
- **Shape**: How many rows? How many columns? Large datasets (>500k rows) require fast models and efficient preprocessing.
- **Dtypes**: Are there `object` columns that should be numeric? Are there `int64` columns that should be categorical?
- **Head**: Does the data look as expected? Are IDs sequential? Any obvious anomalies?

#### 3.2 Missing Values

```python
missing = train.isnull().sum()
missing_pct = missing / len(train)
missing_df = pd.DataFrame({
    'count': missing,
    'pct': missing_pct
}).sort_values('pct', ascending=False)
print(missing_df[missing_df['count'] > 0])
```

What to look for:
- **Missing completely at random (MCAR)**: Values are missing for no reason related to other features. Safe to impute with mean/median.
- **Missing at random (MAR)**: Missingness is related to other observed features. Better to use model-based imputation.
- **Missing not at random (MNAR)**: Missingness is related to the missing value itself (e.g., income not reported because it is very high or very low). The missingness is informative — create a binary "was_missing" indicator.

When you see high missingness (>30%) in a column:
- Create `feature_X_was_missing` = 1 where X was NaN, else 0
- Impute the original column separately

#### 3.3 Target Distribution

For **classification**:
```python
print(f'Target counts:\n{train[TARGET].value_counts()}')
print(f'Target proportions:\n{train[TARGET].value_counts(normalize=True)}')
```

For **regression**:
```python
print(f'Target stats:\n{train[TARGET].describe()}')
print(f'Skewness: {train[TARGET].skew():.4f}')
print(f'Kurtosis: {train[TARGET].kurtosis():.4f}')
```

What to look for:

- **Class imbalance**: If one class is >80% of training data, accuracy is a misleading metric. Use class_weight='balanced' or adjust decision threshold.
- **Target skewness** (regression): If |skewness| > 1, consider log-transforming the target. This often significantly improves RMSE. Remember to inverse-transform predictions.
- **Target range** (regression): Are there outliers in the target? Clip or transform them.
- **Multi-class balance**: For macro F1, all classes are weighted equally regardless of size — pay attention to small classes.

#### 3.4 Numerical Feature Analysis

```python
# Distribution statistics
print(train[num_cols].describe())

# Correlation with target
correlations = train[num_cols + [TARGET]].corr()[TARGET].sort_values(ascending=False)
print(f'Feature-target correlations:\n{correlations}')

# Identify outliers
for col in num_cols[:5]:  # Check first 5
    q1, q99 = train[col].quantile([0.01, 0.99])
    outliers = ((train[col] < q1) | (train[col] > q99)).sum()
    print(f'{col}: {outliers} outliers beyond 1st/99th percentile')
```

What to look for:
- **Large ranges**: Features on different scales (one in [0,1], another in [0, 1,000,000]) — standardize.
- **High correlation with target**: These are your most valuable features — make sure you process them carefully.
- **Outliers**: Extreme values can destabilize some models. Consider clipping or log-transforming.

#### 3.5 Categorical Feature Analysis

```python
for col in cat_cols:
    n_unique = train[col].nunique()
    most_common = train[col].value_counts().iloc[:3].to_dict()
    print(f'{col}: {n_unique} unique values | Top 3: {most_common}')
```

What to look for:
- **Low cardinality** (< 20 unique values): Use one-hot encoding or ordinal encoding.
- **High cardinality** (> 100 unique values): One-hot encoding creates too many columns. Use target encoding, frequency encoding, or embedding.
- **Rare categories**: Categories appearing <0.1% of the time may need grouping into "other."
- **Unseen categories in test**: Test set may have categories not in training. Your encoder must handle `unknown` gracefully.

#### 3.6 Train/Test Distribution Comparison

```python
# Check that test features have similar distributions to train features
for col in num_cols[:10]:
    train_mean = train[col].mean()
    test_mean  = test[col].mean()
    shift = abs(train_mean - test_mean)
    if shift > train[col].std():
        print(f'WARNING: {col} has significant train/test distribution shift')
```

Distribution shift is a major problem. If train and test have different distributions, your model learns the wrong patterns. This can happen when:
- The dataset spans different time periods (train = 2020, test = 2023)
- Different data collection processes were used
- The test set is from a different geographic region

---

## 4. Standard Preprocessing Pipeline

Once EDA is done, you need a preprocessing pipeline that correctly handles all feature types. The scikit-learn `Pipeline` + `ColumnTransformer` combination is the standard approach.

### Why Use Pipelines?

Without a pipeline, you manually apply transformations:
```python
# BAD: Manual preprocessing (risk of data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.fit_transform(X_val)  # BUG: fitting scaler on validation set
```

With a pipeline:
```python
# GOOD: Pipeline prevents leakage
pipeline = Pipeline([('scaler', StandardScaler()), ('model', LGBMClassifier())])
# cross_val_score fits scaler on train fold, transforms val fold correctly
cv_scores = cross_val_score(pipeline, X, y, cv=5)
```

The pipeline guarantees that preprocessing steps are fit only on training data within each cross-validation fold.

### Complete Preprocessing Recipe

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, OrdinalEncoder, TargetEncoder
)
from sklearn.impute import SimpleImputer

def build_preprocessor(num_cols, low_card_cats, high_card_cats, y_train=None):
    """
    num_cols: numerical column names
    low_card_cats: categorical columns with <20 unique values
    high_card_cats: categorical columns with >=20 unique values
    """
    
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    low_card_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        ))
    ])
    
    high_card_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', TargetEncoder(target_type='binary'))  # sklearn 1.3+
    ])
    
    transformers = []
    if num_cols:       transformers.append(('num', num_pipe, num_cols))
    if low_card_cats:  transformers.append(('low', low_card_pipe, low_card_cats))
    if high_card_cats: transformers.append(('high', high_card_pipe, high_card_cats))
    
    return ColumnTransformer(transformers, remainder='drop')
```

### Handling Categorical Cardinality

**Low cardinality (< 20 unique values)**: Ordinal encoding is safe and fast. For very low cardinality (< 10), one-hot encoding avoids ordinal assumptions.

**Medium cardinality (20–100 unique values)**: Target encoding encodes each category as the mean of the target within that category. This is powerful but requires careful cross-fitting to avoid leakage.

**High cardinality (100+ unique values)**: Frequency encoding (replace category with its frequency in training set) or target encoding. Consider hashing tricks for extremely high cardinality.

### LightGBM Shortcut

LightGBM can handle categorical features natively without encoding:
```python
from lightgbm import LGBMClassifier

# Just encode categoricals as integers
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].fillna('missing'))
    X_test[col] = le.transform(X_test[col].fillna('missing'))

model = LGBMClassifier(
    categorical_feature=cat_cols,  # Tell LightGBM which cols are categorical
    random_state=SEED
)
```

This is faster to implement and often works just as well. Use it in the baseline, then refine later.

---

## 5. Choosing the Right First Model

The "right" first model is the one that gives you a strong baseline fastest.

### Decision Tree

```
Is the data tabular (rows and columns)?
  YES -> Use LightGBM as baseline
    Does it have images? -> Add CNN
    Does it have text? -> Add TF-IDF or BERT
  NO -> What modality?
    Images -> Pretrained CNN (ResNet, EfficientNet)
    Text -> TF-IDF + LightGBM or pretrained BERT
    Sequences -> LSTM or Transformer
```

### Why LightGBM for Tabular Data?

LightGBM (Light Gradient Boosting Machine) wins on tabular data because:
1. **Handles mixed types**: Numerical, categorical, and even boolean features natively.
2. **Fast**: Histogram-based algorithm trains 10–100x faster than XGBoost on large datasets.
3. **Regularization built in**: `num_leaves`, `min_child_samples`, `subsample` provide strong regularization.
4. **High accuracy by default**: Even with default hyperparameters, LightGBM is competitive.
5. **Feature importance**: Built-in feature importance for understanding which features matter.

### Key LightGBM Hyperparameters

For a competition baseline, set these:

```python
model = LGBMClassifier(
    n_estimators=1000,           # Large number; use early stopping
    learning_rate=0.05,          # Lower = slower but better generalization
    num_leaves=31,               # Main complexity parameter; default is fine
    min_child_samples=20,        # Regularization: minimum samples per leaf
    subsample=0.8,               # Row sampling; adds randomness
    colsample_bytree=0.8,        # Feature sampling; adds randomness
    random_state=SEED,
    verbose=-1,
    n_jobs=-1
)
```

For a baseline, you do not need to tune these. Default or near-default values give 90% of the performance with 0% of the tuning effort.

---

## 6. Cross-Validation Setup

Cross-validation is how you measure whether a change is actually an improvement. Getting this right is critical.

### Stratified K-Fold (for classification)

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
```

`StratifiedKFold` ensures each fold has the same class distribution as the full dataset. This is critical for imbalanced datasets — without it, some folds might have no minority class samples.

### Time-Series Cross-Validation

If the data has a temporal component (sales by day, events by timestamp):

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
# This creates splits where train is always before validation in time
```

**Never use random shuffled CV on time-series data.** It creates artificially high scores because the model sees "future" information.

### The OOF (Out-of-Fold) Pattern

For competitions, it is better to compute OOF predictions manually (rather than using `cross_val_score`) because you get:
1. OOF predictions for the training set (needed for stacking)
2. Test predictions averaged across folds (often more stable than full-data training)

```python
def train_with_oof(model, X, y, X_test, cv, metric_fn=roc_auc_score):
    """
    Train model with OOF prediction and test averaging.
    Returns: oof_predictions, test_predictions, fold_scores
    """
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr = X.iloc[tr_idx] if hasattr(X, 'iloc') else X[tr_idx]
        X_val = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
        y_tr = y.iloc[tr_idx] if hasattr(y, 'iloc') else y[tr_idx]
        y_val = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]
        
        model.fit(X_tr, y_tr)
        
        # OOF predictions
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        
        # Test predictions
        test_preds += model.predict_proba(X_test)[:, 1] / cv.n_splits
        
        fold_score = metric_fn(y_val, oof_preds[val_idx])
        fold_scores.append(fold_score)
        print(f'  Fold {fold+1}: {fold_score:.4f}')
    
    overall_score = metric_fn(y, oof_preds)
    print(f'  OOF Score: {overall_score:.4f} | Mean Fold: {np.mean(fold_scores):.4f}')
    
    return oof_preds, test_preds, fold_scores
```

Note: Use OOF score (computed on all training data) rather than mean fold score — it is a more stable estimate.

---

## 7. Iterating From the Baseline

Once you have a working baseline, improvement is iterative. Here is a structured approach.

### The Improvement Loop

```
1. Identify the biggest source of error
2. Hypothesize why it occurs
3. Build a fix (feature, preprocessing change, model change)
4. Measure: does CV score improve?
5. If yes: keep the change. If no: revert.
6. Repeat.
```

This sounds simple. The discipline is in step 5 — **reverting changes that do not help**. Many students keep changes that hurt "because I worked hard on them." This is survivorship bias in your own experimentation. The change did not help. Remove it.

### What to Try (In Order)

**Round 1: Better preprocessing**
- Create `was_missing` indicators for high-missingness features
- Use target encoding instead of ordinal encoding for high-cardinality features
- Log-transform heavily skewed numerical features

**Round 2: Feature engineering**
- Interaction features: `A * B`, `A / B`, `A - B`
- Aggregate features: for each category in cat_col, compute mean/std/count of num_col
- Polynomial features (for small feature sets)
- Domain-specific features (if the problem description gives domain hints)

**Round 3: Model improvements**
- Increase `n_estimators` and add early stopping
- Add XGBoost or CatBoost for diversity
- Try more `num_leaves` or `max_depth`

**Round 4: Ensemble**
- Average LightGBM + XGBoost predictions
- Rank averaging for robustness

### Measuring Progress Correctly

Always compare against the **OOF score** on the same data split (same seed). Do not compare CV scores computed with different seeds — the variance can mask real signal.

```python
# WRONG: Different seeds give different scores even with the same model
cv1 = cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=1))
cv2 = cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=2))
# cv1.mean() and cv2.mean() may differ by 0.005-0.010 due to random fold assignment

# RIGHT: Fix the seed and CV split, change only what you are testing
fixed_cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
```

---

## 8. The 80/20 Rule in Competitions

The Pareto principle applies directly to competition ML:

- **The first 20% of effort** (baseline + basic preprocessing) produces **80% of achievable score**
- **The remaining 80% of effort** (advanced feature engineering, hypertuning, ensembles) produces **the remaining 20% of score**

This means:
- In a 5-hour exam, you need roughly 1 hour to get 80% of the way there
- The remaining 4 hours fight over the remaining 20%

**Corollary**: If your baseline is missing or broken, you are competing with one arm tied behind your back. The time you save by not building a baseline is dwarfed by the points you lose from not having one.

---

## 9. Feature Engineering Templates

These are patterns you should know so well that implementing them takes minutes, not hours.

### Aggregate Features

```python
def add_group_features(train, test, group_col, agg_col, operations=['mean', 'std', 'min', 'max', 'count']):
    """Add aggregated features: mean/std/etc. of agg_col grouped by group_col"""
    full = pd.concat([train, test], axis=0)
    
    for op in operations:
        col_name = f'{agg_col}_{op}_by_{group_col}'
        if op == 'count':
            agg = full.groupby(group_col)[agg_col].transform('count')
        else:
            agg = full.groupby(group_col)[agg_col].transform(op)
        train[col_name] = agg.iloc[:len(train)].values
        test[col_name]  = agg.iloc[len(train):].values
    
    return train, test
```

### Frequency Encoding

```python
def frequency_encode(train, test, col):
    """Replace category with its frequency in training data"""
    freq_map = train[col].value_counts(normalize=True).to_dict()
    train[f'{col}_freq'] = train[col].map(freq_map).fillna(0)
    test[f'{col}_freq']  = test[col].map(freq_map).fillna(0)
    return train, test
```

### Date Feature Extraction

```python
def extract_date_features(df, date_col):
    """Extract useful features from datetime column"""
    df[date_col] = pd.to_datetime(df[date_col])
    df[f'{date_col}_year']      = df[date_col].dt.year
    df[f'{date_col}_month']     = df[date_col].dt.month
    df[f'{date_col}_day']       = df[date_col].dt.day
    df[f'{date_col}_dayofweek'] = df[date_col].dt.dayofweek
    df[f'{date_col}_hour']      = df[date_col].dt.hour
    df[f'{date_col}_is_weekend']= (df[date_col].dt.dayofweek >= 5).astype(int)
    return df
```

### TF-IDF for Text

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

def add_tfidf_features(train, test, text_col, max_features=1000):
    """Add TF-IDF features from a text column"""
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    train_tfidf = tfidf.fit_transform(train[text_col].fillna(''))
    test_tfidf  = tfidf.transform(test[text_col].fillna(''))
    return train_tfidf, test_tfidf
```

---

## 10. The Submit Function

Always have a standard submit function:

```python
def submit(predictions, filename, sub_template, id_col='id', target_col='target'):
    """Create and save a submission file"""
    assert len(predictions) == len(sub_template), \
        f'Length mismatch: {len(predictions)} predictions, {len(sub_template)} expected'
    
    sub = sub_template.copy()
    sub[target_col] = predictions
    sub.to_csv(filename, index=False)
    
    # Sanity checks
    assert sub[target_col].isnull().sum() == 0, 'NaN in predictions!'
    assert sub[id_col].duplicated().sum() == 0, 'Duplicate IDs!'
    
    print(f'Saved: {filename}')
    print(f'Shape: {sub.shape}')
    print(f'Target stats: mean={sub[target_col].mean():.4f}, '
          f'std={sub[target_col].std():.4f}, '
          f'min={sub[target_col].min():.4f}, '
          f'max={sub[target_col].max():.4f}')
    return sub
```

---

## 11. Summary: The Baseline Commandments

1. **Set seeds first.** Every time. Before anything else.
2. **Load, inspect, understand** before coding.
3. **Use a Pipeline** to prevent data leakage.
4. **Use StratifiedKFold** for classification. Use TimeSeriesSplit for temporal data.
5. **Get a valid submission** before improving anything.
6. **Record every experiment** with its CV score and description.
7. **Revert changes that do not improve CV score.**
8. **The 80/20 rule**: A working baseline beats a failed complexity.
9. **Check your submission format** before every submit: same columns, same rows, no NaN.
10. **Always run your final code from scratch** (restart kernel and run all) before the final submission.

---

*Next: Lesson 4 — Competition Metrics Explained*
