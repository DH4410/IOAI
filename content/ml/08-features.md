---
title: Feature Engineering
track: ml
order: 8
estimatedTime: 45
difficulty: intermediate
---

# Feature Engineering

In ML competitions, the model matters less than the features you feed it. A simple logistic regression with great features often beats a complex neural network with raw data. Feature engineering is the craft of turning raw data into the numbers that make patterns easy to learn.

---

## 1. Why Features Matter

The model can only use what you give it. If the pattern is "houses with more than 3 bedrooms sell for more", but you only give the model a single `size` column, it cannot find that pattern — even if size and bedroom count are correlated.

Good features:
- Make the relationship with the target more linear
- Reduce the number of examples needed to learn the pattern
- Encode domain knowledge the model cannot discover from raw data

---

## 2. Numerical Features

### Scaling

Most models (linear regression, SVM, neural networks) are sensitive to the scale of features. Always scale.

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# StandardScaler: mean=0, std=1 (most common)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)   # SAME scaler, no re-fit

# MinMaxScaler: range [0, 1]
mms = MinMaxScaler()
X_train_mm = mms.fit_transform(X_train)
```

**Important:** Always `fit` on training data only, then `transform` test data. Fitting on test data is data leakage.

### Log Transforms

Skewed features (like income, prices) work better after a log transform — the log compresses large values and spreads out small ones:

```python
import numpy as np

X['income_log'] = np.log1p(X['income'])   # log1p = log(1 + x), handles 0s
```

### Binning

Turn a continuous feature into categories:

```python
X['age_group'] = pd.cut(X['age'], bins=[0, 18, 35, 60, 100], labels=['child', 'young', 'adult', 'senior'])
```

---

## 3. Categorical Features

### One-Hot Encoding

Turn a category into binary columns. Use for nominal categories (no order).

```python
import pandas as pd

X_encoded = pd.get_dummies(X, columns=['color', 'city'], drop_first=True)
```

Or with scikit-learn:

```python
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat_encoded = enc.fit_transform(X[['color', 'city']])
```

### Ordinal Encoding

For ordered categories (e.g. `small < medium < large`):

```python
from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder(categories=[['small', 'medium', 'large']])
X['size_enc'] = enc.fit_transform(X[['size']])
```

### Target Encoding

Replace category with the mean target value for that category. Very powerful for high-cardinality categories.

```python
# Simple target encoding (do proper cross-validation in practice)
target_means = X_train.groupby('city')['price'].mean()
X_train['city_enc'] = X_train['city'].map(target_means)
X_test['city_enc'] = X_test['city'].map(target_means).fillna(target_means.mean())
```

---

## 4. Creating New Features

This is where you add the most value. Think about what combinations of features would be predictive.

```python
# Interaction: multiply two features
X['area_x_rooms'] = X['area'] * X['num_rooms']

# Ratio
X['price_per_sqm'] = X['price'] / X['area']

# Date features from a timestamp
X['month'] = pd.to_datetime(X['date']).dt.month
X['day_of_week'] = pd.to_datetime(X['date']).dt.dayofweek
X['is_weekend'] = X['day_of_week'].isin([5, 6]).astype(int)

# Distance from a reference point
X['dist_to_center'] = np.sqrt((X['lat'] - 51.5)**2 + (X['lon'] - -0.12)**2)
```

**Quick check:** You have a column `signup_date` and `purchase_date`. What useful feature could you create?
> `days_to_purchase = (purchase_date - signup_date).dt.days` — captures how quickly users converted.

---

## 5. Feature Selection

More features is not always better. Irrelevant features add noise and slow training. Use feature importance to prune.

```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

importance = pd.Series(model.feature_importances_, index=X_train.columns)
importance.sort_values(ascending=False).head(20).plot(kind='bar')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()
```

Drop features with very low importance (< 0.001). Refit and check if the score stays the same or improves.

---

## 6. Handling Missing Values

Real data always has missing values. Do not drop rows unless the missing rate is very high.

```python
# Check missing values
print(X.isnull().sum())

# Fill numerical with median (robust to outliers)
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='median')
X_filled = imp.fit_transform(X_num)

# Add a binary "was missing" indicator
X['age_missing'] = X['age'].isnull().astype(int)
X['age'] = X['age'].fillna(X['age'].median())
```

The "was missing" indicator sometimes has predictive power — whether a value is missing is itself informative.

---

## Summary

| Step | Purpose | Key tool |
|---|---|---|
| Scale | Normalize ranges | `StandardScaler` |
| Log transform | Fix skewed distributions | `np.log1p` |
| One-hot encode | Convert nominal categories | `pd.get_dummies` |
| Target encode | Handle high-cardinality categories | Group mean |
| Create interactions | Capture non-linear relationships | Multiply/divide columns |
| Select features | Remove noise | Random forest importance |
| Impute missing | Avoid dropped rows | `SimpleImputer` |

In competitions, spending 80% of your time on feature engineering and 20% on model selection often works better than the other way around.
