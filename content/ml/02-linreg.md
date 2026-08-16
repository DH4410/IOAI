---
title: Linear Regression
track: ml
order: 2
estimatedTime: 35
difficulty: intermediate
---

# Linear Regression

Linear regression is the simplest ML model that actually predicts numbers. It finds the best straight line through your data. Even if the final model you use in a competition is a gradient boosted tree or a neural network, understanding linear regression helps you understand everything else.

---

## 1. The Idea

You want to predict house price from size. You assume the relationship is roughly a line:

```
price = weight * size + bias
```

`weight` and `bias` are numbers the model learns. `weight` says "how much does price go up per extra square meter" and `bias` is the baseline price when size is 0.

For multiple features:

```
price = w1 * size + w2 * bedrooms + w3 * age + bias
```

In math: `y = Xw + b` where X is your feature matrix.

---

## 2. How the Model Learns: The Loss Function

The model learns by measuring how wrong it is. The most common measure is **Mean Squared Error (MSE)**:

```
MSE = (1/n) * sum((predicted - actual)^2)
```

Squaring makes large errors count more than small ones. The model's job is to find weights that minimize MSE.

**Quick check:** Your model predicts 100 but the actual value is 80. And another prediction of 95 vs actual 85. MSE for these two points?
> `((100-80)^2 + (95-85)^2) / 2 = (400 + 100) / 2 = 250`

---

## 3. From Scratch

```python
import numpy as np

class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        # Closed-form solution: w = (X^T X)^(-1) X^T y
        X_b = np.c_[np.ones(len(X)), X]   # add a column of 1s for bias
        theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        self.bias    = theta[0]
        self.weights = theta[1:]

    def predict(self, X):
        return X @ self.weights + self.bias

# Test it
np.random.seed(42)
X = np.random.randn(100, 1)
y = 3 * X.squeeze() + 2 + np.random.randn(100) * 0.5

model = LinearRegression()
model.fit(X, y)
print(f'Weight: {model.weights[0]:.2f}')   # should be close to 3
print(f'Bias:   {model.bias:.2f}')          # should be close to 2
```

---

## 4. Using Scikit-Learn

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)

print(f'MSE: {mean_squared_error(y_test, preds):.4f}')
print(f'R2:  {r2_score(y_test, preds):.4f}')

print(f'Coefficients: {model.coef_}')
print(f'Intercept:    {model.intercept_:.4f}')
```

**R2 score** (R-squared): measures how much variance the model explains.
- R2 = 1.0: perfect predictions
- R2 = 0.0: model is no better than just predicting the mean
- R2 < 0: model is worse than predicting the mean

**Quick check:** Your model gets R2 = 0.85. What does this mean?
> The model explains 85% of the variation in the target. 15% is unexplained (due to noise or missing features).

---

## 5. Multiple Features and Scaling

When you have multiple features, scaling helps:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)  # use same scaler, do NOT fit again

model = LinearRegression()
model.fit(X_train_scaled, y_train)
```

**Why scale?** If feature A ranges from 0 to 1 and feature B ranges from 0 to 1,000,000, the model will focus on B just because the numbers are bigger, even if A matters more. Scaling puts them on the same level.

Always `fit_transform` on training data, `transform` only on test data. Fitting on test data leaks information.

---

## 6. When Linear Regression Fails

Linear regression assumes the relationship is linear. If it is not, the model will be wrong no matter how much data you give it.

Signs it is not working:
- R2 is low even with many features
- Residuals (predicted - actual) form a curve when plotted vs predictions
- The relationship in a scatter plot is clearly curved

When this happens: try adding polynomial features, or switch to a tree-based model.

---

## Summary

| Concept | Key point |
|---|---|
| Model | `y = Xw + b`, learns weights from data |
| Loss | MSE: mean of squared errors |
| R2 score | 1 = perfect, 0 = no better than mean |
| Feature scaling | Use `StandardScaler`, fit on train only |
| When it fails | When the true relationship is not linear |

Linear regression is also the building block of logistic regression (classification) and ridge/lasso regression (regularized). The ideas carry over directly.
