---
title: Linear Regression
track: ml
order: 2
estimatedTime: 40
difficulty: intermediate
---

# Linear Regression

Linear regression is the simplest model that predicts numbers. Even if you end up using XGBoost or a neural network at IOAI, understanding linear regression is crucial — it teaches you *how* models learn, what a loss function is, and why scaling matters. Everything builds on these ideas.

---

## 1. The Model

You want to predict house price from size. You assume a linear relationship:

$$\hat{y} = w \cdot x + b$$

- $w$ (weight) = how much price changes per extra square meter
- $b$ (bias/intercept) = the baseline price

For **multiple features**, each one gets its own weight:

$$\hat{y} = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b = \mathbf{w}^T \mathbf{x} + b$$

In matrix form for a whole dataset at once: $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$

The model's job: find $\mathbf{w}$ and $b$ that make predictions as close to reality as possible.

---

## 2. The Loss Function: MSE

To judge "how wrong" the model is, we use **Mean Squared Error (MSE)**:

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2$$

Why square the errors?
1. Cancels out positive and negative errors (otherwise they'd cancel)
2. Penalizes large errors much more than small ones (a prediction that's off by 10 is 100× worse than off by 1)

**Example:** Two predictions:
- Predict 100, actual 80 → error = 20, squared = 400
- Predict 95, actual 85  → error = 10, squared = 100
- MSE = (400 + 100) / 2 = **250**

Root MSE (RMSE) = √250 ≈ 15.8 — in the same units as your target (dollars, kg, etc.)

---

## 3. Finding the Minimum: Two Approaches

**Approach A: Closed-Form Solution**

For linear regression specifically, there's a formula that gives the exact answer in one shot:

$$\mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

This is called the **Normal Equation**. Works perfectly for small datasets. But inverting a matrix is $O(n^3)$ — too slow for millions of features.

**Approach B: Gradient Descent**

For large datasets, we iterate: compute the gradient of MSE with respect to $\mathbf{w}$, then step in the opposite direction. This is covered in the next lesson — it's the core of all neural network training too.

---

## 4. Build It from Scratch

```python
import numpy as np

class LinearRegressionScratch:
    def __init__(self):
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        # Normal equation: w = (X^T X)^-1 X^T y
        # We add a bias column so we can solve for w and b together
        n = len(X)
        X_b = np.column_stack([np.ones(n), X])  # shape (n, 1+features)
        theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        self.bias    = theta[0]
        self.weights = theta[1:]

    def predict(self, X):
        return X @ self.weights + self.bias

    def mse(self, X, y):
        preds = self.predict(X)
        return np.mean((preds - y) ** 2)

    def r2(self, X, y):
        preds = self.predict(X)
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot

# Generate noisy linear data: y = 3x + 2 + noise
np.random.seed(42)
X = np.random.randn(200, 1)
y = 3 * X.squeeze() + 2 + np.random.randn(200) * 0.8

model = LinearRegressionScratch()
model.fit(X, y)
print(f"Weight (true ≈ 3.0): {model.weights[0]:.3f}")
print(f"Bias   (true ≈ 2.0): {model.bias:.3f}")
print(f"MSE: {model.mse(X, y):.4f}")
print(f"R²:  {model.r2(X, y):.4f}")
```

---

## 5. Using scikit-learn (What You Actually Use)

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Simulate a multi-feature dataset (e.g., house pricing)
np.random.seed(42)
n = 500
X = np.random.randn(n, 3)        # 3 features: size, rooms, age
y = 5*X[:,0] - 2*X[:,1] + 3*X[:,2] + 10 + np.random.randn(n)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features — critical for gradient-based methods, good habit always
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit + transform on train
X_test  = scaler.transform(X_test)        # ONLY transform on test (same mean/std!)

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
r2   = r2_score(y_test, preds)

print(f"RMSE: {rmse:.3f}")
print(f"R²:   {r2:.4f}")
print(f"Learned weights: {model.coef_.round(3)}")  # should be ≈ [5, -2, 3]
print(f"Intercept:       {model.intercept_:.3f}")  # should be ≈ 10
```

---

## 6. Understanding R²

$$R^2 = 1 - \frac{\sum(\hat{y}_i - y_i)^2}{\sum(\bar{y} - y_i)^2}$$

The denominator is the variance if you just always predict the mean. R² measures how much better your model is.

| R² value | Meaning |
|---|---|
| 1.0 | Perfect — all predictions match |
| 0.9 | Excellent — explains 90% of variance |
| 0.5 | Mediocre — better than random, but misses a lot |
| 0.0 | No better than predicting the mean |
| < 0 | Worse than predicting the mean (something is wrong!) |

---

## 7. Regularization: Ridge and Lasso

**Problem:** With many features, linear regression can overfit — it finds weights that fit noise in the training data.

**Solution:** Add a penalty to the loss function for large weights.

**Ridge (L2):** $\text{Loss} = \text{MSE} + \lambda \sum w_i^2$
- Pushes all weights toward zero, but rarely to exactly zero
- Good when all features might matter

**Lasso (L1):** $\text{Loss} = \text{MSE} + \lambda \sum |w_i|$
- Can set some weights to exactly zero — automatic feature selection
- Good when you suspect many features are irrelevant

```python
from sklearn.linear_model import Ridge, Lasso

ridge = Ridge(alpha=1.0)   # alpha = λ, the regularization strength
lasso = Lasso(alpha=0.1)

ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)

print("Ridge weights:", ridge.coef_.round(3))
print("Lasso weights:", lasso.coef_.round(3))  # some may be exactly 0.0
```

**Cross-validate lambda:** Use `RidgeCV` or `LassoCV` to find the best regularization strength automatically.

---

## 8. Gradient Slider: Watch the Loss Minimize

```widget
{
  "type": "gradient-slider",
  "title": "Loss surface for 1D linear regression",
  "subtitle": "Drag the weight slider — see how MSE changes. The minimum is the optimal weight."
}
```

Notice: the loss curve is **convex** (bowl-shaped). No local minima — gradient descent is guaranteed to find the global minimum for linear regression. This is a special property that neural networks don't have.

---

## 9. When Linear Regression Fails

| Symptom | Likely cause | Fix |
|---|---|---|
| R² very low on train data | Relationship is non-linear | Add polynomial features or switch to tree model |
| R² high on train, low on test | Overfitting (too many features) | Use Ridge/Lasso regularization |
| Residuals form a curve | Non-linearity in the data | Polynomial features or different model |
| One feature dominates | Features on very different scales | Apply `StandardScaler` |

---

## Match the Concept

```widget
{
  "type": "concept-sort",
  "title": "Linear Regression: What Does Each Term Do?",
  "categories": [
    { "name": "Ridge (L2)", "color": "#5B5BD6" },
    { "name": "Lasso (L1)", "color": "#F97316" },
    { "name": "No regularization", "color": "#22C55E" }
  ],
  "items": [
    { "text": "Can set some feature weights to exactly zero", "category": "Lasso (L1)" },
    { "text": "Shrinks all weights toward zero but keeps all features", "category": "Ridge (L2)" },
    { "text": "Best when you have many features and suspect most are irrelevant", "category": "Lasso (L1)" },
    { "text": "Adds λ·Σwᵢ² to the loss function", "category": "Ridge (L2)" },
    { "text": "Exact solution via normal equation: w = (XᵀX)⁻¹Xᵀy", "category": "No regularization" },
    { "text": "Good default when features are all potentially useful", "category": "Ridge (L2)" },
    { "text": "Performs automatic feature selection", "category": "Lasso (L1)" },
    { "text": "Minimizes only MSE with no extra penalty term", "category": "No regularization" }
  ]
}
```

---

## Summary

| Concept | Key point |
|---|---|
| Model | $\hat{y} = \mathbf{w}^T\mathbf{x} + b$ — a weighted sum of features |
| Loss | MSE = mean of squared errors |
| Normal equation | Exact solution in one shot; slow for huge datasets |
| R² | Fraction of variance explained; 1 = perfect |
| Ridge | L2 penalty — shrinks weights toward zero |
| Lasso | L1 penalty — can zero out weights (feature selection) |
| Scaling | Always `fit_transform` on train, `transform` on test |

Linear regression is the foundation. Logistic regression adds a sigmoid on top. Ridge/Lasso add regularization. Neural networks replace the linear equation with stacked non-linear layers. The core idea — minimize a loss function over data — stays constant.
