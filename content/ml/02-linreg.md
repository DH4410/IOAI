---
title: Linear Regression
track: ml
order: 2
estimatedTime: 50
difficulty: intermediate
---

# Linear Regression

Linear regression is the foundation of machine learning. It is the simplest predictive model, it has a beautiful closed-form solution, and understanding it deeply will make every more advanced algorithm easier to grasp. Do not underestimate it — linear regression (with good features) is competitive on many real-world tasks.

---

## The Linear Equation

The core idea is simple: we predict the output `y` as a **weighted sum** of the input features, plus a constant bias term.

For one feature:

```
y = w * x + b
```

Where:
- `y` is the prediction
- `x` is the input feature (e.g., house size in sqft)
- `w` is the **weight** (how much does one unit of x change y?)
- `b` is the **bias** (the intercept — what is y when x=0?)

This is exactly the equation of a straight line from middle school math: `y = mx + c`.

For multiple features (the general case):

```
y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

In matrix form, if we absorb the bias into the weight vector by adding a column of 1s to X:

```
y = X @ w
```

---

## Why Does the Line Fit? The Loss Function

The key question is: which line is the **best** line?

We need a way to measure how wrong our predictions are. This is called the **loss function** (also called the cost function).

The most common loss for regression is **Mean Squared Error (MSE)**:

```
MSE = (1/n) * Σ(yᵢ - ŷᵢ)²
```

Where:
- `n` is the number of training examples
- `yᵢ` is the true label for example `i`
- `ŷᵢ` is our model's prediction for example `i`
- `yᵢ - ŷᵢ` is the **residual** (the error)

We square the errors for two important reasons:
1. **Positive and negative errors cancel out otherwise.** Squaring makes all errors positive.
2. **Large errors are penalized more.** A prediction that is off by 10 contributes 100 to the loss — much more than ten predictions each off by 1 (which contribute 10 total). This pushes the model to avoid large mistakes.

```python
import numpy as np

def mse(y_true, y_pred):
    """Mean Squared Error."""
    errors = y_true - y_pred
    return np.mean(errors ** 2)

# Example
y_true = np.array([100, 150, 200, 250, 300])
y_pred = np.array([110, 140, 210, 245, 315])  # our model's guesses

residuals = y_true - y_pred
print("Residuals:", residuals)
print("Squared residuals:", residuals ** 2)
print("MSE:", mse(y_true, y_pred))
print("RMSE (same units as y):", np.sqrt(mse(y_true, y_pred)))
```

---

## Why We Minimize MSE

Our goal is to find the weights `w` and bias `b` that **minimize** the MSE on the training data.

Intuitively: smaller MSE = predictions closer to true values = better model.

The MSE is a smooth, convex function of the weights. This means:
- There is exactly one global minimum (no local minima to get stuck in)
- We can find it analytically using calculus, or numerically using gradient descent

This is a huge advantage of linear regression. More complex models like neural networks have non-convex loss functions with many local minima.

---

## Deriving the Closed-Form Solution

For linear regression, we can find the exact optimal weights in one step using calculus.

In matrix form, the model is:

```
ŷ = X @ w
```

The MSE is:

```
L(w) = (1/n) ||y - Xw||²
```

To minimize, take the derivative with respect to `w` and set it to zero:

```
dL/dw = -(2/n) Xᵀ(y - Xw) = 0
Xᵀy = XᵀXw
w = (XᵀX)⁻¹ Xᵀy
```

This is the **Normal Equation**. Given your data matrix X and labels y, you can compute the optimal weights in a single matrix operation.

```python
import numpy as np

def linear_regression_closed_form(X, y):
    """
    Solve linear regression using the Normal Equation.
    Adds a bias column (all 1s) automatically.
    Returns weights w where w[0] is the bias term.
    """
    n = X.shape[0]
    # Add column of 1s for the bias term
    ones = np.ones((n, 1))
    X_aug = np.hstack([ones, X])   # shape: (n, p+1)

    # Normal equation: w = (X^T X)^{-1} X^T y
    XtX = X_aug.T @ X_aug          # shape: (p+1, p+1)
    Xty = X_aug.T @ y              # shape: (p+1,)
    w = np.linalg.solve(XtX, Xty)  # more stable than np.linalg.inv(XtX) @ Xty

    return w

# Test on simple data
np.random.seed(42)
x = np.linspace(0, 10, 50)
y = 3.0 * x + 7.0 + np.random.randn(50) * 1.5

X = x.reshape(-1, 1)
w = linear_regression_closed_form(X, y)

print(f"True:    bias=7.0,  weight=3.0")
print(f"Learned: bias={w[0]:.3f}, weight={w[1]:.3f}")
```

> **Note:** We use `np.linalg.solve(A, b)` instead of `np.linalg.inv(A) @ b` because solving a linear system is numerically more stable than inverting a matrix.

---

## Implementing Linear Regression from Scratch

Let's build a complete, reusable class:

```python
import numpy as np

class LinearRegressionScratch:
    """Linear regression using the Normal Equation."""

    def __init__(self):
        self.weights = None  # includes bias as weights[0]

    def fit(self, X, y):
        """Fit model to training data."""
        n, p = X.shape
        # Prepend ones column for bias
        X_aug = np.column_stack([np.ones(n), X])
        # Normal equation
        self.weights = np.linalg.solve(X_aug.T @ X_aug, X_aug.T @ y)
        return self

    def predict(self, X):
        """Make predictions."""
        n = X.shape[0]
        X_aug = np.column_stack([np.ones(n), X])
        return X_aug @ self.weights

    @property
    def bias_(self):
        return self.weights[0]

    @property
    def coef_(self):
        return self.weights[1:]


# ── Test on house price data ──────────────────────────────────────
np.random.seed(7)
n = 100

sqft     = np.random.randint(800, 3000, n).astype(float)
bedrooms = np.random.randint(1, 6, n).astype(float)

price = 30 + 0.12 * sqft + 25 * bedrooms + np.random.randn(n) * 20

X = np.column_stack([sqft, bedrooms])

model = LinearRegressionScratch()
model.fit(X, price)

print(f"Bias (intercept): {model.bias_:.2f}")
print(f"Weights: sqft={model.coef_[0]:.4f}, bedrooms={model.coef_[1]:.4f}")
print(f"(True values: sqft=0.12, bedrooms=25.0)")

# Predict for a new house
new_house = np.array([[1500, 3]])
print(f"\nPredicted price for 1500sqft, 3bed: ${model.predict(new_house)[0]:.1f}k")
```

---

## Comparing with Sklearn

Sklearn's LinearRegression uses similar math under the hood (via Singular Value Decomposition, which is more numerically stable). The results should match closely:

```python
import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(7)
n = 100
sqft     = np.random.randint(800, 3000, n).astype(float)
bedrooms = np.random.randint(1, 6, n).astype(float)
price    = 30 + 0.12 * sqft + 25 * bedrooms + np.random.randn(n) * 20

X = np.column_stack([sqft, bedrooms])

# Our scratch implementation
our_model = LinearRegressionScratch()
our_model.fit(X, price)

# Sklearn
sk_model = LinearRegression()
sk_model.fit(X, price)

print("Comparison:")
print(f"  Bias:     scratch={our_model.bias_:.4f}   sklearn={sk_model.intercept_:.4f}")
print(f"  sqft:     scratch={our_model.coef_[0]:.4f}   sklearn={sk_model.coef_[0]:.4f}")
print(f"  bedrooms: scratch={our_model.coef_[1]:.4f}   sklearn={sk_model.coef_[1]:.4f}")
```

---

## The R² Score

MSE tells you the average squared error, but it's hard to interpret in isolation. Is an MSE of 400 good or bad? Depends on the problem!

**R² (R-squared)** is a normalized metric that ranges from 0 to 1 (and can be negative for terrible models):

```
R² = 1 - SS_res / SS_tot

SS_res = Σ(yᵢ - ŷᵢ)²         (residual sum of squares — model's error)
SS_tot = Σ(yᵢ - ȳ)²          (total sum of squares — baseline error)
```

Interpretation:
- `R² = 1.0`: Perfect predictions
- `R² = 0.0`: Model is no better than predicting the mean every time
- `R² < 0.0`: Model is worse than the mean baseline

```python
import numpy as np
from sklearn.metrics import r2_score

def r2_scratch(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot

y_true = np.array([100, 150, 200, 250, 300])
y_pred = np.array([110, 145, 195, 255, 295])

print(f"R² (scratch): {r2_scratch(y_true, y_pred):.4f}")
print(f"R² (sklearn): {r2_score(y_true, y_pred):.4f}")

# Perfect prediction
print(f"\nPerfect R²: {r2_score(y_true, y_true):.4f}")

# Predicting the mean always
mean_pred = np.full_like(y_true, y_true.mean(), dtype=float)
print(f"Mean-only R²: {r2_score(y_true, mean_pred):.4f}")
```

---

## Residuals: Diagnosing Your Model

**Residuals** are the differences between actual and predicted values: `e = y - ŷ`.

For a good linear regression model, residuals should be:
1. **Randomly scattered** around zero (no pattern)
2. **Roughly constant variance** (homoscedastic)
3. **Approximately normally distributed**

If residuals have a pattern, the linear model is missing something.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)
n = 200

X = np.random.uniform(0, 10, (n, 1))
y_linear  = 2 * X.flatten() + 5 + np.random.randn(n) * 1.0
y_nonlin  = X.flatten()**2 + np.random.randn(n) * 1.5  # truly nonlinear

for label, y in [("Linear data", y_linear), ("Nonlinear data", y_nonlin)]:
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred

    print(f"\n{label}:")
    print(f"  R² = {1 - np.sum(residuals**2)/np.sum((y-y.mean())**2):.3f}")
    print(f"  Residual mean   = {residuals.mean():.4f} (want ~0)")
    print(f"  Residual std    = {residuals.std():.3f}")
    # Check for trend: correlation between X and residuals
    corr = np.corrcoef(X.flatten(), residuals)[0, 1]
    print(f"  Residual-X corr = {corr:.3f} (want ~0)")
```

---

## Multiple Features: The Matrix Form

With multiple features, the model becomes:

```
ŷ = X @ w + b
```

Where X has shape `(n_samples, n_features)`.

The math works exactly the same — we just add a bias column of 1s to X and solve the Normal Equation.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

np.random.seed(42)
n = 300

# Multiple features
X = np.column_stack([
    np.random.uniform(500, 3000, n),    # sqft
    np.random.randint(1, 7, n),          # bedrooms
    np.random.randint(0, 50, n),         # age
    np.random.uniform(0.5, 30, n),       # distance from center
])

# True relationship
y = (
    40
    + 0.14 * X[:, 0]
    + 18   * X[:, 1]
    - 1.5  * X[:, 2]
    - 2.5  * X[:, 3]
    + np.random.randn(n) * 15
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Multiple linear regression:")
print(f"  R² on test: {r2_score(y_test, y_pred):.4f}")
print(f"\n  Feature coefficients:")
features = ["sqft", "bedrooms", "age", "distance"]
for name, coef in zip(features, model.coef_):
    print(f"    {name:>12}: {coef:+.4f}")
print(f"    {'intercept':>12}: {model.intercept_:+.4f}")
```

---

## Standardization and Normalization: Why It Matters

Features can have very different scales. `sqft` might range from 500 to 5000, while `bedrooms` only goes from 1 to 7.

This causes two problems:
1. **Gradient descent converges slowly** (the loss landscape is very elongated)
2. **Coefficients are hard to interpret** (you cannot compare a coefficient for sqft vs. bedrooms without knowing their scales)

**Standardization (Z-score normalization):** subtract mean, divide by standard deviation.

```
x_scaled = (x - μ) / σ
```

After this, each feature has mean=0 and std=1.

**Min-Max Normalization:** scale to the range [0, 1].

```
x_scaled = (x - x_min) / (x_max - x_min)
```

```python
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

np.random.seed(42)
n = 200
X = np.column_stack([
    np.random.uniform(500, 3000, n),   # sqft: range ~2500
    np.random.randint(1, 7, n),         # bedrooms: range 6
    np.random.randint(0, 50, n),        # age: range 50
])
y = 40 + 0.12 * X[:,0] + 20 * X[:,1] - 1.0 * X[:,2] + np.random.randn(n) * 20

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Without scaling
model_raw = LinearRegression()
model_raw.fit(X_train, y_train)
r2_raw = r2_score(y_test, model_raw.predict(X_test))

# With StandardScaler
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model_scaled = LinearRegression()
model_scaled.fit(X_train_s, y_train)
r2_scaled = r2_score(y_test, model_scaled.predict(X_test_s))

print(f"R² without scaling: {r2_raw:.6f}")
print(f"R² with scaling:    {r2_scaled:.6f}")
print("\n(For linear regression, R² is the same — closed form finds global optimum.")
print("For gradient descent, scaling makes a BIG difference in speed.)")

print("\nStandardScaler statistics:")
print(f"  sqft   — mean: {scaler.mean_[0]:.1f}, std: {scaler.scale_[0]:.1f}")
print(f"  bed    — mean: {scaler.mean_[1]:.2f}, std: {scaler.scale_[1]:.2f}")
print(f"  age    — mean: {scaler.mean_[2]:.1f}, std: {scaler.scale_[2]:.1f}")
```

> **Critical rule:** Always fit the scaler on the training data only. Then use `transform()` (not `fit_transform()`) on the validation and test sets. Fitting on all data lets test-set statistics leak into training.

---

## Bias-Variance Tradeoff: A First Look

Linear regression is a **high-bias, low-variance** model.

- **Bias**: Error from incorrect assumptions. If the true relationship is nonlinear but we fit a line, we have high bias. The model systematically misses the pattern.
- **Variance**: Sensitivity to small fluctuations in training data. Linear regression changes only a little when you swap out a few training points.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(42)

# True relationship is a sine curve
X_true = np.linspace(0, 6, 300)
y_true = np.sin(X_true)

# Training data (small, noisy sample)
n_train = 20
idx = np.random.choice(len(X_true), n_train, replace=False)
X_train = X_true[idx].reshape(-1, 1)
y_train = y_true[idx] + np.random.randn(n_train) * 0.15

# Underfitting: straight line
model_under = LinearRegression()
model_under.fit(X_train, y_train)

# Good fit: degree-3 polynomial
poly3 = PolynomialFeatures(degree=3, include_bias=False)
X_train_p3 = poly3.fit_transform(X_train)
model_good = LinearRegression()
model_good.fit(X_train_p3, y_train)

# Overfitting: degree-15 polynomial
poly15 = PolynomialFeatures(degree=15, include_bias=False)
X_train_p15 = poly15.fit_transform(X_train)
model_over = LinearRegression()
model_over.fit(X_train_p15, y_train)

# Evaluate on a large clean test set
X_test = np.linspace(0, 6, 300).reshape(-1, 1)
y_test = np.sin(X_test.flatten())

from sklearn.metrics import mean_squared_error

mse_under = mean_squared_error(y_test, model_under.predict(X_test))
mse_good  = mean_squared_error(y_test, model_good.predict(poly3.transform(X_test)))
mse_over  = mean_squared_error(y_test, model_over.predict(poly15.transform(X_test)))

print(f"Underfitting (degree 1)  — Test MSE: {mse_under:.4f}  (high bias)")
print(f"Good fit     (degree 3)  — Test MSE: {mse_good:.4f}  (balanced)")
print(f"Overfitting  (degree 15) — Test MSE: {mse_over:.4f}  (high variance)")
```

---

## When Linear Regression Fails

Linear regression makes strong assumptions. It will struggle when:

1. **The relationship is nonlinear** — a line can't fit a curve well
2. **There are outliers** — MSE is very sensitive to extreme values (consider using MAE or Huber loss instead)
3. **Features are highly correlated** — multicollinearity makes the coefficients unstable
4. **There are interaction effects** — e.g., price depends on the combination of sqft AND neighborhood, not each independently

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

np.random.seed(42)
n = 100
x = np.linspace(0, 10, n)

scenarios = {
    "Linear (works great)":   2 * x + 3 + np.random.randn(n),
    "Quadratic (fails)":      x**2 - 5*x + np.random.randn(n) * 3,
    "Sine wave (fails)":      10 * np.sin(x) + np.random.randn(n),
    "Exponential (fails)":    np.exp(0.4 * x) + np.random.randn(n) * 5,
}

X = x.reshape(-1, 1)
print(f"{'Scenario':<30} {'R²':>8}")
print("-" * 40)
for name, y in scenarios.items():
    model = LinearRegression()
    model.fit(X, y)
    r2 = r2_score(y, model.predict(X))
    print(f"{name:<30} {r2:>8.4f}")
```

---

## Full Example: Predicting Boston-Style House Prices

Let's put everything together in a realistic pipeline:

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load a real dataset
data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names

print(f"Dataset: {X.shape[0]} houses, {X.shape[1]} features")
print(f"Target: median house value (in $100k)")
print(f"Features: {feature_names}")
print(f"Price range: ${y.min():.2f}k – ${y.max():.2f}k")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Train
model = LinearRegression()
model.fit(X_train_s, y_train)

# Evaluate
y_pred = model.predict(X_test_s)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"\nTest Results:")
print(f"  RMSE: ${rmse*100:.0f}k")
print(f"  MAE:  ${mae*100:.0f}k")
print(f"  R²:   {r2:.4f}")

print(f"\nFeature Importance (by |coefficient|):")
pairs = sorted(zip(feature_names, model.coef_), key=lambda p: abs(p[1]), reverse=True)
for name, coef in pairs:
    bar = "+" * int(abs(coef) * 5)
    sign = "+" if coef > 0 else "-"
    print(f"  {name:>12}: {coef:+.4f}  {sign}{bar}")

# Look at residuals
residuals = y_test - y_pred
print(f"\nResiduals:")
print(f"  Mean: {residuals.mean():.4f}")
print(f"  Std:  {residuals.std():.4f}")
print(f"  Max overpredict:  ${residuals.min()*100:.0f}k")
print(f"  Max underpredict: ${residuals.max()*100:.0f}k")
```

---

## Summary

| Concept | Formula / Key Idea |
|---|---|
| Linear model | ŷ = w₁x₁ + w₂x₂ + ... + b |
| MSE loss | (1/n) Σ(y - ŷ)² |
| Normal Equation | w = (XᵀX)⁻¹Xᵀy |
| R² score | 1 - SS_res / SS_tot |
| Standardization | (x - μ) / σ |
| Residual | True value minus predicted value |
| High bias | Model too simple, underfits |
| High variance | Model too complex, overfits |

Linear regression is your baseline model. If you can't beat a line, you need to understand why before reaching for something more complex. In the next lesson, we'll learn **gradient descent** — the optimization engine that drives almost all of modern ML.
