---
title: Logistic Regression
track: ml
order: 4
estimatedTime: 45
difficulty: intermediate
---

# Logistic Regression

Despite its name, logistic regression is a **classification** algorithm, not a regression one. It is one of the most widely used models in machine learning, statistics, and medicine. It is interpretable, fast, and often competitive with more complex models. This lesson covers everything from the core idea to multi-class classification and the ROC curve.

---

## From Regression to Classification

In linear regression, we output a number: `ŷ = wᵀx + b`. For classification, we want to output a **probability** between 0 and 1 — specifically, the probability that the example belongs to class 1.

The problem: `wᵀx + b` can produce any value from −∞ to +∞. We need to squash it into [0, 1].

The solution: the **sigmoid function**.

---

## The Sigmoid Function

```
σ(z) = 1 / (1 + e^{-z})
```

The sigmoid function:
- Maps any real number to the range (0, 1)
- Outputs 0.5 when z = 0
- Approaches 1 as z → +∞
- Approaches 0 as z → −∞
- Has a nice derivative: `σ'(z) = σ(z) * (1 - σ(z))`

```python
import numpy as np

def sigmoid(z):
    """Sigmoid function: maps any value to (0, 1)."""
    return 1 / (1 + np.exp(-z))

# Properties
z_values = np.array([-10, -5, -2, -1, 0, 1, 2, 5, 10])
print(f"{'z':>6} | {'σ(z)':>10} | {'interpretation':>25}")
print("-" * 48)
for z in z_values:
    s = sigmoid(z)
    interp = "strong class 0" if s < 0.2 else "strong class 1" if s > 0.8 else "uncertain"
    print(f"{z:>6} | {s:>10.4f} | {interp:>25}")

print(f"\nAt z=0: σ(0) = {sigmoid(0.0):.4f} — completely uncertain")
print(f"Derivative at z=2: σ(2)*(1-σ(2)) = {sigmoid(2)*(1-sigmoid(2)):.4f}")

# Numerical stability: avoid overflow for very negative z
def sigmoid_stable(z):
    """Numerically stable sigmoid."""
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))

test = np.array([-1000, -10, 0, 10, 1000])
print(f"\nStable sigmoid at extremes: {sigmoid_stable(test)}")
```

---

## The Logistic Regression Model

The full logistic regression model:

```
z = wᵀx + b          (linear combination)
p̂ = σ(z)             (probability of class 1)
ŷ = 1 if p̂ ≥ 0.5, else 0   (predicted class)
```

The **decision boundary** is where p̂ = 0.5, which corresponds to z = 0, which corresponds to `wᵀx + b = 0`. This is a hyperplane in feature space.

```python
import numpy as np

def logistic_predict_proba(X, w, b):
    """Compute class 1 probability for each example."""
    z = X @ w + b
    return sigmoid(z)

def logistic_predict(X, w, b, threshold=0.5):
    """Predict class label (0 or 1)."""
    probas = logistic_predict_proba(X, w, b)
    return (probas >= threshold).astype(int)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Toy example: predict if tumor is malignant (1) or benign (0)
# Features: [size_cm, irregularity_score]
X_toy = np.array([
    [1.2, 0.3],  # small, regular → likely benign
    [3.5, 0.8],  # medium, somewhat irregular
    [5.0, 0.9],  # large, irregular → likely malignant
    [0.8, 0.1],  # very small, very regular → benign
    [4.2, 0.7],  # large, irregular → malignant
])

# Hypothetical learned weights
w = np.array([0.5, 2.0])
b = -3.0

probas = logistic_predict_proba(X_toy, w, b)
preds  = logistic_predict(X_toy, w, b)

print(f"{'Example':>8} {'P(malignant)':>14} {'Prediction':>12}")
print("-" * 36)
for i, (p, pred) in enumerate(zip(probas, preds)):
    label = "Malignant" if pred == 1 else "Benign"
    print(f"{i+1:>8} {p:>14.4f} {label:>12}")
```

---

## Cross-Entropy Loss

We cannot use MSE for classification. Why? Because the sigmoid makes MSE a non-convex function with many local minima. Instead, we use **binary cross-entropy** (also called log loss).

For one example:

```
L(y, p̂) = -[y log(p̂) + (1-y) log(1-p̂)]
```

Intuition:
- If y = 1 (true class is 1): loss = -log(p̂). If p̂ → 1, loss → 0 (correct). If p̂ → 0, loss → ∞ (very wrong).
- If y = 0 (true class is 0): loss = -log(1-p̂). If p̂ → 0, loss → 0 (correct). If p̂ → 1, loss → ∞ (very wrong).

For the entire dataset:

```
L = -(1/n) Σ[yᵢ log(p̂ᵢ) + (1-yᵢ) log(1-p̂ᵢ)]
```

This loss function has a beautiful property: it is **convex** when combined with the logistic function, so gradient descent always finds the global minimum.

```python
import numpy as np

def binary_cross_entropy(y_true, y_pred_proba, eps=1e-15):
    """
    Binary cross-entropy loss.
    eps prevents log(0) = -infinity.
    """
    p = np.clip(y_pred_proba, eps, 1 - eps)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

# Demonstrate: loss is low when predictions match truth
y_true = np.array([1, 1, 0, 0, 1])

# Good predictions (high confidence and correct)
y_good = np.array([0.95, 0.88, 0.05, 0.12, 0.92])
# Bad predictions (confident but wrong)
y_bad  = np.array([0.10, 0.15, 0.90, 0.85, 0.08])
# Random predictions
y_rand = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

print(f"Good predictions BCE:    {binary_cross_entropy(y_true, y_good):.4f}")
print(f"Random predictions BCE:  {binary_cross_entropy(y_true, y_rand):.4f}")
print(f"Bad predictions BCE:     {binary_cross_entropy(y_true, y_bad):.4f}")

# Manually verify for one example
# y=1, p=0.95: loss = -log(0.95) = 0.0513
print(f"\nManual check: -log(0.95) = {-np.log(0.95):.4f}")
# y=0, p=0.90: loss = -log(1-0.90) = -log(0.10) = 2.303
print(f"Manual check: -log(0.10) = {-np.log(0.10):.4f}")
```

---

## Gradient of Cross-Entropy Loss

The gradient of binary cross-entropy with respect to `w` is:

```
∂L/∂w = (1/n) Xᵀ(p̂ - y)
∂L/∂b = (1/n) Σ(p̂ᵢ - yᵢ)
```

This is the same form as linear regression! The only difference is that the "error" term is now `p̂ - y` where `p̂ = σ(Xw + b)`.

---

## Implementing Logistic Regression from Scratch

```python
import numpy as np

class LogisticRegressionScratch:
    """Logistic regression trained with gradient descent."""

    def __init__(self, lr=0.1, n_iter=1000):
        self.lr     = lr
        self.n_iter = n_iter

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n, p = X.shape
        self.w = np.zeros(p)
        self.b = 0.0
        self.losses = []

        for _ in range(self.n_iter):
            # Forward pass
            z    = X @ self.w + self.b
            p_hat = self._sigmoid(z)

            # Loss (binary cross-entropy)
            eps  = 1e-15
            p_c  = np.clip(p_hat, eps, 1-eps)
            loss = -np.mean(y * np.log(p_c) + (1-y) * np.log(1-p_c))
            self.losses.append(loss)

            # Gradients
            error   = p_hat - y
            grad_w  = (1/n) * X.T @ error
            grad_b  = (1/n) * error.sum()

            # Update
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

        return self

    def predict_proba(self, X):
        """Returns P(y=1|x) for each example."""
        z = X @ self.w + self.b
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# ── Test on synthetic binary classification ──────────────────────
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
X, y = make_classification(
    n_samples=500, n_features=4, n_informative=3,
    n_redundant=1, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = LogisticRegressionScratch(lr=0.5, n_iter=500)
model.fit(X_train_s, y_train)

train_acc = model.score(X_train_s, y_train)
test_acc  = model.score(X_test_s, y_test)

print(f"Train accuracy: {train_acc:.4f}")
print(f"Test accuracy:  {test_acc:.4f}")
print(f"Final BCE loss: {model.losses[-1]:.4f}")
```

---

## Comparing with Sklearn

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

np.random.seed(42)
X, y = make_classification(n_samples=500, n_features=4, n_informative=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Sklearn's logistic regression
sk_model = LogisticRegression(max_iter=1000, random_state=42)
sk_model.fit(X_train_s, y_train)

print(f"Sklearn train accuracy: {sk_model.score(X_train_s, y_train):.4f}")
print(f"Sklearn test accuracy:  {sk_model.score(X_test_s, y_test):.4f}")
print(f"Coefficients: {sk_model.coef_[0]}")
print(f"Intercept:    {sk_model.intercept_[0]:.4f}")

# Predicted probabilities (two columns: P(y=0), P(y=1))
probas = sk_model.predict_proba(X_test_s[:5])
print("\nFirst 5 test examples — probabilities:")
print(f"  {'P(class=0)':>12} {'P(class=1)':>12} {'Prediction':>12}")
for p0, p1 in probas:
    pred = 1 if p1 >= 0.5 else 0
    print(f"  {p0:>12.4f} {p1:>12.4f} {pred:>12}")
```

---

## The Decision Boundary

The decision boundary is the set of points where the model is exactly 50/50 uncertain. Mathematically: `wᵀx + b = 0`.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

np.random.seed(0)
X, y = make_classification(
    n_samples=200, n_features=2, n_informative=2,
    n_redundant=0, n_clusters_per_class=1, random_state=0
)

scaler = StandardScaler()
X_s = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_s, y)

w = model.coef_[0]
b = model.intercept_[0]

print(f"Decision boundary: {w[0]:.3f}*x1 + {w[1]:.3f}*x2 + {b:.3f} = 0")
print(f"=> x2 = {-w[0]/w[1]:.3f}*x1 + {-b/w[1]:.3f}")

# Check points on the boundary
# When z = 0, predict exactly 0.5
x1_values = np.linspace(-3, 3, 5)
x2_boundary = (-w[0] * x1_values - b) / w[1]

print("\nPoints on the decision boundary (should predict prob = 0.5):")
for x1, x2 in zip(x1_values, x2_boundary):
    point = np.array([[x1, x2]])
    prob  = model.predict_proba(point)[0, 1]
    print(f"  ({x1:>5.1f}, {x2:>6.2f}) → P(y=1) = {prob:.4f}")
```

---

## Multi-Class Classification with Softmax

For more than 2 classes, we use the **softmax function**:

```
softmax(z)_k = e^{z_k} / Σⱼ e^{z_j}
```

Softmax converts a vector of scores into a probability distribution (all positive, sums to 1).

For K classes, we learn K sets of weights. The predicted class is the one with the highest probability.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def softmax(z):
    """Numerically stable softmax."""
    z_shifted = z - z.max(axis=1, keepdims=True)  # subtract max for stability
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)

# Example with 3 classes (Iris dataset)
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# multi_class='multinomial' uses softmax
model = LogisticRegression(multi_class='multinomial', max_iter=500, random_state=42)
model.fit(X_train_s, y_train)

print(f"Iris 3-class classification:")
print(f"  Test accuracy: {model.score(X_test_s, y_test):.4f}")
print(f"  Classes: {iris.target_names}")
print(f"  Weight matrix shape: {model.coef_.shape}  (K classes × n_features)")

# Show probabilities for first 5 test examples
probas = model.predict_proba(X_test_s[:5])
preds  = model.predict(X_test_s[:5])

print("\nFirst 5 test examples:")
print(f"  {'setosa':>10} {'versicolor':>12} {'virginica':>10} {'Predicted':>12} {'True':>8}")
for i, (p, pred) in enumerate(zip(probas, preds)):
    true_label = iris.target_names[y_test[i]]
    pred_label = iris.target_names[pred]
    print(f"  {p[0]:>10.4f} {p[1]:>12.4f} {p[2]:>10.4f} {pred_label:>12} {true_label:>8}")

# Demonstrate softmax
z = np.array([[2.0, 1.0, 0.5]])
print(f"\nSoftmax([2.0, 1.0, 0.5]) = {softmax(z)[0]}")
print(f"Sum = {softmax(z)[0].sum():.4f}  (always 1)")
```

---

## One-vs-Rest (OvR) Classification

An alternative to softmax: train K binary classifiers, one for each class vs. all others.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# One-vs-Rest
ovr_model = LogisticRegression(multi_class='ovr', max_iter=500)
ovr_model.fit(X_train_s, y_train)

# Multinomial (softmax)
mn_model = LogisticRegression(multi_class='multinomial', max_iter=500)
mn_model.fit(X_train_s, y_train)

print(f"One-vs-Rest accuracy:   {ovr_model.score(X_test_s, y_test):.4f}")
print(f"Multinomial accuracy:   {mn_model.score(X_test_s, y_test):.4f}")
```

---

## Regularization: L1 and L2

Logistic regression can overfit, especially with many features. Regularization adds a penalty to the loss function to keep weights small.

**L2 (Ridge):** Adds `λ Σwᵢ²` to the loss. Keeps all weights small but non-zero.

**L1 (Lasso):** Adds `λ Σ|wᵢ|` to the loss. Drives some weights exactly to zero — acts as feature selection.

In sklearn's LogisticRegression, the regularization strength is controlled by `C = 1/λ`. Smaller C = stronger regularization.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
# Many features — some irrelevant
X, y = make_classification(
    n_samples=300, n_features=20, n_informative=5,
    n_redundant=5, n_repeated=0, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"{'C':>8} {'penalty':>8} {'train acc':>10} {'test acc':>10} {'nonzero weights':>16}")
print("-" * 56)
for C in [0.01, 0.1, 1.0, 10.0]:
    for penalty in ['l1', 'l2']:
        solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
        model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=1000)
        model.fit(X_train_s, y_train)
        tr_acc  = model.score(X_train_s, y_train)
        te_acc  = model.score(X_test_s, y_test)
        nonzero = np.sum(np.abs(model.coef_[0]) > 1e-6)
        print(f"{C:>8.2f} {penalty:>8} {tr_acc:>10.4f} {te_acc:>10.4f} {nonzero:>16}")
```

---

## Interpreting Coefficients

The odds ratio interpretation: for logistic regression, `exp(wᵢ)` is the multiplicative change in odds for a one-unit increase in feature i.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(42)
n = 500

# Medical dataset: predict diabetes (0/1)
glucose = np.random.normal(120, 30, n)
bmi     = np.random.normal(28, 5, n)
age     = np.random.normal(45, 15, n)

# True relationship: probability depends on glucose and bmi
logit = -4 + 0.03 * glucose + 0.1 * bmi + 0.02 * age
prob  = 1 / (1 + np.exp(-logit))
y     = (prob > np.random.uniform(0, 1, n)).astype(int)

X = np.column_stack([glucose, bmi, age])
feature_names = ['glucose', 'bmi', 'age']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = LogisticRegression(C=1.0, max_iter=1000)
model.fit(X_train_s, y_train)

print(f"Test accuracy: {model.score(X_test_s, y_test):.4f}")
print(f"Diabetes base rate: {y.mean():.3f}\n")

print(f"{'Feature':>10} {'Coeff':>10} {'Odds Ratio':>12} {'Interpretation':>30}")
print("-" * 66)
for name, coef in zip(feature_names, model.coef_[0]):
    odds_ratio = np.exp(coef)
    if odds_ratio > 1:
        interp = f"+{(odds_ratio-1)*100:.0f}% odds per std unit"
    else:
        interp = f"{(odds_ratio-1)*100:.0f}% odds per std unit"
    print(f"{name:>10} {coef:>10.4f} {odds_ratio:>12.4f} {interp:>30}")
```

---

## ROC Curve and AUC

The **ROC (Receiver Operating Characteristic) curve** shows how the true positive rate (recall) and false positive rate trade off as you vary the classification threshold.

- TPR (True Positive Rate / Recall) = TP / (TP + FN)
- FPR (False Positive Rate) = FP / (FP + TN)

**AUC (Area Under the Curve)**: a single number from 0 to 1.
- AUC = 1.0: perfect classifier
- AUC = 0.5: random (no better than chance)
- AUC < 0.5: worse than random (flip the predictions!)

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

np.random.seed(42)
X, y = make_classification(n_samples=500, n_features=5, n_informative=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_s, y_train)

# Get predicted probabilities (not class labels)
y_proba = model.predict_proba(X_test_s)[:, 1]  # P(y=1)

auc = roc_auc_score(y_test, y_proba)
print(f"AUC-ROC: {auc:.4f}")

# Compute ROC curve manually for a few thresholds
thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
print(f"\n{'Threshold':>10} {'TPR (Recall)':>14} {'FPR':>8} {'Precision':>10}")
print("-" * 46)
for thresh in thresholds:
    preds = (y_proba >= thresh).astype(int)
    tp = ((preds == 1) & (y_test == 1)).sum()
    fp = ((preds == 1) & (y_test == 0)).sum()
    tn = ((preds == 0) & (y_test == 0)).sum()
    fn = ((preds == 0) & (y_test == 1)).sum()

    tpr  = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0

    print(f"{thresh:>10.1f} {tpr:>14.4f} {fpr:>8.4f} {prec:>10.4f}")

# Using sklearn
fprs, tprs, _ = roc_curve(y_test, y_proba)
print(f"\nAUC computed by sklearn: {roc_auc_score(y_test, y_proba):.4f}")
print(f"Number of threshold points on ROC curve: {len(fprs)}")
```

---

## When Logistic Regression Works Well

Logistic regression thrives when:
1. **The relationship is (roughly) linear in feature space** — the classes can be separated by a hyperplane
2. **Features are independent and well-scaled** — collinearity hurts interpretation but not always accuracy
3. **Interpretability is important** — coefficients have clear meaning
4. **Training data is small to medium** — it trains fast even with little data
5. **A probability is needed, not just a class label** — the output is calibrated

It struggles when:
- The decision boundary is highly nonlinear
- There are strong interactions between features that are not captured as input
- The number of features greatly exceeds the number of samples (without regularization)

---

## Summary

| Concept | Key Formula / Idea |
|---|---|
| Sigmoid | σ(z) = 1/(1+e^{-z}) → maps to (0,1) |
| Logistic model | p̂ = σ(wᵀx + b) |
| Decision boundary | wᵀx + b = 0 (where p̂ = 0.5) |
| Binary cross-entropy | -[y log(p̂) + (1-y) log(1-p̂)] |
| Gradient | (1/n) Xᵀ(p̂ - y) |
| Multi-class softmax | Generalizes sigmoid to K classes |
| L1/L2 regularization | Controlled by C=1/λ in sklearn |
| AUC-ROC | Area under TPR vs FPR curve (0.5 = random, 1.0 = perfect) |

Logistic regression is your go-to first classifier. In the next lesson, we'll move to **decision trees** — a completely different paradigm that makes decisions through a series of if-then rules.
