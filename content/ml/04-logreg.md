---
title: Logistic Regression
track: ml
order: 4
estimatedTime: 40
difficulty: intermediate
---

# Logistic Regression

Despite the name, logistic regression is a **classifier**, not a regressor. It predicts the probability that an input belongs to a class. It's fast, interpretable, and often the right first model to try before anything fancier.

---

## 1. The Sigmoid Function

Logistic regression builds on linear regression by squashing the output into a probability:

$$\hat{p} = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

The sigmoid $\sigma(z)$ maps any real number to $(0, 1)$:

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Properties:
print(sigmoid(0))    # 0.5  — exactly on the boundary
print(sigmoid(5))    # 0.993 — very confident class 1
print(sigmoid(-5))   # 0.007 — very confident class 0
print(sigmoid(100))  # ≈ 1.0
print(sigmoid(-100)) # ≈ 0.0

z = np.linspace(-6, 6, 100)
plt.plot(z, sigmoid(z))
plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='decision boundary')
plt.title('Sigmoid Function')
plt.xlabel('z = w·x + b')
plt.ylabel('P(class=1)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**Decision rule:** If $\hat{p} > 0.5$, predict class 1; otherwise predict class 0. (The 0.5 threshold can be adjusted for different precision/recall tradeoffs.)

---

## 2. The Loss Function: Binary Cross-Entropy

We can't use MSE for classification (it creates a non-convex loss surface). Instead, we use **binary cross-entropy (log loss)**:

$$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

**Why this makes sense:**
- If $y = 1$ and $\hat{p} = 0.99$: loss = $-\log(0.99) \approx 0.01$ (tiny — correct prediction)
- If $y = 1$ and $\hat{p} = 0.01$: loss = $-\log(0.01) \approx 4.6$ (huge — wrong and confident)

```python
def binary_cross_entropy(y_true, y_pred, eps=1e-9):
    # eps prevents log(0) which is undefined
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Good model vs bad model
good_preds = np.array([0.95, 0.05, 0.90, 0.10])  # high confidence, correct
bad_preds  = np.array([0.55, 0.45, 0.60, 0.40])  # barely above/below threshold
true       = np.array([1, 0, 1, 0])

print(f"Good model loss: {binary_cross_entropy(true, good_preds):.4f}")  # low
print(f"Bad model loss:  {binary_cross_entropy(true, bad_preds):.4f}")   # higher
```

---

## 3. From Scratch (Gradient Descent)

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LogisticRegressionScratch:
    def __init__(self, lr=0.1, epochs=500):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        n, f = X.shape
        self.w = np.zeros(f)
        self.b = 0.0
        self.losses = []

        for _ in range(self.epochs):
            z     = X @ self.w + self.b
            p     = 1 / (1 + np.exp(-z))
            error = p - y                          # gradient of log loss
            self.w -= self.lr * (X.T @ error) / n
            self.b -= self.lr * error.mean()
            loss = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
            self.losses.append(loss)

    def predict_proba(self, X):
        return 1 / (1 + np.exp(-(X @ self.w + self.b)))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# Test
X, y = make_classification(n_samples=500, n_features=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

model = LogisticRegressionScratch(lr=0.5, epochs=300)
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc   = np.mean(preds == y_test)
print(f"Test accuracy: {acc:.2%}")
```

---

## 4. Using scikit-learn

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# C = 1/λ — the inverse of regularization strength
# Higher C = less regularization = more complex model
model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]  # probability of class 1

print(f"Accuracy:  {accuracy_score(y_test, preds):.2%}")
print(f"AUC-ROC:   {roc_auc_score(y_test, proba):.4f}")
print("\n", classification_report(y_test, preds))
```

---

## 5. Adjusting the Decision Threshold

The default threshold is 0.5, but you can change it to trade precision for recall:

```python
# Example: medical diagnosis where missing a positive (recall) is worse
threshold = 0.3   # lower threshold = more positives predicted = higher recall

preds_strict = (proba >= threshold).astype(int)

from sklearn.metrics import precision_score, recall_score, f1_score
print(f"Threshold: {threshold}")
print(f"Precision: {precision_score(y_test, preds_strict):.3f}")
print(f"Recall:    {recall_score(y_test, preds_strict):.3f}")
print(f"F1:        {f1_score(y_test, preds_strict):.3f}")
```

In competition settings, the optimal threshold often isn't 0.5. Use your validation set to search for the best value.

---

## 6. Multiclass Classification

For more than 2 classes, logistic regression uses **softmax** (also called multinomial logistic regression):

$$\hat{p}_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

Each class gets a score, and softmax converts them to probabilities that sum to 1.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)  # 3 classes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# multi_class='auto' uses softmax for 3+ classes
model = LogisticRegression(multi_class='auto', max_iter=1000)
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.2%}")

# Predicted probabilities: one column per class
proba = model.predict_proba(X_test)  # shape: (n_test, 3)
print("First test sample probabilities:", proba[0].round(3))
```

---

## 7. Feature Importance

Unlike tree models, logistic regression gives you **weights per feature**, directly interpretable as importance:

```python
feature_names = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']
for i, name in enumerate(feature_names):
    print(f"{name:15s} → weights: {model.coef_[:, i].round(3)}")
```

A large absolute weight (positive or negative) means that feature strongly influences the prediction.

---

## 8. Classify These Scenarios

```widget
{
  "type": "concept-sort",
  "title": "Choose the Right Output",
  "categories": [
    { "name": "Use sigmoid (binary)", "color": "#5B5BD6" },
    { "name": "Use softmax (multiclass)", "color": "#F97316" }
  ],
  "items": [
    { "text": "Spam / Not spam", "category": "Use sigmoid (binary)" },
    { "text": "Cat, Dog, or Bird?", "category": "Use softmax (multiclass)" },
    { "text": "Disease present or absent", "category": "Use sigmoid (binary)" },
    { "text": "Digit 0-9 recognition", "category": "Use softmax (multiclass)" },
    { "text": "Click or no-click prediction", "category": "Use sigmoid (binary)" },
    { "text": "Sentiment: positive/neutral/negative", "category": "Use softmax (multiclass)" }
  ]
}
```

---

## Practice Questions

**Quick check:** Logistic regression outputs P(spam) = 0.72. What class does it predict with threshold 0.5? What if you change the threshold to 0.8?
> At threshold 0.5: **spam** (0.72 > 0.5). At threshold 0.8: **not spam** (0.72 < 0.8). Raising the threshold makes the model more conservative — it only predicts positive when very confident. This increases precision but reduces recall.

**Quick check:** You have a spam filter. A false positive (legitimate email marked as spam) is much more costly than a false negative (spam gets through). How should you adjust the classification threshold?
> **Raise the threshold** (e.g., from 0.5 to 0.8). This reduces false positives (the model only labels something spam when very confident) at the cost of more false negatives. The precision-recall tradeoff directly serves the business requirement.

**Quick check:** Logistic regression gets 78% accuracy on training but only 79% on test. What does this tell you, and what should you try?
> The model is **underfitting** — tiny gap but both numbers are modest. It's too simple for the problem. Try: adding feature interactions (x₁·x₂), polynomial features, or switch to a tree-based model that can capture non-linearity.

---

## Summary

| Concept | Key point |
|---|---|
| Sigmoid | Squashes linear output to (0, 1) — the predicted probability |
| Decision boundary | The hyperplane where $\sigma(\mathbf{w}^T\mathbf{x} + b) = 0.5$ |
| Log loss | Penalizes confident wrong predictions harshly |
| C parameter | `C = 1/λ` — higher C = less regularization |
| Threshold | Default is 0.5 — tune on val set for precision/recall tradeoff |
| Softmax | Generalization of sigmoid to K classes — outputs sum to 1 |

Logistic regression is your fastest, most interpretable classification baseline. If it works well, don't overcomplicate. If not, move to trees or neural networks.
