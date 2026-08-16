---
title: Logistic Regression
track: ml
order: 4
estimatedTime: 35
difficulty: intermediate
---

# Logistic Regression

Logistic regression is a classification model. Despite the name, it does not regress - it classifies. It is simple, fast, and often a good baseline before trying anything more complex.

---

## 1. The Core Idea

Linear regression predicts a number. Logistic regression predicts a **probability** between 0 and 1.

The trick: take the linear output (`w * x + b`) and squash it through the **sigmoid function**:

```
sigmoid(z) = 1 / (1 + e^(-z))
```

This turns any number into a value between 0 and 1. If the output is > 0.5, predict class 1. If < 0.5, predict class 0.

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

print(sigmoid(0))    # 0.5 - right on the boundary
print(sigmoid(3))    # ~0.95 - confident class 1
print(sigmoid(-3))   # ~0.05 - confident class 0
```

**Quick check:** The model outputs sigmoid(2.5) = 0.92. What is the predicted class?
> Class 1 (0.92 > 0.5). The model is quite confident.

---

## 2. Training: Binary Cross-Entropy Loss

MSE does not work well for classification because sigmoid makes it non-convex. Instead, logistic regression uses **binary cross-entropy**:

```
Loss = -(1/n) * sum( y * log(p) + (1-y) * log(1-p) )
```

Where `y` is the true label (0 or 1) and `p` is the predicted probability.

When `y=1`: the loss is `-log(p)`. If p=0.99, loss is tiny. If p=0.01, loss is huge.
When `y=0`: the loss is `-log(1-p)`. Penalizes high predicted probability for a 0-class sample.

The model minimizes this with gradient descent, just like linear regression.

---

## 3. From Scratch

```python
import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.1, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)

        for _ in range(self.epochs):
            z = X @ self.w + self.b
            p = 1 / (1 + np.exp(-z))         # sigmoid

            dw = (X.T @ (p - y)) / n
            db = (p - y).mean()

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_proba(self, X):
        z = X @ self.w + self.b
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


# Test
np.random.seed(42)
X = np.random.randn(200, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

model = LogisticRegression(lr=0.5, epochs=200)
model.fit(X, y)
preds = model.predict(X)
print(f'Accuracy: {(preds == y).mean():.2%}')
```

---

## 4. Using Scikit-Learn

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)

print(f'Accuracy: {accuracy_score(y_test, preds):.2%}')
print(classification_report(y_test, preds))
```

`classification_report` shows precision, recall, and F1 for each class.

**Quick check:** Your model gets 90% accuracy but the dataset has 90% class-0 samples. Is this a good model?
> Not necessarily. A model that always predicts class 0 also gets 90% accuracy. Check the F1 score and recall for class 1.

---

## 5. Multi-Class Classification

For more than 2 classes, scikit-learn handles it automatically with `multi_class='auto'`. The two main strategies:

**One-vs-Rest (OvR):** Train one binary classifier per class ("is this class A or not?"). Pick the class with the highest confidence.

**Softmax (Multinomial):** One model outputs a probability for each class. Probabilities sum to 1.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)   # 3 classes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
print(f'Accuracy: {model.score(X_test, y_test):.2%}')

# Probabilities for each class
proba = model.predict_proba(X_test[:3])
print(proba)  # each row sums to 1
```

---

## 6. The Decision Boundary

Logistic regression draws a **straight line** (or hyperplane in higher dimensions) between classes. Everything on one side is predicted class 1, everything on the other is class 0.

This means it cannot separate classes that are not linearly separable (like two concentric circles). For those cases, you need either feature engineering or a different model.

**Quick check:** Your data has two features and the classes are arranged in concentric circles. Will logistic regression work well?
> No. The decision boundary is a straight line, and no straight line can separate circles. Try a kernel SVM or a neural network.

---

## Summary

| Concept | Key point |
|---|---|
| Output | Probability via sigmoid function |
| Loss | Binary cross-entropy |
| Decision boundary | A straight line (linear) |
| Multi-class | OvR or softmax, handled automatically |
| When to use | Fast baseline, interpretable, works when classes are linearly separable |

In competition, logistic regression is often your first baseline. If it gets 85%, you know a good model can probably reach 90-95%. If it gets 55%, the task is hard or the features need work.
