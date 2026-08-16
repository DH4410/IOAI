---
title: What is Machine Learning?
track: ml
order: 1
estimatedTime: 25
difficulty: beginner
---

# What is Machine Learning?

Machine learning is about building programs that learn from data instead of following hand-written rules. This lesson gives you the mental model you need before diving into specific algorithms.

---

## 1. Why Machine Learning?

Imagine writing rules to detect spam email. You might try:
- If it contains "FREE MONEY" -> spam
- If sender is unknown -> spam

This breaks immediately. Spammers write "Fr3e M0n3y" and your rules fail. You spend all your time chasing edge cases.

Machine learning takes a different approach: show the program thousands of examples of spam and non-spam and let it learn the patterns itself.

**Core idea:** Learn from data, not from hand-crafted rules.

---

## 2. The Three Types of ML

**Supervised Learning:** You provide labeled examples (input, correct output). The model learns to map inputs to outputs.
- Example: 1000 emails labeled "spam" or "not spam". The model learns to classify new emails.
- Examples: classification, regression.

**Unsupervised Learning:** No labels. The model finds structure in the data on its own.
- Example: Group customers by purchasing behavior without knowing what the groups are.
- Examples: clustering, dimensionality reduction.

**Reinforcement Learning:** An agent takes actions in an environment and learns from rewards and penalties.
- Example: A game-playing agent learns to win by getting rewarded for good moves.

In IOAI competitions, you mostly do supervised learning.

**Quick check:** You have a dataset of house sizes and prices. What type of ML is predicting the price of a new house?
> Supervised learning (regression). You have labeled examples (size -> price) and want to predict a number for new houses.

---

## 3. Training, Validation, and Test Sets

Never evaluate your model on the data it trained on. That would be like a student memorizing exam answers instead of learning the material.

Always split your data:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Training set (80%):** The model sees this data and adjusts its weights/parameters.

**Validation set:** Used during training to tune hyperparameters and detect overfitting. Carve it out of the training set.

**Test set (20%):** Evaluate the final model. Look at this only once, at the very end.

---

## 4. Overfitting and Underfitting

**Underfitting:** The model is too simple. It performs badly on both training and test data. (High bias)

**Overfitting:** The model memorizes the training data and fails on new data. Training accuracy is high but test accuracy is low. (High variance)

```
Underfit:  Train=60%  Test=58%   (model is too simple)
Good fit:  Train=92%  Test=89%   (model generalizes well)
Overfit:   Train=99%  Test=71%   (model memorizes training data)
```

**Quick check:** Your model gets 99% on training data but 65% on test data. What is this called and how would you fix it?
> Overfitting. Fixes: use a simpler model, get more training data, add regularization, use dropout (for neural networks).

---

## 5. The Basic ML Workflow

This pattern works for almost every ML task:

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load and split data
X, y = load_data()   # your loading code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Preprocess (scale features)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)    # use the SAME scaler

# 3. Train
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Evaluate
preds = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, preds):.2%}')
```

The five steps are always: load, split, preprocess, train, evaluate.

---

## 6. Key Metrics

**Classification:**
- **Accuracy:** fraction of correct predictions. Misleading when classes are imbalanced.
- **Precision:** of the samples predicted positive, how many were actually positive?
- **Recall:** of the actual positives, how many did the model catch?
- **F1 Score:** harmonic mean of precision and recall. Use when classes are imbalanced.

**Regression:**
- **MAE** (Mean Absolute Error): average size of errors
- **MSE** (Mean Squared Error): like MAE but penalizes large errors more
- **R2:** how much variance does the model explain? 1.0 = perfect, 0.0 = no better than the mean

```python
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

# Classification
print(f'Accuracy: {accuracy_score(y_test, preds):.2%}')
print(f'F1:       {f1_score(y_test, preds, average="weighted"):.3f}')

# Regression
print(f'MSE: {mean_squared_error(y_test, preds):.4f}')
print(f'R2:  {r2_score(y_test, preds):.4f}')
```

---

## Summary

| Concept | Key point |
|---|---|
| Supervised learning | Labeled data, learn input -> output mapping |
| Train/Test split | Never evaluate on training data |
| Overfitting | Model memorizes training data, fails on new data |
| Underfitting | Model too simple for the pattern |
| F1 score | Better than accuracy when classes are imbalanced |

The rest of this track covers specific algorithms. Each one is another tool in your toolkit. The workflow above stays the same for all of them.
