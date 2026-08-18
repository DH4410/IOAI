---
title: What is Machine Learning?
track: ml
order: 1
estimatedTime: 30
difficulty: beginner
---

# What is Machine Learning?

Machine learning is the science of writing programs that learn from data. Instead of telling the computer every rule, you show it thousands of examples and it figures out the patterns itself.

---

## 1. The Core Idea — Rules vs. Learning

Imagine trying to write a spam filter using hand-crafted rules:

```
if "FREE MONEY" in email → spam
if sender unknown → spam
if has attachment → maybe spam?
```

This breaks in days. Spammers write "Fr3e M0n3y" or use your contact list. You spend your life chasing edge cases.

**Machine learning flips this**: give the system 100,000 labeled emails (spam / not spam) and let it discover the patterns automatically. It finds signals you never would have thought of — specific word sequences, timing, sender reputation, link domains — and combines them optimally.

> **Mental model:** Programming = you write the rules. ML = the data writes the rules.

---

## 2. The Three Families of ML

```
Machine Learning
├── Supervised Learning   — learn from labeled data
├── Unsupervised Learning — find structure in unlabeled data
└── Reinforcement Learning — learn from rewards and penalties
```

### Supervised Learning
You give the model inputs **and** the correct outputs. It learns the mapping.

| Task | Input | Output |
|---|---|---|
| Spam detection | Email text | Spam / Not spam |
| House prices | Size, location, rooms | Price ($) |
| Image classification | Pixel values | Cat / Dog / Car |
| IOAI competition | Feature matrix | Class label or score |

### Unsupervised Learning
No labels. The model finds structure on its own.

- **Clustering**: Group customers by purchase behavior (without telling it what the groups should be)
- **Dimensionality reduction**: Compress a 1000-feature dataset to 2 features for visualization (PCA, UMAP)
- **Anomaly detection**: Find unusual patterns in network traffic

### Reinforcement Learning
An agent takes actions in an environment, receives rewards, and learns to maximize them.
- AlphaGo / AlphaZero playing board games
- Robot locomotion
- Game-playing agents (Atari DQN)

**At IOAI competitions, you almost exclusively do supervised learning on structured data.**

---

## 3. Sort These Algorithms

```widget
{
  "type": "concept-sort",
  "title": "Supervised vs Unsupervised?",
  "categories": [
    { "name": "Supervised", "color": "#5B5BD6" },
    { "name": "Unsupervised", "color": "#22C55E" }
  ],
  "items": [
    { "text": "Linear Regression", "category": "Supervised" },
    { "text": "k-Means Clustering", "category": "Unsupervised" },
    { "text": "Decision Tree", "category": "Supervised" },
    { "text": "PCA", "category": "Unsupervised" },
    { "text": "Logistic Regression", "category": "Supervised" },
    { "text": "DBSCAN", "category": "Unsupervised" },
    { "text": "Random Forest", "category": "Supervised" },
    { "text": "Autoencoder", "category": "Unsupervised" }
  ]
}
```

---

## 4. What Does a Classifier Actually Do?

Every classification model learns a **decision boundary** — a line (or curve) that separates the classes in feature space. Try different datasets and see how the boundary changes:

```widget
{
  "type": "decision-boundary",
  "title": "What a Classifier Learns",
  "dataset": "linear"
}
```

- **Linear** dataset: a straight line separates the classes well
- **XOR** dataset: no straight line works — you need a non-linear model
- **Moons** dataset: a curve is needed — real data often looks like this

Increase the "complexity" slider to see what overfitting looks like: the boundary becomes jagged trying to get every training point right, but would fail on new data.

---

## 5. The Train / Validation / Test Split

This is the most important rule in ML: **never evaluate your model on data it was trained on**.

```
                  ┌──────── DATA ─────────┐
                  │                       │
         TRAINING (70-80%)        TEST (20-30%)
         ┌──────┴──────┐               │
    actual training   VALIDATION    Touch only
       data           (10-20%)      once at end
```

```python
from sklearn.model_selection import train_test_split

# Split into train+val and test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Further split train into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, random_state=42
)
```

**Golden rule:** The test set is sacred. You look at it exactly once — when you finalize your model. At IOAI, the public leaderboard is your validation set; the private leaderboard is your test set.

---

## 6. Overfitting vs Underfitting — The Core Tradeoff

```
UNDERFITTING              GOOD FIT              OVERFITTING
(too simple)           (generalizes well)    (too complex)

  • •  •              • • .|. • •              •  •
    •  •             • •  | •  •           .•  •  •.
  •  •               • |  •  •          .•   •   •  .
  ───────           / \ ─────\         /‾•‾\‾•‾/‾•‾\

Train: 62%         Train: 91%           Train: 99%
Test:  58%         Test:  88%           Test:  67%
```

- **Underfitting** (high bias): the model is too simple to capture the pattern
- **Overfitting** (high variance): the model memorized noise in the training data

**Fixes for overfitting:**
1. Get more data
2. Use a simpler model (fewer parameters)
3. Add regularization (L1, L2)
4. Use dropout (for neural networks)
5. Reduce features

---

## 7. The Universal ML Workflow

This pattern works for almost every supervised ML problem:

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ── Step 1: Load your data ──────────────────────────────────
X, y = load_data()   # X: (n_samples, n_features), y: (n_samples,)
print(f"Shape: {X.shape}, Classes: {np.unique(y)}")

# ── Step 2: Split into train + test ─────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify keeps class ratio
)

# ── Step 3: Preprocess (scale features) ─────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # compute mean/std from TRAIN only
X_test  = scaler.transform(X_test)       # apply same transform to TEST

# ── Step 4: Train a model ───────────────────────────────────
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── Step 5: Evaluate ────────────────────────────────────────
preds = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.2%}")
print(f"F1 Score: {f1_score(y_test, preds, average='weighted'):.3f}")
print(classification_report(y_test, preds))
```

Note: `fit_transform` on training data, just `transform` on test data. If you fit the scaler on test data too, you have **data leakage** — a common mistake that inflates scores and fails in production.

---

## 8. Choosing a Metric

Use the **right metric** for your task. Accuracy can be deeply misleading.

| Task | Good metric | Why |
|---|---|---|
| Balanced classification | Accuracy | Simple, interpretable |
| Imbalanced classes | F1, Precision, Recall | Accuracy masks poor minority-class performance |
| Medical diagnosis | Recall (sensitivity) | Missed positives are dangerous |
| Spam filter | Precision | False positives (non-spam → spam folder) are annoying |
| Regression | RMSE, R² | Units match the target |
| Ranking / retrieval | AUC-ROC | Measures discrimination, threshold-independent |

At IOAI, the problem statement tells you the metric. **Optimize for that metric specifically.**

---

## Practice Questions

**Quick check:** A dataset has 1000 train examples and 200 test examples. You fit a StandardScaler on all 1200 examples, then train on 1000. Is this valid? Why?
> **No — this is data leakage.** The scaler learned statistics (mean, std) from the 200 test examples too. The model benefits from information about test distribution that it shouldn't have access to. Always `fit_transform` on train only, then `transform` on test.

**Quick check:** Your model achieves 99% accuracy on a medical test where 99% of patients are healthy. Is this a good model?
> **No.** A trivial model that always predicts "healthy" also gets 99% accuracy. The model likely learned to always predict the majority class. Use F1 or recall to evaluate performance on the rare disease-positive class — what actually matters clinically.

**Quick check:** What's the difference between unsupervised and supervised learning? Give an example of each.
> **Supervised**: labeled data, learns a mapping — e.g., email (features) → spam/not-spam (label). **Unsupervised**: no labels, finds structure — e.g., k-means clustering customers into groups without any predefined group labels. IOAI competitions are mostly supervised (you're given labels and must predict test labels).

---

## Summary

| Concept | Remember this |
|---|---|
| **Supervised learning** | Labeled data → learn input→output mapping |
| **Train/Val/Test split** | Never evaluate on training data |
| **Overfitting** | Good train, bad test → model too complex |
| **Underfitting** | Bad train, bad test → model too simple |
| **Data leakage** | Fitting scaler on test data = cheating |
| **Metric selection** | Always match the metric to the task |

Every algorithm in this track follows the same workflow. What changes is what happens inside `model.fit()`.
