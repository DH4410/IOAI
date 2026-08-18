---
title: Evaluation Metrics
track: ml
order: 7
estimatedTime: 40
difficulty: intermediate
---

# Evaluation Metrics

You cannot improve what you do not measure. Picking the right metric matters as much as picking the right model — especially in IOAI, where the competition specifies which metric your submission is scored on.

---

## 1. Classification Metrics

### Accuracy

The fraction of predictions that were correct.

```python
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_true, y_pred)
print(f'Accuracy: {acc:.2%}')   # e.g. 91.50%
```

**When to use:** Only when classes are balanced. If 95% of your data is class 0, a model that always predicts 0 gets 95% accuracy — but that is useless.

### Confusion Matrix

Shows all four outcomes for a binary classifier:

```
                 Predicted:
                  Pos   Neg
Actual: Pos    [ TP   FN ]
        Neg    [ FP   TN ]
```

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm).plot()
```

### Precision and Recall

**Precision** = of all the times we said "positive", how often were we right?
```
Precision = TP / (TP + FP)
```

**Recall** = of all actual positives, how many did we catch?
```
Recall = TP / (TP + FN)
```

There is always a tradeoff: if you predict positive more aggressively (lower threshold), recall goes up but precision goes down.

**Quick check:** A spam filter marks 10 emails as spam. 9 of them are actually spam, but it missed 5 real spam emails. What are precision and recall?
> Precision = 9/10 = 90%. Recall = 9/(9+5) = 64%. High precision, lower recall.

### F1 Score

The harmonic mean of precision and recall. Use it when both matter and classes are imbalanced.

```
F1 = 2 * (precision * recall) / (precision + recall)
```

```python
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

print(classification_report(y_true, y_pred))
# Gives precision, recall, F1 for each class
```

For multi-class problems, average across classes:
- `average='macro'` — equal weight to each class
- `average='weighted'` — weight by class frequency
- `average='micro'` — sum TP/FP/FN across all classes first

```python
f1 = f1_score(y_true, y_pred, average='weighted')
```

### ROC-AUC

AUC (Area Under the ROC Curve) measures how well a model separates the two classes, regardless of the decision threshold.

```python
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_true, y_scores)   # y_scores = predicted probabilities
```

AUC of 0.5 = random. AUC of 1.0 = perfect. Good models land between 0.85 and 0.98.

---

## 2. Regression Metrics

### MAE — Mean Absolute Error

The average absolute difference between prediction and truth. Easy to interpret: "on average, my predictions are off by X units."

```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred)
```

### MSE and RMSE

**MSE** penalizes large errors more than MAE (because errors are squared).

**RMSE** = sqrt(MSE). Same units as the target, like MAE, but more sensitive to outliers.

```python
from sklearn.metrics import mean_squared_error
import numpy as np

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
```

### R² — Coefficient of Determination

How much variance in the target does the model explain?
- R² = 1.0 → perfect
- R² = 0.0 → model is no better than just predicting the mean
- R² < 0 → model is worse than predicting the mean

```python
from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)
```

---

## 3. Choosing the Right Metric

| Situation | Use |
|---|---|
| Balanced classes | Accuracy |
| Imbalanced classes | F1, ROC-AUC |
| Cost of FP ≠ cost of FN | Precision or Recall separately |
| Regression, care about big errors | RMSE |
| Regression, robust to outliers | MAE |
| Regression, want interpretable % | R² |

**In IOAI:** Read the problem statement carefully. It will tell you exactly which metric you are scored on. Optimize for that metric specifically — sometimes this means choosing a different loss function or post-processing your predictions.

---

## 4. Cross-Validation for Reliable Scores

A single train/test split gives noisy estimates. Use 5-fold CV:

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=200, random_state=42)
scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
print(f'F1: {scores.mean():.4f} ± {scores.std():.4f}')
```

Always report mean ± std. A model with 0.91 ± 0.02 is more reliable than one with 0.93 ± 0.06.

---

## 5. Sort These Metric Scenarios

```widget
{
  "type": "concept-sort",
  "title": "Which Metric? Match the Scenario",
  "categories": [
    { "name": "Use Accuracy", "color": "#22C55E" },
    { "name": "Use F1 / AUC", "color": "#5B5BD6" },
    { "name": "Use RMSE / MAE", "color": "#F97316" }
  ],
  "items": [
    { "text": "Predicting house prices", "category": "Use RMSE / MAE" },
    { "text": "MNIST digit recognition (balanced classes)", "category": "Use Accuracy" },
    { "text": "Fraud detection (0.1% fraudulent transactions)", "category": "Use F1 / AUC" },
    { "text": "Cancer diagnosis (FN is life-threatening)", "category": "Use F1 / AUC" },
    { "text": "Predicting tomorrow's temperature in °C", "category": "Use RMSE / MAE" },
    { "text": "Spam filter (balanced spam/ham dataset)", "category": "Use Accuracy" }
  ]
}
```

---

## Practice Questions

**Quick check:** A classifier has TP=80, FP=20, FN=40, TN=860. Calculate precision, recall, and F1.
> Precision = 80/(80+20) = **0.80**. Recall = 80/(80+40) = **0.667**. F1 = 2·(0.80·0.667)/(0.80+0.667) = **0.727**. The model is more precise than it is sensitive — it misses 1 in 3 actual positives.

**Quick check:** Why is AUC-ROC useful even when you don't know the final decision threshold yet?
> AUC measures how well the model *ranks* examples — whether it gives higher scores to positives than negatives — across ALL possible thresholds. It's threshold-independent. A model with AUC=0.92 means: pick a random positive and a random negative — there's a 92% chance the positive gets a higher score.

**Quick check:** Model A: precision 0.90, recall 0.40. Model B: precision 0.60, recall 0.85. For a spam filter where false positives (blocking real emails) are the priority concern, which model is better?
> **Model A** (precision 0.90). High precision means most things it flags as spam really are spam — few legitimate emails get blocked. Model B blocks more spam (higher recall) but also blocks many legitimate emails (lower precision).

---

## Summary

| Metric | Formula | When to use |
|---|---|---|
| Accuracy | correct / total | Balanced classes |
| Precision | TP/(TP+FP) | When FP cost is high |
| Recall | TP/(TP+FN) | When FN cost is high |
| F1 | 2·P·R/(P+R) | Imbalanced classification |
| AUC | — | Ranking/threshold-agnostic |
| RMSE | sqrt(MSE) | Regression, punish big errors |
| MAE | mean(|y-ŷ|) | Regression, robust to outliers |
| R² | — | Regression, % variance explained |
