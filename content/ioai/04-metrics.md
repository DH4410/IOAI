---
title: Competition Metrics Explained
track: ioai
order: 4
estimatedTime: 45
difficulty: advanced
---

# Competition Metrics Explained

If you optimize for the wrong metric, you lose. This is not a metaphor — it is a mathematical fact. A model optimized for accuracy with class imbalance can score 95% accuracy while completely ignoring the minority class, achieving 0 F1 for the important positive class. A model optimized for RMSE behaves differently from one optimized for MAE on the same dataset.

Understanding competition metrics deeply — knowing the formula, knowing what behavior it rewards, knowing how to maximize it directly — is one of the highest-leverage skills in IOAI preparation.

---

## 1. Why the Metric Matters

### A Concrete Example

Imagine a medical dataset: 95% of patients do not have a rare disease (class 0), 5% do (class 1). The competition metric is **macro F1**.

**Model A**: Always predicts class 0.
- Accuracy: 95% (looks great!)
- F1 for class 0: 2*precision*recall / (precision+recall) = 2*(0.95*1.0)/(0.95+1.0) ≈ 0.974
- F1 for class 1: undefined (predicts no positives, precision = 0/0, recall = 0)
- Macro F1: ≈ 0.487 / 2 = catastrophic

**Model B**: Deliberately sacrifices some accuracy to catch positives.
- Accuracy: 88%
- F1 for class 0: 0.930
- F1 for class 1: 0.720
- Macro F1: 0.825

Model B loses on accuracy. Model B wins the competition.

This is why you read the metric first, before any other part of the problem.

---

## 2. Classification Metrics Deep Dive

### The Confusion Matrix

Everything in classification metrics flows from the confusion matrix:

```
                  Predicted Positive    Predicted Negative
Actual Positive      TP (True Pos)        FN (False Neg)
Actual Negative      FP (False Pos)       TN (True Neg)
```

**Key relationships**:
- N = TP + FP + TN + FN (total samples)
- P = TP + FN (actual positives)
- N_neg = TN + FP (actual negatives)

All classification metrics are derived from these four numbers.

### Accuracy

```
Accuracy = (TP + TN) / (TP + FP + TN + FN)
```

**When it works**: Balanced classes, equal cost of errors.

**When it fails**: Class imbalance. With 99% negative class, a classifier that always predicts negative achieves 99% accuracy while being completely useless.

**IOAI relevance**: Accuracy is rarely the primary competition metric in serious competitions precisely because of this weakness.

### Precision and Recall

```
Precision = TP / (TP + FP)   [Of what I predicted positive, how many truly were?]
Recall    = TP / (TP + FN)   [Of all actual positives, how many did I find?]
```

**The precision-recall tradeoff**: As you lower your classification threshold (predict positive more aggressively):
- Recall increases (you catch more true positives)
- Precision decreases (you also generate more false positives)

**Which matters more?**
- Medical diagnosis: Recall matters more (don't miss diseases, even at cost of false alarms)
- Spam filtering: Precision matters more (don't block legitimate emails, even if some spam gets through)
- Competition context: Check the metric!

### F1 Score

The harmonic mean of precision and recall:

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
   = 2*TP / (2*TP + FP + FN)
```

Why harmonic mean (not arithmetic mean)?
- Arithmetic mean of precision=1.0 and recall=0.0 would be 0.5 — misleadingly high for a useless model
- Harmonic mean = 0.0 — correctly identifies the model as useless
- Harmonic mean is always ≤ arithmetic mean; it penalizes imbalance between the two components

### F_beta Score (Generalization of F1)

```
F_beta = (1 + beta²) * (Precision * Recall) / (beta² * Precision + Recall)
```

- **beta = 1**: Equal weight to precision and recall → F1
- **beta = 2**: Recall twice as important as precision (F2)
- **beta = 0.5**: Precision twice as important as recall (F0.5)

### Macro vs. Micro vs. Weighted F1

For multi-class classification:

**Macro F1**: Compute F1 for each class independently, then average.
```
Macro F1 = mean([F1_class1, F1_class2, ..., F1_classK])
```
- Treats all classes equally regardless of support (number of samples).
- Penalizes poor performance on minority classes heavily.
- Use when: all classes matter equally (e.g., medical diagnosis of rare diseases).

**Micro F1**: Aggregate TP, FP, FN across all classes, then compute F1 once.
```
Micro F1 = 2 * sum(TP_k) / (2 * sum(TP_k) + sum(FP_k) + sum(FN_k))
```
- Weights classes by their frequency (size of support).
- Equivalent to accuracy when every sample belongs to exactly one class.
- Use when: larger classes matter more.

**Weighted F1**: Compute F1 for each class, average weighted by support.
```
Weighted F1 = sum(F1_k * support_k) / N
```
- A compromise between macro and micro.

**IOAI tip**: When you see "F1" in a competition, immediately check whether it is macro, micro, or weighted. They can produce very different scores and require different optimization strategies.

### AUC-ROC (Area Under the ROC Curve)

The ROC curve plots True Positive Rate (Recall) vs. False Positive Rate at all possible thresholds.

```
TPR (Recall) = TP / (TP + FN)
FPR          = FP / (FP + TN)
```

**AUC** = Area under this curve, ranging from 0 to 1.
- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random classifier (equivalent to chance)
- AUC = 0.0: Perfect reverse classifier (swap its predictions)

**What AUC measures**: The probability that a randomly chosen positive example gets a higher predicted probability than a randomly chosen negative example.

**Why AUC is popular**: It is threshold-independent. You do not need to choose a decision threshold to compute AUC — it evaluates the model's ranking quality across all thresholds.

**When AUC is not ideal**: When you care about a specific threshold (e.g., for deployment, you will use threshold 0.5). Optimizing AUC does not guarantee good calibration at any particular threshold.

### Log Loss (Binary Cross-Entropy)

```
Log Loss = -1/N * sum(y_i * log(p_i) + (1-y_i) * log(1-p_i))
```

Where `y_i` is the true label (0 or 1) and `p_i` is the predicted probability.

**What it measures**: Penalizes confident wrong predictions much more than uncertain wrong predictions.
- Predicting 0.5 for a true positive: log loss = -log(0.5) ≈ 0.693
- Predicting 0.9 for a true positive: log loss = -log(0.9) ≈ 0.105
- Predicting 0.1 for a true positive: log loss = -log(0.1) ≈ 2.303

**Lower log loss = better**.

**What log loss rewards**: Well-calibrated probabilities. A model that outputs 0.7 for positives and 0.3 for negatives achieves lower log loss if its calibration is accurate.

**Key property**: Log loss is sensitive to output probabilities, not just rankings. This makes it distinct from AUC.

### Matthews Correlation Coefficient (MCC)

```
MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

Range: -1 (perfect inverse predictor) to 0 (random) to +1 (perfect predictor).

**Why MCC is excellent**: It produces a high score only if the model performs well on ALL four quadrants of the confusion matrix. It is robust to class imbalance in a way that accuracy is not.

**IOAI relevance**: MCC is used in some AI/bioinformatics competitions. Worth knowing cold.

### Balanced Accuracy

```
Balanced Accuracy = (TPR + TNR) / 2
                  = (Recall_pos + Recall_neg) / 2
```

Where TNR (True Negative Rate) = TN/(TN+FP) = specificity.

This is the average recall across classes, equivalent to macro recall for binary classification. Robust to class imbalance.

---

## 3. Regression Metrics Deep Dive

### Mean Absolute Error (MAE)

```
MAE = (1/N) * sum(|y_i - ŷ_i|)
```

**Properties**:
- In the same unit as the target
- All errors weighted equally (robust to outliers)
- Not differentiable at 0 (but subgradient exists)
- Optimal prediction is the **median** of the target (not mean!)

**When to use**: When you want to treat large and small errors equally and outliers are common.

### Mean Squared Error (MSE) and RMSE

```
MSE  = (1/N) * sum((y_i - ŷ_i)²)
RMSE = sqrt(MSE)
```

**Properties**:
- RMSE is in the same unit as the target
- Large errors penalized quadratically (sensitive to outliers)
- Optimal prediction is the **mean** of the target
- Differentiable everywhere (convenient for gradient-based optimization)

**When to use**: When large errors are especially costly. If predicting house prices, a $100k error should be penalized much more than two $50k errors: RMSE handles this, MAE does not.

### RMSLE (Root Mean Squared Log Error)

```
RMSLE = sqrt((1/N) * sum((log(1 + ŷ_i) - log(1 + y_i))²))
```

**Properties**:
- Penalizes under-prediction more than over-prediction (asymmetric)
- Robust to scale differences: percentage errors matter more than absolute errors
- Requires non-negative predictions (since log(negative) is undefined)

**When to use**: When the target spans several orders of magnitude (e.g., sales from $100 to $10,000,000). A prediction of $900 for a true value of $1,000 (10% error) should be penalized similarly to a prediction of $90,000 for a true value of $100,000 (10% error).

**Competition tip**: When you see RMSLE as the metric, log-transform your target before training and use RMSE as your training loss. Then inverse-transform (expm1) predictions at the end.

### R² (Coefficient of Determination)

```
R² = 1 - SS_res/SS_tot = 1 - sum((y_i - ŷ_i)²) / sum((y_i - ȳ)²)
```

**Properties**:
- Range: (-∞, 1]. R²=1 is perfect, R²=0 means model is no better than predicting the mean, R²<0 means model is worse than predicting the mean.
- Scale-invariant (a percentage of explained variance)
- Can be misleading for non-linear models or when extrapolating

### MAPE (Mean Absolute Percentage Error)

```
MAPE = (100/N) * sum(|y_i - ŷ_i| / |y_i|)
```

**Properties**:
- Expressed as a percentage — easy to interpret
- Undefined when any `y_i = 0` (division by zero)
- Asymmetric: 100% over-prediction is bounded (you can only over by 100%), but 200%+ under-prediction is common

**When used**: Business forecasting where percentage accuracy is more meaningful than absolute error.

---

## 4. Multi-Label Metrics

In multi-label classification, each sample can have multiple labels simultaneously (e.g., an image can be "dog" AND "outdoor" AND "sunny").

### Hamming Loss

```
Hamming Loss = (1/N) * (1/L) * sum over i,j of (y_ij != ŷ_ij)
```

Where L is the number of labels. Proportion of labels that are incorrectly predicted.

### Multi-Label F1 Variants

**Micro**: Aggregate TP/FP/FN across all (sample, label) pairs.
**Macro**: Compute F1 per label, then average (all labels weighted equally).
**Sample-averaged**: Compute F1 per sample, then average.

---

## 5. Ranking Metrics

Used when the task is to rank items (search results, recommendations).

### NDCG (Normalized Discounted Cumulative Gain)

```
DCG@k  = sum_{i=1}^{k} (2^rel_i - 1) / log2(i + 1)
NDCG@k = DCG@k / IDCG@k
```

Where `rel_i` is the relevance of item at position i, and IDCG is the DCG of the ideal ranking (best possible ordering).

**Range**: 0 to 1 (1 = perfect ranking).

**IOAI relevance**: If a competition involves ranking, recommendation, or information retrieval, NDCG is likely.

### MAP (Mean Average Precision)

```
AP@k = (1/R) * sum_{k=1}^{K} P(k) * rel(k)
MAP  = (1/Q) * sum_{q=1}^{Q} AP@k_q
```

Where R = total relevant items, P(k) = precision at cut k, rel(k) = 1 if item at k is relevant.

---

## 6. Probability Calibration

### What Is Calibration?

A model is **calibrated** if, among all samples where it predicts probability 0.7, approximately 70% are actually positive. An uncalibrated model might have AUC 0.90 but terrible log loss because its probability estimates are systematically off.

**Calibration curve** (reliability diagram): Plot predicted probabilities (x-axis) vs. observed frequency (y-axis). A well-calibrated model's points fall on the diagonal y=x.

### Why Calibration Matters

- For **AUC**: Calibration does not matter. Only ranking matters.
- For **log loss**: Calibration is critical. Overconfident predictions (0.99 for events that happen 70% of the time) get severely penalized.
- For **threshold-based metrics** (F1, accuracy): Calibration affects optimal threshold selection.

### Platt Scaling

Train a logistic regression on top of the raw scores of your model:

```python
from sklearn.calibration import CalibratedClassifierCV

# Method 1: Direct calibration on held-out data
calibrator = CalibratedClassifierCV(base_model, method='sigmoid', cv='prefit')
# base_model must already be fitted
calibrator.fit(X_calib, y_calib)
calibrated_probs = calibrator.predict_proba(X_test)[:, 1]
```

**Platt scaling**: Fits a logistic regression `σ(a*score + b)` to map raw scores to probabilities.
**Isotonic regression**: Fits a non-decreasing step function — more flexible but needs more data (>1000 samples recommended).

### Threshold Moving

For F1 optimization, the decision threshold (default 0.5) is often not optimal. Move it:

```python
from sklearn.metrics import f1_score
import numpy as np

def find_optimal_threshold(y_true, y_prob, metric=f1_score):
    """Find threshold that maximizes F1 on validation data."""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_score = 0
    
    for thresh in thresholds:
        preds = (y_prob >= thresh).astype(int)
        score = metric(y_true, preds)
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    return best_threshold, best_score

threshold, score = find_optimal_threshold(y_val, val_probs)
final_preds = (test_probs >= threshold).astype(int)
```

**Warning**: Fitting the threshold on validation data then applying it to test data is fine. Fitting it on test data is leakage.

---

## 7. Directly Optimizing for the Competition Metric

### Gradient Boosting with Custom Loss

For metrics that are not standard loss functions (e.g., AUC, F1), you can sometimes optimize a surrogate:

- **For AUC**: Use log loss or binary cross-entropy as surrogate (they both reward good ranking/calibration)
- **For F1**: Optimize log loss, then tune threshold
- **For RMSLE**: Log-transform target, optimize RMSE
- **For MAPE**: Use Tweedie loss or custom objective in LightGBM

LightGBM custom objective example:
```python
# Optimize for RMSLE: log-transform target
import numpy as np

y_log = np.log1p(y_train)
model = LGBMRegressor(objective='regression', metric='rmse')
model.fit(X_train, y_log)
preds_log = model.predict(X_test)
preds_original = np.expm1(preds_log)
```

### Custom Metric for CV Evaluation

Even if you cannot optimize a metric directly, always evaluate using the competition metric:

```python
from sklearn.metrics import make_scorer, f1_score

# Create scorer for macro F1
macro_f1_scorer = make_scorer(f1_score, average='macro')
cv_scores = cross_val_score(model, X, y, cv=5, scoring=macro_f1_scorer)
```

---

## 8. Common IOAI Metric Patterns

Based on the types of problems likely in IOAI, here is what to expect:

| Task Type | Likely Metric | Key Property |
|---|---|---|
| Binary classification (balanced) | Accuracy or AUC-ROC | Accuracy is interpretable; AUC is threshold-free |
| Binary classification (imbalanced) | F1, AUC-ROC, or MCC | Penalizes ignoring minority class |
| Multi-class (balanced) | Accuracy or micro F1 | Similar behavior for balanced data |
| Multi-class (imbalanced) | Macro F1 | Equal weight to all classes |
| Regression | RMSE or MAE | RMSE penalizes outliers more |
| Ranking/recommendation | NDCG, MAP | Rewards correct ordering |
| Probabilistic output | Log Loss | Rewards calibrated probabilities |

---

## 9. Metric Pitfalls — What to Watch For

### Pitfall 1: Checking "Accuracy" When Metric Is F1

Students often compute `model.score(X_val, y_val)` which returns accuracy by default. If the competition metric is F1, this is meaningless. Always evaluate using the correct metric.

### Pitfall 2: Forgetting to Average Type for Multi-Class F1

`f1_score(y_true, y_pred)` with default arguments raises an error for multi-class. You must specify `average=`. Common mistake: using `average='binary'` when the problem is multi-class.

### Pitfall 3: RMSLE on Negative Predictions

If your model predicts negative values and the metric is RMSLE, you get a ValueError (log of negative number). Always clip predictions to be non-negative when using RMSLE:
```python
preds = np.maximum(0, raw_preds)
```

### Pitfall 4: Macro F1 on Unseen Classes

If test data contains classes not seen in training, your model cannot predict them correctly, and their F1 is 0 — dragging macro F1 down severely. Handle this during EDA: check for unseen classes in the test label distribution.

### Pitfall 5: Log Loss with Extreme Probabilities

Log loss becomes very large when the predicted probability for the true class is close to 0. Clip probabilities to avoid numerical issues:
```python
eps = 1e-15
preds_clipped = np.clip(preds, eps, 1 - eps)
```

---

## 10. The Metric-First Workflow

Always begin your competition preparation with this three-step check:

**Step 1**: Write out the metric formula by hand.

**Step 2**: Code it from scratch in numpy (do not just import from sklearn — you need to understand it).

**Step 3**: Verify your implementation against sklearn's implementation on a synthetic example.

```python
import numpy as np
from sklearn.metrics import f1_score

# Step 1-2: Implement F1 from scratch (binary case)
def f1_from_scratch(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1

# Step 3: Verify
y_true = np.array([1, 1, 0, 1, 0, 1, 0, 0])
y_pred = np.array([1, 0, 1, 1, 0, 1, 0, 1])

my_f1     = f1_from_scratch(y_true, y_pred)
sklearn_f1 = f1_score(y_true, y_pred)
print(f'My F1: {my_f1:.4f}, sklearn F1: {sklearn_f1:.4f}')
assert abs(my_f1 - sklearn_f1) < 1e-9, "Implementation mismatch!"
```

If you can implement a metric from its formula, you truly understand it.

---

## 11. Summary

The five most important metric insights for IOAI:

1. **Read the metric before reading anything else in the problem.**

2. **Accuracy is almost never the right metric for serious competitions** — it hides class imbalance.

3. **AUC measures ranking quality; log loss measures calibration quality. They are not the same.**

4. **"F1" without specifying macro/micro/weighted is ambiguous — always confirm.**

5. **When in doubt, implement the metric from scratch and verify against sklearn.** This process reveals your misunderstandings before they cost you points.

---

*Next: Lesson 5 — IOAI 2024 Problems and Analysis*
