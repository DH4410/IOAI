---
title: Competition Strategy
track: ioai
order: 2
estimatedTime: 45
difficulty: advanced
---

# Competition Strategy

The difference between a student who scores in the top 10% and one who scores in the top 1% at IOAI is rarely about raw knowledge. Both students know gradient descent. Both know how to implement a neural network. The difference is **strategy**: how they allocate their time, how they read problems, how they decide what to do next, and how they manage the pressure of competition.

This lesson is about the meta-game. It is about how to compete, not just what to know.

---

## 1. The IOAI Mindset

Before any specific tactics, you need the right mental frame. Competition AI is different from research AI and from production AI. Here is how:

### Speed + Accuracy + Risk Management

In research, you explore freely and accuracy over time is what matters. In production, reliability and maintainability dominate. In competition, you face a brutal three-way tradeoff:

- **Speed**: The clock is always running. Every minute spent on one approach is a minute not spent on another.
- **Accuracy**: Wrong predictions do not score. Buggy code does not score.
- **Risk management**: A "safe" approach that scores 0.82 is often better than a "risky" approach that might score 0.89 or might crash with 30 minutes left.

The optimal competition strategy is not to maximize expected score. It is to maximize the **probability of finishing with a good score** — which means minimizing the risk of catastrophic failure (crashing, data leakage, time running out) while steadily improving a working solution.

### The "Good Enough Fast" Philosophy

This may feel counterintuitive: in competition, "perfect" is often the enemy of "good." Consider two students:

**Student A** spends 3 hours building an elaborate feature engineering pipeline with 50 features, careful hyperparameter tuning, and a stacking ensemble. Due to a bug in their ensemble combiner, they have no valid submission with 20 minutes left. They scramble and submit a buggy file. Score: catastrophic.

**Student B** spends 1 hour building a simple LightGBM model with basic preprocessing. It scores 0.78. They spend the remaining 4 hours improving it incrementally, each improvement tested end-to-end. Final score: 0.86.

Student B wins, not because they knew more, but because they executed better.

The philosophy: **get a working submission as fast as possible, then improve it iteratively**. Never find yourself with no valid submission and time running out.

### Growth Mindset Under Pressure

Competition pressure is real and it affects performance. The students who perform best:
- Have practiced enough that the basic techniques are automatic (no mental load)
- Treat setbacks (bugs, poor scores) as information, not failure
- Stay calm when the leaderboard shows them in 15th place after 2 hours (there are 6 hours left)
- Do not compare themselves to others in real-time — they focus on their own execution

---

## 2. Reading the Problem Statement

This is the single most important skill that students neglect. Before writing a single line of code, you must fully understand the problem. Here is a systematic approach.

### The Five Questions

When you receive the problem statement, answer these five questions before touching your keyboard:

1. **What is the prediction target?**
   - Is it a single number (regression)?
   - A class label (classification)?
   - Multiple labels (multi-label)?
   - A probability (probabilistic output)?

2. **What is the evaluation metric?**
   - Read this extremely carefully. The metric determines everything.
   - Is it accuracy, F1, AUC, RMSE, log loss, NDCG?
   - Is it macro, micro, or weighted for F1?
   - What is the formula for the metric?

3. **What is the baseline?**
   - Is there a provided baseline score?
   - What would the "naive" model score (predict the majority class always, predict the mean always)?
   - What score separates "passing" from "competitive"?

4. **What are the constraints?**
   - Time limit
   - Submission count limit
   - Memory limit (important for large datasets)
   - Any restrictions on approach (e.g., "use only provided features")

5. **What does the data look like?**
   - How many rows? How many columns?
   - What types of features (numerical, categorical, text, image)?
   - Is there a train/test split? What is the split ratio?
   - Is there additional unlabeled data you can use?

### Annotate as You Read

Print the problem statement (mentally or on paper) and mark:
- The metric (circle it)
- The target variable name (underline it)
- The submission format (star it)
- Any unusual rules (flag them)

Students who read once and code immediately often misunderstand the submission format or the metric, costing them significant time when they discover the error.

### The Metric Above All

The evaluation metric deserves special emphasis. Different metrics reward different model behaviors:

- **AUC-ROC**: Rewards good ranking, not calibration. A model that ranks examples correctly but outputs wrong probabilities still gets a high AUC.
- **Log loss**: Rewards well-calibrated probabilities. A model that outputs 0.5 for everything gets a terrible log loss even if it classifies correctly.
- **F1 (macro)**: Treats all classes equally regardless of size. A model that ignores a small class is heavily penalized.
- **RMSE**: Large errors are penalized much more than small errors (squared). Outliers matter a lot.
- **MAE**: All errors are penalized proportionally. Outliers matter less.

If you optimize for the wrong metric, you lose. Period.

---

## 3. The Competition Workflow

Every competitive data scientist has a workflow — a sequence of steps they follow in every competition. Having a practiced workflow means you never waste time wondering "what should I do next?" You always have an answer.

Here is the IOAI-optimized workflow:

### Phase 1: EDA (15% of time — 45 min in a 5-hour exam)

```python
# Step 1: Load data
import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

# Step 2: Basic shape and dtypes
print(train.shape, test.shape)
print(train.dtypes.value_counts())
print(train.head())

# Step 3: Target analysis
print(train['target'].describe())
print(train['target'].value_counts(normalize=True))  # For classification

# Step 4: Missing values
print(train.isnull().sum().sort_values(ascending=False).head(20))

# Step 5: Numerical feature distributions
train.describe()

# Step 6: Categorical feature cardinality
cat_cols = train.select_dtypes('object').columns
for col in cat_cols:
    print(f"{col}: {train[col].nunique()} unique values")
```

What you are looking for in EDA:
- **Target imbalance**: If one class is 95% of the data, accuracy is misleading. You need to handle imbalance.
- **Missing values**: Where are they, how many, and can they be informative (missingness as a feature)?
- **High-cardinality categoricals**: Text columns with many unique values need special handling.
- **Feature correlations**: Highly correlated features can be redundant.
- **Distribution shape**: Is the target skewed? (For regression, log-transform if yes.)
- **Data leakage red flags**: Are any features suspiciously correlated with the target?

**Time rule**: Spend no more than 45 minutes on EDA. You are not trying to understand everything — you are trying to understand enough to build a first baseline. You can always return to EDA during the iteration phase.

### Phase 2: Baseline (20% of time — 60 min)

The goal of the baseline phase is to get **something that works end-to-end** and submits a valid file. Do not tune anything. Do not engineer features. Just make something work.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score

# Quick preprocessing
num_cols = train.select_dtypes(include='number').columns.tolist()
num_cols.remove('target')

# For this example: numerical only baseline
X = train[num_cols].fillna(train[num_cols].median())
y = train['target']
X_test = test[num_cols].fillna(train[num_cols].median())

# Baseline model
model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
cv_score = cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()
print(f'Baseline CV AUC: {cv_score:.4f}')

# Train on full train set and predict
model.fit(X, y)
preds = model.predict_proba(X_test)[:, 1]

# Create submission
submission['target'] = preds
submission.to_csv('submission_baseline.csv', index=False)
```

**This is your safety net**. If everything else fails, this submission is in the bag.

### Phase 3: Iteration (45% of time — 135 min)

This is where you improve incrementally. The key principle: **make one change at a time and measure its effect on CV score before moving on**.

Iteration order (roughly by expected impact, highest first):

1. **Handle missing values properly** (instead of filling with median, try model-based imputation or creating a "was_missing" indicator)
2. **Encode categorical features** (OrdinalEncoder, TargetEncoder, or get_dummies for low-cardinality)
3. **Feature engineering** (interaction terms, polynomial features, domain-specific features)
4. **Model choice** (upgrade from baseline model if needed: LightGBM → XGBoost → CatBoost → ensemble)
5. **Hyperparameter tuning** (if you have time, use Optuna or simple grid search on key parameters)
6. **More features** (if the feature engineering is working, do more of it)

**Record every experiment**:

```python
# Keep a simple log
experiments = []

def log_experiment(name, cv_score, notes=''):
    experiments.append({'name': name, 'cv': cv_score, 'notes': notes})
    print(f'{name}: {cv_score:.4f} ({notes})')

# After each experiment:
log_experiment('baseline_lgbm', 0.821, 'numerical only, median imputation')
log_experiment('add_categoricals', 0.834, 'ordinal encoded categoricals')
log_experiment('add_missing_indicators', 0.839, 'binary flags for missing values')
```

This log is priceless at the end when you need to choose which submission to use.

### Phase 4: Ensemble (15% of time — 45 min)

If your individual models are good and you have time, ensemble them. The two simplest ensembling approaches:

**Averaging**: Average the probability predictions from different models.
```python
final_preds = (preds_lgbm + preds_xgb + preds_catboost) / 3
```

**Rank averaging**: Average the ranks instead of raw scores (more robust to scale differences).
```python
from scipy.stats import rankdata
rank1 = rankdata(preds_lgbm) / len(preds_lgbm)
rank2 = rankdata(preds_xgb) / len(preds_xgb)
final_preds = (rank1 + rank2) / 2
```

**When NOT to ensemble**:
- If your models are all very similar (trained on the same features with similar hyperparameters)
- If you are running out of time
- If your CV scores are inconsistent (high variance), ensemble may amplify instability

### Phase 5: Final Submission (5% of time — 15 min)

This phase is about decision-making, not coding.

- Review your experiment log
- Choose the submission with the best **cross-validation** score (not necessarily the best public leaderboard score — more on this below)
- Verify the submission file format matches `sample_submission.csv` exactly
- Submit with a few minutes to spare
- Do NOT make last-minute changes to code you haven't tested

---

## 4. Managing the Leaderboard

The public leaderboard in IOAI practical exams typically shows scores on a **subset** of the test data (usually 20–30%). The final ranking is determined by your score on the **private** (hidden) subset (the remaining 70–80%).

### The Overfitting-to-Public-LB Trap

This is one of the most common mistakes in Kaggle-style competitions, and it applies to IOAI:

You submit 5 times. The 3rd submission scores highest on the public leaderboard (0.862 vs. 0.855 for submission 5). You choose submission 3. But submission 3 was slightly overfit to the public split. On the private set (what actually counts for medals), submission 5 wins.

**Rule**: Unless you have a specific reason to trust the public leaderboard over your CV score, **always choose your best cross-validation submission as your final selection**, not your best public leaderboard submission.

### When the Leaderboard IS Trustworthy

The public leaderboard becomes more trustworthy when:
- The test set is large (thousands of samples, not hundreds)
- The public/private split is random (not temporal or structural)
- Your CV setup is flawed or has high variance

### Submission Strategy

- Use all available submission slots, but not carelessly
- Submit early and often during the iteration phase to build a history
- Save your last 1–2 submissions for carefully-chosen final models
- Never submit experimental code as your final submission without verifying it locally

---

## 5. Kaggle Strategies That Transfer to IOAI

The professional Kaggle community has developed a rich body of strategic knowledge over the years. Here is what transfers directly:

### Feature Engineering Is King (for tabular data)

In almost every tabular competition, the highest-scoring solution is not the most complex model — it is the one with the most creative and informative features. Before reaching for a neural network, ask: have I extracted all the information from the features I already have?

Common feature engineering ideas:
- **Interaction features**: `feature_A * feature_B`, `feature_A / feature_B`
- **Aggregations**: Group by a categorical column and compute mean, std, min, max, count of a numerical column
- **Date/time features**: Day of week, month, hour, is_weekend, days_since_event
- **Text features**: Character count, word count, presence of specific keywords, TF-IDF
- **Rank features**: Rank within a group

### Stacking / Blending

For the most dedicated competitors, stacking (training a meta-model on out-of-fold predictions) can provide meaningful gains. However, in a time-constrained IOAI exam, simple averaging is almost always the better choice. Reserve stacking for when you have a strong set of diverse base models and at least 2 hours remaining.

### Pseudo-Labeling

Pseudo-labeling: train a model on labeled data, predict labels for the test set, add the high-confidence test predictions to your training data, retrain. This can be powerful but introduces risk (if your initial model is wrong, pseudo-labels propagate errors).

**In IOAI**: Use pseudo-labeling only if you understand it well and have time. Do not implement it under pressure if you have never done it before.

### The Diversity Principle

When ensembling, diversity is more important than individual quality. Two models with CV scores of 0.83 that make different errors will ensemble better than two models with CV 0.85 that make the same errors.

Create diversity by:
- Using different algorithms (LightGBM + XGBoost + CatBoost)
- Using different feature sets
- Using different random seeds
- Training on different subsets (bagging)

---

## 6. Building Reliable Code Fast

In a timed competition, code reliability matters as much as algorithm quality. A bug in your code is a bug in your score.

### The Reproducibility First Principle

At the top of every competition notebook, before anything else:

```python
import numpy as np
import random
import os

SEED = 42

def set_seeds(seed=SEED):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

set_seeds()
```

This ensures that if you run your notebook again, you get identical results.

### Utility Functions to Have Ready

Memorize or have ready-to-paste versions of these utilities:

```python
# Cross-validation scorer (classification)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def cv_score(model, X, y, n_splits=5, metric=roc_auc_score, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_val)[:, 1]
        scores.append(metric(y_val, preds))
        print(f'  Fold {fold+1}: {scores[-1]:.4f}')
    print(f'  Mean: {np.mean(scores):.4f} ± {np.std(scores):.4f}')
    return np.mean(scores)

# Submission creator
def make_submission(test_ids, predictions, filename, id_col='id', target_col='target'):
    sub = pd.DataFrame({id_col: test_ids, target_col: predictions})
    sub.to_csv(filename, index=False)
    print(f'Saved {filename} with {len(sub)} rows')
    return sub
```

### Defensive Coding Habits

- **Assert shapes**: `assert X_train.shape[0] == y_train.shape[0]`
- **Check for NaN before model**: `assert not X_train.isnull().any().any(), "NaN in X_train!"`
- **Verify submission format**: Check that submission columns match `sample_submission.csv` exactly
- **Test on a small sample first**: Before running a slow model on all data, test it on 1000 rows

### When to Use Jupyter vs. Python Scripts

In IOAI competitions, Jupyter notebooks are standard. Use a single notebook with numbered cells in order. Avoid having important code buried in out-of-order cells that you run selectively — this creates reproducibility issues.

Structure your notebook with clear markdown headers:
```
# 0. Setup & Seeds
# 1. Data Loading
# 2. EDA
# 3. Preprocessing
# 4. Baseline Model
# 5. Feature Engineering v1
# 6. Improved Model
# 7. Ensemble
# 8. Final Submission
```

---

## 7. Simple vs. Complex Models

One of the most important strategic decisions in IOAI is knowing when to use a simple model and when to reach for something more powerful.

### The Complexity Ladder

**Logistic Regression / Ridge**:
- Use when: few features, linear relationships, fast iteration needed, interpretability required
- Train time: seconds
- Baseline quality: often surprisingly good
- Typical CV AUC: 0.70–0.80

**LightGBM / XGBoost / CatBoost**:
- Use when: tabular data, mixed feature types, need strong performance
- Train time: seconds to minutes
- Competition-grade quality: yes
- Typical CV AUC: 0.80–0.90 (domain-dependent)

**Neural networks (tabular)**:
- Use when: very large datasets (100k+ rows), embeddings needed, interactions are very complex
- Train time: minutes to hours
- Often beaten by gradient boosting on tabular data
- Use only if you have deep learning experience and time

**Transformer / CNN / etc.**:
- Use when: image, text, or sequence data is involved
- Essential for those modalities
- Do not use for tabular data unless specifically warranted

### The Rule of Thumb

For tabular data in a time-constrained competition:
1. Start with LightGBM (fastest to strong performance)
2. Add XGBoost or CatBoost for ensemble diversity
3. Only add neural networks if gradient boosting has plateaued and you have time

For image data:
1. Start with a pretrained ResNet/EfficientNet (if weights are provided)
2. Fine-tune with standard augmentation
3. Ensemble different architectures if time permits

For text data:
1. Start with TF-IDF + LightGBM (fast, strong baseline)
2. Add pretrained BERT fine-tuning if time permits

---

## 8. Common Mistakes That Cost Points

### In the Practical Exam

**1. Data Leakage**
Using information from the test set (or the future, in time-series tasks) to train the model. Classic leakage sources:
- Computing statistics (mean, std) on the entire dataset (train + test combined) and using them to scale features
- Correct fix: compute statistics only on the training fold, apply to validation fold

**2. Wrong CV Setup**
Using random shuffled k-fold on time-series data (where chronological order matters). This creates artificially high CV scores that do not generalize.
- Fix: use time-series cross-validation (GroupKFold, TimeSeriesSplit)

**3. Not Checking Submission Format**
Your submission has columns in the wrong order, wrong column names, or wrong number of rows.
- Fix: `assert submission.shape == sample_submission.shape`

**4. Overfitting to Training Set**
Adding 100 features without regularization and getting 0.99 train accuracy, 0.71 validation AUC.
- Fix: always evaluate on held-out data, never on training data

**5. Ignoring Class Imbalance**
With 95% negative class, a model that always predicts negative gets 95% accuracy but 0 F1 on the positive class.
- Fix: use `class_weight='balanced'`, oversampling, or a metric that penalizes class imbalance

### In the Theoretical Exam

**1. Confusing Similar Concepts**
- Precision vs. Recall: Precision = TP/(TP+FP) — of what you predicted positive, how many were correct. Recall = TP/(TP+FN) — of all actual positives, how many did you catch.
- L1 vs. L2 regularization: L1 produces sparse weights; L2 produces small weights.
- Variance vs. Standard Deviation: Variance is the squared standard deviation.

**2. Not Showing Work**
On short-answer questions, always show your calculation steps. Partial credit is real.

**3. Misreading Question Modifiers**
"Which of the following is NOT true?" — students often forget the NOT and answer the opposite of what's asked.

---

## 9. Mental Preparation

### The Night Before

- Stop studying at 8 PM
- Eat a good dinner
- Sleep 8 hours minimum
- Lay out everything you need (ID, materials)
- No last-minute cramming — it raises anxiety without adding knowledge

### During the Exam

- Read the entire problem statement before coding
- If you feel panic, take three slow breaths and return to your workflow
- If you are stuck on a theory question, skip it and come back
- Keep an eye on the clock: at the halfway point, you should have a working baseline submitted

### Dealing With Setbacks

You will hit bugs. Your model will not improve as fast as you hoped. A competitor will post a higher score on the leaderboard. These are normal. The response to every setback is the same: return to your workflow, make one change, measure it, move forward.

The students who fall apart under pressure do so because they start improvising rather than following their workflow. Your workflow is your anchor.

---

## 10. Team Collaboration Strategy

IOAI assigns individual scores, but your team shares a coach, practice sessions, and morale. Here is how to make teamwork work:

### Before the Competition

- **Divide specializations**: One team member becomes the go-to for CV/image tasks, another for NLP, another for tabular data, another for theory.
- **Peer review practice submissions**: Review each other's code during practice. This builds the habit of clean, readable code.
- **Share utilities**: Build a shared library of utility functions that all team members know.
- **Practice under pressure**: Simulate the full competition environment at least twice — same time limits, no external resources.

### During the Competition

Individual exams mean you each work independently. However:
- Brief (2-minute) status checks between team members during breaks can reveal if someone is going in a completely wrong direction
- Share morale — acknowledge when someone has a good idea or overcomes a problem

### After the Competition

Regardless of results, debrief:
- What did each person do well?
- What would you do differently?
- What did you learn that you did not know before?

The teams that improve most across competitions are the ones that debrief thoroughly.

---

## Which Phase is This?

```widget
{
  "type": "concept-sort",
  "title": "Match the Action to the Competition Phase",
  "categories": [
    { "name": "EDA (first 45 min)", "color": "#5B5BD6" },
    { "name": "Baseline (first 60 min)", "color": "#22C55E" },
    { "name": "Iteration (middle 135 min)", "color": "#F97316" },
    { "name": "Final 15 min", "color": "#EF4444" }
  ],
  "items": [
    { "text": "Check class imbalance and missing value counts", "category": "EDA (first 45 min)" },
    { "text": "Submit a working LightGBM model with default params", "category": "Baseline (first 60 min)" },
    { "text": "Add target-encoded categoricals and measure CV improvement", "category": "Iteration (middle 135 min)" },
    { "text": "Choose best CV submission and verify file format matches sample_submission.csv", "category": "Final 15 min" },
    { "text": "Check the target distribution and confirm evaluation metric", "category": "EDA (first 45 min)" },
    { "text": "Generate your first valid submission file — the safety net", "category": "Baseline (first 60 min)" },
    { "text": "Average LightGBM + XGBoost predictions for ensemble", "category": "Iteration (middle 135 min)" },
    { "text": "Do NOT make untested code changes with less than 15 minutes left", "category": "Final 15 min" }
  ]
}
```

---

## 11. Summary: The 10 Rules of IOAI Strategy

1. **Get a working submission before you optimize anything.**
2. **The metric is the only thing that matters — know it cold before you code.**
3. **Follow your workflow, especially when things go wrong.**
4. **Make one change at a time, measure it, move on.**
5. **CV score is more trustworthy than public leaderboard score.**
6. **Simple models that work beat complex models that crash.**
7. **Set seeds everywhere. Verify reproducibility. Check submission format.**
8. **Spend 15% of time on EDA, 20% on baseline, 45% on iteration.**
9. **Never make untested changes in the last 15 minutes.**
10. **Stay calm. Every competitor faces the same problem. Trust your preparation.**

---

*Next: Lesson 3 — Building a Strong Baseline*
