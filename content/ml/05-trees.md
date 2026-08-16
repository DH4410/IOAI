---
title: Decision Trees
track: ml
order: 5
estimatedTime: 35
difficulty: intermediate
---

# Decision Trees

A decision tree makes predictions by asking a series of yes/no questions about the features. They are simple to understand, require no feature scaling, and handle both numbers and categories.

---

## 1. How It Works

Imagine predicting whether someone buys a product:

```
Is age > 30?
  YES -> Is income > 50k?
           YES -> Buys (80% chance)
           NO  -> Does not buy (70% chance)
  NO  -> Is student?
           YES -> Buys (65% chance)
           NO  -> Does not buy (75% chance)
```

Each split divides the data into groups. The model picks the split that makes the groups as pure (one class) as possible.

---

## 2. How Splits Are Chosen

At each node, the tree tries every feature and every possible threshold, and picks the split that reduces impurity the most.

**Gini impurity** measures how mixed a group is:
```
Gini = 1 - sum(p_i^2)
```

Where `p_i` is the fraction of class i in the group. If a group is all one class, Gini = 0 (pure). If it is 50/50, Gini = 0.5 (maximally impure).

**Information Gain** is another option - uses entropy instead of Gini. In practice both give similar trees.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# criterion='gini' or 'entropy' - gini is faster, results are similar
tree = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
tree.fit(X_train, y_train)
print(f'Test accuracy: {tree.score(X_test, y_test):.2%}')
```

**Quick check:** After training, the tree asks only 3 questions before giving a prediction. What does `max_depth=3` mean?
> The tree can ask at most 3 questions (3 levels of splits) on the path from root to any prediction.

---

## 3. Overfitting

A deep tree will memorize the training data perfectly but fail on new data. This is overfitting.

```python
# Deep tree - memorizes training data
deep_tree = DecisionTreeClassifier(max_depth=None)  # no limit
deep_tree.fit(X_train, y_train)
print(f'Train: {deep_tree.score(X_train, y_train):.2%}')   # likely 100%
print(f'Test:  {deep_tree.score(X_test, y_test):.2%}')     # much lower

# Controlled tree
shallow_tree = DecisionTreeClassifier(max_depth=3)
shallow_tree.fit(X_train, y_train)
print(f'Train: {shallow_tree.score(X_train, y_train):.2%}')
print(f'Test:  {shallow_tree.score(X_test, y_test):.2%}')
```

Key hyperparameters to control depth:
- `max_depth`: maximum number of splits on any path
- `min_samples_split`: a node must have at least N samples to be split
- `min_samples_leaf`: a leaf must have at least N samples

---

## 4. Feature Importance

After training, the tree tells you which features it used most:

```python
import numpy as np
import matplotlib.pyplot as plt

importances = tree.feature_importances_
feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']

idx = np.argsort(importances)[::-1]
plt.bar(range(len(importances)), importances[idx])
plt.xticks(range(len(importances)), [feature_names[i] for i in idx], rotation=45)
plt.title('Feature Importance')
plt.tight_layout()
plt.show()
```

Feature importance sums to 1. A feature with importance 0.7 means 70% of the improvement in purity came from splits on that feature.

---

## 5. Regression Trees

Decision trees also work for predicting numbers (regression). Instead of class labels at each leaf, the tree stores the mean of the target values in that region.

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

reg_tree = DecisionTreeRegressor(max_depth=4, random_state=42)
reg_tree.fit(X_train, y_train)
preds = reg_tree.predict(X_test)
print(f'MSE: {mean_squared_error(y_test, preds):.4f}')
```

The loss used for splitting in regression trees is MSE (mean squared error), not Gini impurity.

---

## 6. Visualizing the Tree

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 8))
plot_tree(
    tree,
    feature_names=['sepal length', 'sepal width', 'petal length', 'petal width'],
    class_names=['setosa', 'versicolor', 'virginica'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.show()
```

Each box shows: the split condition, the Gini impurity, the number of samples, and the dominant class.

**Quick check:** You visualize the tree and see a node with `gini=0.0` and `samples=30`. What does this mean?
> All 30 samples at this node belong to the same class (pure node, no mixing). This is a leaf node.

---

## Summary

| Concept | Key point |
|---|---|
| Split criterion | Gini or entropy - both measure how mixed a node is |
| Overfitting | Deep trees memorize; limit with `max_depth` |
| Feature importance | `.feature_importances_` shows which features the tree used most |
| Regression | `DecisionTreeRegressor` splits on MSE instead of Gini |
| Visualization | `plot_tree` lets you read every decision |

On their own, decision trees are rarely the best model. Their real power comes when you combine hundreds of them into a Random Forest or a Gradient Boosted ensemble - covered next.
