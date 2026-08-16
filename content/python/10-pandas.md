---
title: Pandas - Working with Data
track: python
order: 10
estimatedTime: 35
difficulty: intermediate
---

# Pandas - Working with Data

In IOAI competitions, your data comes as a CSV file. Pandas is the tool you use to load it, inspect it, clean it, and get it into a shape your model can use. This lesson covers what you actually need during a competition.

---

## 1. The Two Core Objects

Pandas has two objects you need to know:

- **Series** - a single column of data (1D)
- **DataFrame** - a table of data with rows and columns (2D)

```python
import pandas as pd
import numpy as np

# A Series is like a labeled list
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(s['b'])   # 20

# A DataFrame is like a spreadsheet
df = pd.DataFrame({
    'name':  ['Ada', 'Bob', 'Carol'],
    'score': [92, 78, 88],
    'grade': ['A', 'C', 'B']
})
print(df)
```

In practice, you almost never create DataFrames by hand. You load them from a file.

---

## 2. Loading and Inspecting Data

```python
df = pd.read_csv('train.csv')
```

The first thing to do after loading - always:

```python
print(df.shape)         # (rows, columns)
print(df.dtypes)        # data type of each column
print(df.head(5))       # first 5 rows
print(df.describe())    # count, mean, std, min, max for each number column
print(df.isnull().sum())  # how many missing values per column
```

**Quick check:** You load a CSV and `df.shape` is `(1000, 15)`. What does this mean?
> 1000 rows (data points) and 15 columns (features + possibly a target column).

`df.describe()` is your first sanity check. Unusual min/max values (like `-999` or `9999`) often signal missing values coded as numbers.

---

## 3. Selecting Data

```python
# One column (returns a Series)
df['score']

# Multiple columns (returns a DataFrame)
df[['name', 'score']]

# Rows by position (like Python list slicing)
df.iloc[0]       # first row
df.iloc[0:5]     # first 5 rows
df.iloc[10, 2]   # row 10, column 2

# Rows by label (index)
df.loc[0]        # row with index 0
df.loc[0, 'score']  # row 0, column 'score'
```

**Rule of thumb:** Use `.iloc` when you think in positions. Use `.loc` when you think in labels.

---

## 4. Filtering Rows

Pick rows that meet a condition:

```python
# Single condition
high = df[df['score'] > 85]

# Multiple conditions (use & for AND, | for OR, always add parentheses)
good = df[(df['score'] > 80) & (df['grade'] == 'A')]

# Check if value is in a list
subset = df[df['grade'].isin(['A', 'B'])]

# Drop rows that don't match
df_clean = df[df['age'] != 999]   # remove bad placeholder values
```

**Quick check:** You want all rows where `'label'` is either `0` or `2`. Write the filter.
> `df[df['label'].isin([0, 2])]`

---

## 5. Handling Missing Data

Missing values show up as `NaN` in Pandas.

```python
print(df.isnull().sum())   # count NaN per column

# Drop rows that have ANY missing value
df_clean = df.dropna()

# Drop rows only if a specific column is missing
df_clean = df.dropna(subset=['score'])

# Fill missing values with a fixed number
df['age'].fillna(df['age'].median(), inplace=True)

# Fill with the most common value (for categories)
df['city'].fillna(df['city'].mode()[0], inplace=True)
```

For ML competitions, filling with median is usually safe for numbers. Filling with mode is safe for categories. **Do not drop rows unless you have many to spare.**

---

## 6. Adding and Changing Columns

```python
# Add a new column
df['score_normalized'] = df['score'] / df['score'].max()

# Change values based on a condition
df.loc[df['score'] > 90, 'grade'] = 'A+'

# Apply a function to each value in a column
df['name_upper'] = df['name'].str.upper()

# Apply a custom function
df['bonus'] = df['score'].apply(lambda x: x * 1.1 if x > 80 else x)
```

---

## 7. GroupBy - Aggregating by Category

GroupBy splits the data into groups and lets you compute summaries:

```python
# Average score per grade
df.groupby('grade')['score'].mean()

# Multiple aggregations at once
df.groupby('grade').agg({'score': ['mean', 'max', 'count']})

# Group by multiple columns
df.groupby(['grade', 'city'])['score'].mean()
```

**Quick check:** You want the total sales per city. The column is `'sales'`, the category is `'city'`. Write it.
> `df.groupby('city')['sales'].sum()`

---

## 8. From Pandas to NumPy

Your ML model does not take a DataFrame directly. It takes a NumPy array:

```python
# Separate features from target
X = df.drop(columns=['target']).values   # .values gives a numpy array
y = df['target'].values

# Or select specific columns
X = df[['age', 'score', 'income']].values

print(X.shape)   # (n_samples, n_features)
```

If any column has non-numeric data (strings), convert it first:

```python
# Turn a category column into numbers
df['grade_code'] = pd.Categorical(df['grade']).codes

# One-hot encode (creates a separate 0/1 column per category)
df = pd.get_dummies(df, columns=['city'], dtype=int)
```

---

## Summary

| Task | Code |
|---|---|
| Load CSV | `pd.read_csv('file.csv')` |
| First look | `df.head()`, `df.describe()`, `df.isnull().sum()` |
| Select one column | `df['col']` |
| Select by position | `df.iloc[row, col]` |
| Filter rows | `df[df['col'] > 5]` |
| Multiple conditions | `df[(cond1) & (cond2)]` |
| Fill missing values | `df['col'].fillna(median)` |
| Add column | `df['new'] = ...` |
| Group and aggregate | `df.groupby('cat')['val'].mean()` |
| To NumPy | `df.values` or `df[cols].values` |
| One-hot encode | `pd.get_dummies(df, columns=['cat'])` |

Pandas is a tool for getting data ready. Once it is clean and numeric, hand it to NumPy or your ML model and let those do the heavy lifting.
