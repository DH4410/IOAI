---
title: Pandas — Working with Data
track: python
order: 10
estimatedTime: 50
difficulty: intermediate
---

# Pandas — Working with Data

In the real world, data rarely arrives as a clean NumPy array. It comes in CSV files, spreadsheets, SQL tables, and JSON files — messy, with missing values, inconsistent types, and columns that need cleaning before any model can touch them. Pandas is the library that handles all of this.

Pandas gives you two primary data structures — **Series** and **DataFrame** — and hundreds of methods for loading, inspecting, cleaning, filtering, grouping, and transforming your data. It is built on top of NumPy, so everything you learned in the previous lesson still applies.

---

## 1. Series vs DataFrame

### Series — A Labeled 1D Array

A Series is a one-dimensional array where each element has a **label** (called an index):

```python
import pandas as pd
import numpy as np

# Create a Series from a list
temps = pd.Series([22, 25, 19, 30, 27])
print(temps)
# 0    22
# 1    25
# 2    19
# 3    30
# 4    27
# dtype: int64

# Create a Series with a custom index
temps = pd.Series(
    [22, 25, 19, 30, 27],
    index=["Mon", "Tue", "Wed", "Thu", "Fri"]
)
print(temps)
# Mon    22
# Tue    25
# Wed    19
# Thu    30
# Fri    27
# dtype: int64

# Access by label
print(temps["Wed"])   # 19
print(temps[["Mon", "Thu"]])
# Mon    22
# Thu    30

# Series support vectorized operations like NumPy
print(temps * 1.8 + 32)   # Convert Celsius to Fahrenheit
# Mon    71.6
# Tue    77.0
# Wed    66.2
# Thu    86.0
# Fri    80.6

print(temps.mean())   # 24.6
print(temps.max())    # 30
```

### DataFrame — A Labeled 2D Table

A DataFrame is a two-dimensional table with labeled rows (index) and labeled columns. Think of it as a spreadsheet where every column is a Series:

```python
import pandas as pd

df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie", "Diana"],
    "age":  [25, 30, 22, 35],
    "score": [88.5, 92.0, 79.5, 95.0]
})

print(df)
#       name  age  score
# 0    Alice   25   88.5
# 1      Bob   30   92.0
# 2  Charlie   22   79.5
# 3    Diana   35   95.0

print(type(df["name"]))   # <class 'pandas.core.series.Series'>
print(df.columns)         # Index(['name', 'age', 'score'], dtype='object')
print(df.index)           # RangeIndex(start=0, stop=4, step=1)
```

> **Note:** Each column in a DataFrame is a Series. They all share the same index (the row labels, typically 0, 1, 2, …).

---

## 2. Creating DataFrames

### From a Dictionary of Lists

```python
import pandas as pd

# Keys are column names, values are lists of equal length
data = {
    "city":       ["New York", "London", "Tokyo", "Sydney"],
    "country":    ["USA",      "UK",     "Japan", "Australia"],
    "population": [8_336_817, 8_982_000, 13_960_000, 5_312_000],
    "lat":        [40.71,     51.51,     35.68,      -33.87]
}
df = pd.DataFrame(data)
print(df)
#        city    country  population    lat
# 0  New York        USA   8336817  40.71
# 1    London         UK   8982000  51.51
# 2     Tokyo      Japan  13960000  35.68
# 3    Sydney  Australia   5312000 -33.87
```

### From a List of Dictionaries

Each dictionary represents one row:

```python
import pandas as pd

rows = [
    {"name": "Alice", "score": 88, "passed": True},
    {"name": "Bob",   "score": 55, "passed": False},
    {"name": "Carol", "score": 92, "passed": True},
]
df = pd.DataFrame(rows)
print(df)
#     name  score  passed
# 0  Alice     88    True
# 1    Bob     55   False
# 2  Carol     92    True
```

### From a NumPy Array

```python
import pandas as pd
import numpy as np

np.random.seed(0)
array = np.random.randint(0, 100, size=(4, 3))

df = pd.DataFrame(
    array,
    columns=["feature_1", "feature_2", "feature_3"],
    index=["sample_a", "sample_b", "sample_c", "sample_d"]
)
print(df)
#           feature_1  feature_2  feature_3
# sample_a         44         47         64
# sample_b         67         67          9
# sample_c         83         21         36
# sample_d         87         70         88
```

---

## 3. Loading Data: `pd.read_csv()`

In the real world, you almost always load data from a file. CSV (Comma-Separated Values) is the most common format.

### What a CSV Looks Like

A CSV file is a plain text file where:
- The first line is usually the header (column names)
- Each subsequent line is one row
- Values are separated by commas (or sometimes semicolons or tabs)

```
name,age,salary,department
Alice,28,65000,Engineering
Bob,34,72000,Marketing
Carol,29,58000,Design
David,45,95000,Engineering
```

### Loading a CSV

```python
import pandas as pd

# Load from a file
df = pd.read_csv("employees.csv")

# From a URL (Pandas can read directly from the web)
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# Common options:
df = pd.read_csv(
    "data.csv",
    sep=",",           # separator character (default is comma)
    header=0,          # row number of column names (0 = first row)
    index_col=0,       # use first column as index
    usecols=["name", "age"],  # only load these columns
    nrows=100,         # only load first 100 rows
    na_values=["N/A", "-", "?"]  # treat these as NaN (missing)
)

# Other formats
df = pd.read_excel("data.xlsx")         # Excel
df = pd.read_json("data.json")          # JSON
df = pd.read_sql("SELECT * FROM t", connection)  # SQL database
```

> **Tip:** For the IOAI competition, most datasets will be in CSV format. Always check `df.head()` and `df.info()` right after loading to understand what you have.

---

## 4. Viewing and Inspecting Data

After loading data, the first thing you do is inspect it:

```python
import pandas as pd

# Create a sample DataFrame for demonstration
df = pd.DataFrame({
    "name":    ["Alice", "Bob", "Carol", "David", "Eve", "Frank"],
    "age":     [25, 30, None, 28, 35, 22],
    "salary":  [65000, 72000, 58000, 95000, None, 48000],
    "dept":    ["Eng", "Mkt", "Eng", "Eng", "Mkt", "Design"],
    "rating":  [4.5, 3.8, 4.2, 4.9, 3.5, 4.0]
})

# First N rows (default 5)
print(df.head())
print(df.head(3))   # first 3 rows

# Last N rows
print(df.tail(2))

# Shape: (rows, columns)
print(df.shape)   # (6, 5)

# Column names and dtypes
print(df.dtypes)
# name       object
# age       float64
# salary    float64
# dept       object
# rating    float64

# Concise summary: non-null count, dtype per column
df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 6 entries, 0 to 5
# Data columns (total 5 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   name    6 non-null      object
#  1   age     5 non-null      float64
#  2   salary  5 non-null      float64
#  3   dept    6 non-null      object
#  4   rating  6 non-null      float64

# Statistical summary of numeric columns
print(df.describe())
#              age         salary    rating
# count   5.000000       5.000000  6.000000
# mean   28.000000   67600.000000  4.150000
# std     4.743416   17730.460898  0.498997
# min    22.000000   48000.000000  3.500000
# 25%    25.000000   58000.000000  3.850000
# 50%    28.000000   65000.000000  4.100000
# 75%    30.000000   72000.000000  4.425000
# max    35.000000   95000.000000  4.900000
```

---

## 5. Selecting Data

### Selecting Columns

```python
import pandas as pd

df = pd.DataFrame({
    "name":   ["Alice", "Bob", "Carol"],
    "age":    [25, 30, 22],
    "salary": [65000, 72000, 58000],
    "dept":   ["Eng", "Mkt", "Eng"]
})

# Single column → returns a Series
print(df["name"])
# 0    Alice
# 1      Bob
# 2    Carol

# Multiple columns → returns a DataFrame
print(df[["name", "salary"]])
#     name  salary
# 0  Alice   65000
# 1    Bob   72000
# 2  Carol   58000
```

### `loc[]` — Label-Based Selection

`loc` selects by row label and column name:

```python
import pandas as pd

df = pd.DataFrame({
    "name":   ["Alice", "Bob", "Carol", "David"],
    "age":    [25, 30, 22, 28],
    "salary": [65000, 72000, 58000, 95000]
}, index=["a", "b", "c", "d"])

# Single row by label
print(df.loc["b"])
# name      Bob
# age        30
# salary  72000

# Multiple rows
print(df.loc[["a", "c"]])
#     name  age  salary
# a  Alice   25   65000
# c  Carol   22   58000

# Rows and specific columns
print(df.loc["a":"c", "age":"salary"])
#    age  salary
# a   25   65000
# b   30   72000
# c   22   58000
# Note: loc slicing is INCLUSIVE on both ends

# All rows, specific column
print(df.loc[:, "age"])
# a    25
# b    30
# c    22
# d    28
```

### `iloc[]` — Integer Position-Based Selection

`iloc` uses integer positions (like NumPy):

```python
import pandas as pd

df = pd.DataFrame({
    "name":   ["Alice", "Bob", "Carol", "David"],
    "age":    [25, 30, 22, 28],
    "salary": [65000, 72000, 58000, 95000]
})

# Single row by position
print(df.iloc[0])
# name      Alice
# age          25
# salary    65000

# Slice rows and columns (exclusive end, like Python lists)
print(df.iloc[1:3, 0:2])
#     name  age
# 1    Bob   30
# 2  Carol   22

# Last row
print(df.iloc[-1])

# Every other row
print(df.iloc[::2])
#     name  age  salary
# 0  Alice   25   65000
# 2  Carol   22   58000
```

> **When to use loc vs iloc:**
> - `loc`: when you know the label (column name or row name)
> - `iloc`: when you know the position (row 0, column 2, etc.)

---

## 6. Filtering Data

### Boolean Indexing

```python
import pandas as pd

df = pd.DataFrame({
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "dept":   ["Eng",   "Mkt", "Eng",   "Eng",   "Mkt"],
    "salary": [65000, 72000, 58000, 95000, 48000],
    "rating": [4.5, 3.8, 4.2, 4.9, 3.5]
})

# Filter rows where salary > 60000
high_earners = df[df["salary"] > 60000]
print(high_earners)
#     name dept  salary  rating
# 0  Alice  Eng   65000     4.5
# 1    Bob  Mkt   72000     3.8
# 3  David  Eng   95000     4.9

# Filter by string equality
eng_team = df[df["dept"] == "Eng"]
print(eng_team)

# Multiple conditions (use & for AND, | for OR)
# IMPORTANT: wrap each condition in parentheses!
top_eng = df[(df["dept"] == "Eng") & (df["rating"] >= 4.0)]
print(top_eng)
#     name dept  salary  rating
# 0  Alice  Eng   65000     4.5
# 2  Carol  Eng   58000     4.2
# 3  David  Eng   95000     4.9

# OR condition
mkt_or_high_pay = df[(df["dept"] == "Mkt") | (df["salary"] > 90000)]
print(mkt_or_high_pay)
```

> **Common mistake:** Do NOT use Python's `and`/`or` keywords with Pandas conditions. Use `&`/`|` with parentheses instead.

### `query()` — Readable Filtering

```python
import pandas as pd

df = pd.DataFrame({
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "dept":   ["Eng",   "Mkt", "Eng",   "Eng",   "Mkt"],
    "salary": [65000, 72000, 58000, 95000, 48000],
    "rating": [4.5, 3.8, 4.2, 4.9, 3.5]
})

# query() uses a string expression — more readable for complex filters
result = df.query("dept == 'Eng' and salary > 60000")
print(result)

# Use @ to reference Python variables inside query
min_rating = 4.0
result = df.query("rating >= @min_rating")
print(result)
```

---

## 7. Missing Data

Missing data is everywhere in real datasets. Pandas represents missing values as `NaN` (Not a Number), inherited from NumPy.

### Detecting Missing Data

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "name":   ["Alice", "Bob", None, "David"],
    "age":    [25, np.nan, 22, 28],
    "salary": [65000, 72000, np.nan, np.nan]
})

# Check which values are NaN
print(df.isnull())
#     name    age  salary
# 0  False  False   False
# 1  False   True   False
# 2   True  False    True
# 3  False  False    True

# Count missing per column
print(df.isnull().sum())
# name      1
# age       1
# salary    2

# Percentage missing
print(df.isnull().mean() * 100)
# name      25.0
# age       25.0
# salary    50.0

# Opposite: notnull()
print(df["age"].notnull())
# 0     True
# 1    False
# 2     True
# 3     True
```

### Removing Missing Data

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "name":   ["Alice", "Bob", None, "David"],
    "age":    [25, np.nan, 22, 28],
    "salary": [65000, 72000, np.nan, np.nan]
})

# Drop rows with ANY missing value
print(df.dropna())
#     name   age   salary
# 0  Alice  25.0  65000.0

# Drop rows where ALL values are missing
print(df.dropna(how="all"))

# Drop rows with missing values only in specific columns
print(df.dropna(subset=["age"]))
#     name   age   salary
# 0  Alice  25.0  65000.0
# 2   None  22.0      NaN
# 3  David  28.0      NaN

# Drop columns with any missing value
print(df.dropna(axis=1))
# Empty — all columns have at least one NaN
```

### Filling Missing Data

In ML, dropping rows often wastes valuable data. Filling missing values (imputation) is usually better:

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "age":    [25, np.nan, 22, 28, np.nan],
    "salary": [65000, 72000, np.nan, np.nan, 48000],
    "dept":   ["Eng", "Mkt", "Eng", None, "Mkt"]
})

# Fill with a constant
df_filled = df.fillna(0)

# Fill with the column mean (common for numeric ML features)
df["age"] = df["age"].fillna(df["age"].mean())
df["salary"] = df["salary"].fillna(df["salary"].median())

# Fill with the most frequent value (mode) — good for categorical
df["dept"] = df["dept"].fillna(df["dept"].mode()[0])

# Forward fill: propagate the last valid value forward
df_ff = df.fillna(method="ffill")

# Backward fill: propagate the next valid value backward
df_bf = df.fillna(method="bfill")

print(df)
```

> **Why ML needs clean data:** Most ML algorithms (like scikit-learn's classifiers) will throw an error if they encounter NaN values. Always check for and handle missing data before fitting a model.

---

## 8. Adding and Modifying Columns

```python
import pandas as pd

df = pd.DataFrame({
    "name":   ["Alice", "Bob", "Carol"],
    "salary": [65000, 72000, 58000],
    "years":  [3, 7, 2]
})

# Add a new column
df["annual_bonus"] = df["salary"] * 0.1
print(df)
#     name  salary  years  annual_bonus
# 0  Alice   65000      3        6500.0
# 1    Bob   72000      7        7200.0
# 2  Carol   58000      2        5800.0

# Compute a derived column
df["salary_per_year"] = df["salary"] / df["years"]

# Categorical column based on condition
df["level"] = "junior"
df.loc[df["years"] >= 5, "level"] = "senior"
print(df["level"])
# 0    junior
# 1    senior
# 2    junior

# Modify an existing column
df["salary"] = df["salary"] * 1.05   # Give everyone a 5% raise

# Rename columns
df = df.rename(columns={"name": "employee_name", "years": "experience"})

# Delete a column
df = df.drop(columns=["annual_bonus"])
```

---

## 9. Sorting

```python
import pandas as pd

df = pd.DataFrame({
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "dept":   ["Eng",   "Mkt", "Eng",   "Eng",   "Mkt"],
    "salary": [65000, 72000, 58000, 95000, 48000],
    "rating": [4.5, 3.8, 4.2, 4.9, 3.5]
})

# Sort by one column (ascending)
print(df.sort_values("salary"))

# Sort by one column (descending)
print(df.sort_values("salary", ascending=False))

# Sort by multiple columns
# First by dept (A-Z), then by salary (high to low) within each dept
print(df.sort_values(["dept", "salary"], ascending=[True, False]))
#     name dept  salary  rating
# 3  David  Eng   95000     4.9
# 0  Alice  Eng   65000     4.5
# 2  Carol  Eng   58000     4.2
# 1    Bob  Mkt   72000     3.8
# 4    Eve  Mkt   48000     3.5

# Sort by index
df_sorted = df.sort_index()

# Sort and reset the index (common after filtering)
df_filtered = df[df["salary"] > 60000].sort_values("salary").reset_index(drop=True)
print(df_filtered)
#     name dept  salary  rating
# 0  Alice  Eng   65000     4.5
# 1    Bob  Mkt   72000     3.8
# 2  David  Eng   95000     4.9
```

---

## 10. GroupBy — Aggregating by Category

`groupby` is one of the most powerful features in Pandas. It lets you split your data into groups and apply aggregation functions to each group.

The pattern is always: **split → apply → combine**

```python
import pandas as pd

df = pd.DataFrame({
    "name":   ["Alice", "Bob", "Carol", "David", "Eve", "Frank"],
    "dept":   ["Eng",   "Mkt", "Eng",   "Eng",   "Mkt", "Design"],
    "salary": [65000, 72000, 58000, 95000, 48000, 52000],
    "rating": [4.5, 3.8, 4.2, 4.9, 3.5, 4.0]
})

# Average salary by department
print(df.groupby("dept")["salary"].mean())
# dept
# Design    52000.0
# Eng       72666.7
# Mkt       60000.0
# Name: salary, dtype: float64

# Count of employees per department
print(df.groupby("dept")["name"].count())
# dept
# Design    1
# Eng       3
# Mkt       2

# Multiple aggregations at once
print(df.groupby("dept")["salary"].agg(["mean", "min", "max", "count"]))
#               mean    min    max  count
# dept
# Design  52000.000  52000  52000      1
# Eng     72666.667  58000  95000      3
# Mkt     60000.000  48000  72000      2

# GroupBy multiple columns
print(df.groupby(["dept"])["rating"].mean().round(2))

# Custom aggregation with agg() on multiple columns
summary = df.groupby("dept").agg({
    "salary": ["mean", "max"],
    "rating": "mean",
    "name":   "count"
})
print(summary)
```

### Iterating Over Groups

```python
import pandas as pd

df = pd.DataFrame({
    "dept":   ["Eng", "Mkt", "Eng", "Mkt"],
    "salary": [65000, 72000, 58000, 48000]
})

for dept_name, group_df in df.groupby("dept"):
    print(f"Department: {dept_name}")
    print(group_df)
    print()
```

---

## 11. Apply — Applying Custom Functions

`apply()` runs a function on each row or each column:

```python
import pandas as pd

df = pd.DataFrame({
    "name":   ["alice johnson", "bob smith", "carol white"],
    "salary": [65000, 72000, 58000]
})

# Apply a lambda to a column (Series)
df["name"] = df["name"].apply(lambda s: s.title())
print(df["name"])
# 0    Alice Johnson
# 1      Bob Smith
# 2    Carol White

# Apply to multiple columns using axis=1 (row by row)
def classify_salary(row):
    if row["salary"] >= 70000:
        return "high"
    elif row["salary"] >= 60000:
        return "medium"
    else:
        return "low"

df["salary_tier"] = df.apply(classify_salary, axis=1)
print(df)
#            name  salary salary_tier
# 0  Alice Johnson   65000      medium
# 1     Bob Smith   72000        high
# 2   Carol White   58000         low

# Map: transform a column using a dictionary
df["salary_tier_code"] = df["salary_tier"].map({"low": 0, "medium": 1, "high": 2})
```

---

## 12. Merging and Concatenating

### `pd.concat()` — Stack DataFrames Together

```python
import pandas as pd

df1 = pd.DataFrame({"name": ["Alice", "Bob"], "score": [88, 75]})
df2 = pd.DataFrame({"name": ["Carol", "David"], "score": [92, 81]})

# Stack vertically (add more rows)
combined = pd.concat([df1, df2], ignore_index=True)
print(combined)
#     name  score
# 0  Alice     88
# 1    Bob     75
# 2  Carol     92
# 3  David     81

# Stack horizontally (add more columns)
df_a = pd.DataFrame({"name": ["Alice", "Bob"]})
df_b = pd.DataFrame({"score": [88, 75], "grade": ["B", "C"]})
combined = pd.concat([df_a, df_b], axis=1)
print(combined)
#     name  score grade
# 0  Alice     88     B
# 1    Bob     75     C
```

### `pd.merge()` — SQL-Style Joins

```python
import pandas as pd

employees = pd.DataFrame({
    "emp_id": [1, 2, 3, 4],
    "name":   ["Alice", "Bob", "Carol", "David"],
    "dept_id": [10, 20, 10, 30]
})

departments = pd.DataFrame({
    "dept_id": [10, 20, 30],
    "dept_name": ["Engineering", "Marketing", "Design"]
})

# Inner join: only rows with matching keys in BOTH tables
result = pd.merge(employees, departments, on="dept_id", how="inner")
print(result)
#    emp_id   name  dept_id     dept_name
# 0       1  Alice       10   Engineering
# 1       3  Carol       10   Engineering
# 2       2    Bob       20     Marketing
# 3       4  David       30        Design

# Left join: all rows from left, match from right (NaN if no match)
result_left = pd.merge(employees, departments, on="dept_id", how="left")

# Outer join: all rows from both tables
extra_dept = pd.DataFrame({
    "dept_id": [10, 20, 40],
    "dept_name": ["Engineering", "Marketing", "HR"]
})
result_outer = pd.merge(employees, extra_dept, on="dept_id", how="outer")
```

**Join types visualized:**
- **inner**: rows that appear in BOTH tables (intersection)
- **left**: all rows from left + matching rows from right
- **right**: all rows from right + matching rows from left
- **outer**: all rows from BOTH tables (union)

---

## 13. String Operations

Pandas provides vectorized string operations through the `.str` accessor:

```python
import pandas as pd

df = pd.DataFrame({
    "email": ["alice@example.com", "BOB@GMAIL.COM", "carol.white@company.org"],
    "description": ["Senior Engineer", "Marketing Lead", "Junior Designer"]
})

# Case operations
print(df["email"].str.lower())
# 0       alice@example.com
# 1       bob@gmail.com
# 2    carol.white@company.org

print(df["description"].str.upper())
print(df["description"].str.title())

# Check if string contains a pattern
print(df["description"].str.contains("Senior"))
# 0     True
# 1    False
# 2    False

# Extract part of string
print(df["email"].str.split("@").str[0])   # username part
# 0           alice
# 1             BOB
# 2    carol.white

# Replace
print(df["description"].str.replace("Junior", "Associate"))

# String length
print(df["description"].str.len())

# Strip whitespace
df["email"] = df["email"].str.strip().str.lower()

# Filter rows where email contains "gmail"
gmail_users = df[df["email"].str.contains("gmail")]
print(gmail_users)
```

---

## 14. Categorical Data

```python
import pandas as pd

df = pd.DataFrame({
    "name": ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace"],
    "dept": ["Eng", "Mkt", "Eng", "Design", "Eng", "Mkt", "Eng"]
})

# Count occurrences of each category
print(df["dept"].value_counts())
# Eng       4
# Mkt       2
# Design    1
# Name: dept, dtype: int64

# Percentage
print(df["dept"].value_counts(normalize=True).round(3))
# Eng       0.571
# Mkt       0.286
# Design    0.143

# Unique values
print(df["dept"].unique())     # ['Eng' 'Mkt' 'Design']
print(df["dept"].nunique())    # 3 — number of unique values

# Convert to Categorical dtype (saves memory, enables ordering)
df["dept"] = pd.Categorical(
    df["dept"],
    categories=["Design", "Mkt", "Eng"],
    ordered=True   # Design < Mkt < Eng
)
print(df["dept"].dtype)    # category
print(df.sort_values("dept"))   # Sorted by category order

# pd.get_dummies — one-hot encode for ML
dummies = pd.get_dummies(df["dept"], prefix="dept")
print(dummies)
#    dept_Design  dept_Eng  dept_Mkt
# 0        False      True     False
# 1        False     False      True
# 2        False      True     False
# ...
```

---

## 15. From Pandas to NumPy

Since ML algorithms work with NumPy arrays, you will often need to convert your cleaned DataFrame back to NumPy:

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "age":    [25, 30, 22, 28],
    "salary": [65000, 72000, 58000, 95000],
    "rating": [4.5, 3.8, 4.2, 4.9],
    "dept":   ["Eng", "Mkt", "Eng", "Eng"]
})

# Select only numeric columns for ML
numeric_cols = ["age", "salary", "rating"]
X = df[numeric_cols].to_numpy()
print(type(X))    # <class 'numpy.ndarray'>
print(X.shape)    # (4, 3)

# .values is equivalent (slightly older style)
X2 = df[numeric_cols].values
print(type(X2))   # <class 'numpy.ndarray'>

# Convert to a specific dtype
X_float32 = df[numeric_cols].to_numpy(dtype=np.float32)

# Labels as a 1D numpy array
y = df["salary"].to_numpy()
print(y)   # [65000 72000 58000 95000]

# Full ML pipeline
# 1. One-hot encode categorical features
dept_dummies = pd.get_dummies(df["dept"], prefix="dept")

# 2. Combine with numeric features
df_ml = pd.concat([df[numeric_cols], dept_dummies], axis=1)
print(df_ml)

# 3. Convert to NumPy for scikit-learn
X_final = df_ml.to_numpy(dtype=np.float32)
print("Final feature matrix shape:", X_final.shape)   # (4, 6)
```

---

## 16. A Full Data Cleaning Example

Here is a realistic mini-pipeline combining everything:

```python
import pandas as pd
import numpy as np

# Simulate messy data
data = {
    "age":    [25, None, 22, 28, 35, 30],
    "salary": [65000, 72000, None, 95000, 48000, 72000],
    "dept":   ["Eng", "Mkt", "Eng", "eng", None, "MKT"],
    "rating": [4.5, 3.8, 4.2, 4.9, 3.5, None],
    "years":  [3, 7, 2, 10, 1, 5]
}
df = pd.DataFrame(data)

print("=== Original ===")
print(df)
print("\nMissing values:\n", df.isnull().sum())

# Step 1: Standardize categorical column
df["dept"] = df["dept"].str.upper().str.strip()
print("\nAfter dept cleanup:", df["dept"].unique())

# Step 2: Fill missing values
df["age"] = df["age"].fillna(df["age"].median())
df["salary"] = df["salary"].fillna(df["salary"].median())
df["rating"] = df["rating"].fillna(df["rating"].mean())
df["dept"] = df["dept"].fillna(df["dept"].mode()[0])

# Step 3: Add derived feature
df["salary_per_year"] = df["salary"] / df["years"]

# Step 4: Sort and reset index
df = df.sort_values("salary", ascending=False).reset_index(drop=True)

print("\n=== Cleaned ===")
print(df)
print("\nMissing values after cleaning:\n", df.isnull().sum())

# Step 5: Export to NumPy for ML
feature_cols = ["age", "salary", "rating", "years"]
X = df[feature_cols].to_numpy(dtype=np.float64)
print("\nFeature matrix shape:", X.shape)
```

---

## Summary

| Task | Method |
|------|--------|
| Load CSV | `pd.read_csv("file.csv")` |
| Inspect | `head()`, `info()`, `describe()`, `shape`, `dtypes` |
| Select columns | `df["col"]`, `df[["a","b"]]` |
| Select by label | `df.loc[row, col]` |
| Select by position | `df.iloc[row, col]` |
| Filter rows | `df[df["col"] > value]` |
| Multi-condition filter | `df[(cond1) & (cond2)]` |
| Missing values | `isnull()`, `dropna()`, `fillna()` |
| Add column | `df["new"] = expression` |
| Sort | `sort_values("col")` |
| Group & aggregate | `groupby("col").mean()` |
| Apply function | `df["col"].apply(func)` |
| Stack DataFrames | `pd.concat([df1, df2])` |
| Join DataFrames | `pd.merge(df1, df2, on="key", how="inner")` |
| String ops | `df["col"].str.lower()`, `.str.contains()` |
| One-hot encode | `pd.get_dummies(df["col"])` |
| Convert to NumPy | `df.to_numpy()` or `df.values` |

> **Next steps:** Now that you can load, clean, and organize data with Pandas, the next lesson teaches you how to visualize it with Matplotlib — an essential skill for exploring data and presenting results.
