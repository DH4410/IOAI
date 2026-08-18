---
title: Lists, Tuples, and Sets
track: python
order: 6
estimatedTime: 40
difficulty: beginner
---

# Lists, Tuples, and Sets

Up to now, each variable has held a single value — one number, one string, one boolean. But real data comes in bunches: a class of test scores, a row of pixels, a batch of images, a sequence of words. You need a way to store many values in one place. That's what **collections** are for, and the most important one in Python is the **list**.

This lesson covers three collection types — lists, tuples, and sets — and when to use each. Lists especially are the workhorse of Python and the direct ancestor of the NumPy arrays and tensors that hold all your data in machine learning. Getting comfortable here pays off for the rest of the course.

---

## Lists: storing many values

A **list** is an ordered collection of values. You create one with square brackets `[]`, separating items with commas:

```python
scores = [85, 92, 78, 90, 88]
names = ["Ada", "Alan", "Grace"]
mixed = [1, "hello", 3.14, True]   # a list can hold different types
empty = []                          # an empty list

print(scores)
print(len(scores))   # 5 — how many items
```

A few things to notice right away:

- Lists keep their **order**. The items stay in the sequence you put them.
- A list can hold **any** types, even mixed together (though in practice you usually keep them the same type).
- `len()` — the same function you used on strings — tells you how many items are in the list.

Lists are the go-to container in Python. Any time you think "I have several of these," reach for a list.

### Indexing and slicing lists

Good news: everything you learned about string indexing and slicing works exactly the same on lists. Positions start at **0**, negative indexes count from the end, and slicing uses `[start:stop:step]`:

```python
fruits = ["apple", "banana", "cherry", "date", "fig"]

print(fruits[0])     # apple   (first item)
print(fruits[2])     # cherry  (third item)
print(fruits[-1])    # fig     (last item)

print(fruits[1:3])   # ['banana', 'cherry']  (stops before index 3)
print(fruits[:2])    # ['apple', 'banana']
print(fruits[::-1])  # reversed list
```

The consistency here is a gift: master indexing once, and it works on strings, lists, NumPy arrays, and tensors. It's the same idea everywhere.

### Lists are mutable — you can change them

Here's a big difference from strings: **lists can be changed after they're created**. This property is called being **mutable**. You can replace an item by assigning to its index:

```python
scores = [85, 92, 78]
scores[1] = 100        # change the second item
print(scores)          # [85, 100, 78]
```

Remember strings are *immutable* — you couldn't do `word[1] = "x"`. Lists are *mutable* — you can. This makes lists perfect for data that grows and changes as your program runs, which is most data.

---

## List methods: growing, shrinking, and reordering

Lists come with many built-in methods (functions called with a dot) for modifying them. Here are the ones you'll use constantly.

### `.append()` — add to the end

`.append()` adds a single item to the end of the list. This is probably the most-used list method of all:

```python
scores = [85, 92]
scores.append(78)
scores.append(90)
print(scores)   # [85, 92, 78, 90]
```

The classic pattern is starting with an empty list and appending as you go — for example, collecting results inside a loop:

```python
squares = []
for n in range(1, 6):
    squares.append(n * n)
print(squares)   # [1, 4, 9, 16, 25]
```

You'll build lists this way all the time — collecting predictions, gathering loss values across training epochs, accumulating processed data.

### `.insert()` — add at a specific position

`.insert(index, value)` puts an item at a chosen position, shifting everything after it to the right:

```python
letters = ["a", "b", "d"]
letters.insert(2, "c")   # insert "c" at index 2
print(letters)           # ['a', 'b', 'c', 'd']
```

### `.pop()` — remove and return by position

`.pop()` removes an item and gives it back to you. With no argument it removes the last item; with an index it removes that specific one:

```python
stack = [10, 20, 30, 40]
last = stack.pop()       # removes and returns 40
print(last)              # 40
print(stack)             # [10, 20, 30]

first = stack.pop(0)     # removes and returns the item at index 0
print(first)             # 10
print(stack)             # [20, 30]
```

Because `.pop()` both removes *and* returns, it's great when you want to process items one at a time and take them out as you go.

### `.remove()` — remove by value

`.remove(value)` deletes the first item that matches a value (not a position):

```python
fruits = ["apple", "banana", "cherry", "banana"]
fruits.remove("banana")   # removes the FIRST banana only
print(fruits)             # ['apple', 'cherry', 'banana']
```

If the value isn't in the list, `.remove()` raises a `ValueError`. Use the `in` operator first if you're not sure it's there.

### `.sort()` and `.reverse()` — reordering

`.sort()` arranges the items in ascending order, and `.reverse()` flips the order. Both change the list **in place** (they modify the original and return `None`):

```python
numbers = [5, 2, 8, 1, 9]
numbers.sort()
print(numbers)   # [1, 2, 5, 8, 9]

numbers.sort(reverse=True)   # descending
print(numbers)   # [9, 8, 5, 2, 1]

numbers.reverse()
print(numbers)   # [1, 2, 5, 8, 9]
```

> Big gotcha: `.sort()` changes the list and returns `None`. So `x = numbers.sort()` gives `x = None`, not the sorted list! If you write `sorted_scores = scores.sort()` you'll be baffled when `sorted_scores` is `None`. If you want a sorted **copy** while keeping the original, use the `sorted()` function instead: `new_list = sorted(scores)`. Remember this distinction — it catches everyone.

Here's a quick reference table of the main list methods:

| Method | What it does | Changes the list? |
|---|---|---|
| `.append(x)` | add `x` to the end | yes |
| `.insert(i, x)` | insert `x` at index `i` | yes |
| `.pop()` / `.pop(i)` | remove & return last / item `i` | yes |
| `.remove(x)` | remove first item equal to `x` | yes |
| `.sort()` | sort in place | yes |
| `.reverse()` | reverse in place | yes |
| `len(list)` | count items | no |
| `sorted(list)` | return a sorted **copy** | no |

---

## Tuples: lists that can't change

A **tuple** is like a list, but **immutable** — once created, you can't change, add, or remove items. You make one with parentheses `()` instead of square brackets (or sometimes just commas):

```python
point = (3, 5)
rgb = (255, 128, 0)
single = (42,)        # a one-item tuple needs a trailing comma!

print(point[0])       # 3   (indexing works like lists)
print(point[1])       # 5
print(len(rgb))       # 3
```

Everything you can *read* from a list works on a tuple — indexing, slicing, `len()`, looping. But anything that would *change* it fails:

```python
point = (3, 5)
point[0] = 10   # TypeError! tuples cannot be changed
```

### When to use a tuple instead of a list

If tuples are just restricted lists, why use them? A few good reasons:

- **To signal "this shouldn't change."** When a group of values naturally belongs together and shouldn't be modified — like the (x, y) coordinates of a point, or the (red, green, blue) values of a color — a tuple communicates that intent. It also protects you from accidentally changing it.
- **They're slightly faster and use less memory** than lists, because Python knows they won't change.
- **They can be used as dictionary keys** (you'll see dictionaries next lesson), while lists cannot.

A very common everyday use of tuples is returning **multiple values** from a function:

```python
def min_and_max(numbers):
    return min(numbers), max(numbers)   # returns a tuple

low, high = min_and_max([3, 7, 1, 9, 4])
print(low)    # 1
print(high)   # 9
```

The function packs both results into a tuple, and the caller unpacks them into two variables in one clean line. This is a favorite Python pattern, and it's how a lot of library functions hand back several results at once.

In machine learning, you'll see tuples constantly for **shapes**. An image tensor might have shape `(3, 224, 224)` — three color channels, 224 pixels tall, 224 wide. That shape is a tuple because those dimensions are fixed facts about the data that shouldn't be accidentally altered.

---

## Sets: collections of unique items

A **set** is an unordered collection where every item is **unique** — duplicates are automatically removed. You create one with curly braces `{}`:

```python
numbers = {1, 2, 3, 2, 1, 3}
print(numbers)   # {1, 2, 3}  — duplicates gone!

colors = {"red", "green", "blue"}
print(len(colors))   # 3
```

Two things define a set:

- **No duplicates.** Add the same item twice and it only appears once. This is the killer feature.
- **No order.** Sets don't track position, so you can't index them (`numbers[0]` is an error). If order matters, use a list.

The most common use of a set is **removing duplicates** from a list — just convert it to a set and back:

```python
words = ["cat", "dog", "cat", "bird", "dog", "cat"]
unique_words = list(set(words))
print(len(unique_words))   # 3
```

Sets also make it lightning-fast to check membership. Testing `x in my_set` is much faster than `x in my_list` for large collections, which matters when you're checking millions of items.

### Set operations: union, intersection, difference

Sets support the mathematical operations you know from Venn diagrams. These are genuinely useful for comparing groups:

```python
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

print(a | b)    # union: everything in either       {1, 2, 3, 4, 5, 6}
print(a & b)    # intersection: only in both        {3, 4}
print(a - b)    # difference: in a but not b         {1, 2}
```

You can also write these as methods: `a.union(b)`, `a.intersection(b)`, `a.difference(b)`. Set operations are perfect for questions like "which words appear in both documents?" (intersection) or "which users are in group A but not group B?" (difference) — questions that come up all the time in data work.

Here's a quick comparison of the three collection types:

| Type | Syntax | Ordered? | Changeable? | Duplicates? |
|---|---|---|---|---|
| list | `[1, 2, 3]` | yes | yes (mutable) | allowed |
| tuple | `(1, 2, 3)` | yes | no (immutable) | allowed |
| set | `{1, 2, 3}` | no | yes | not allowed |

---

## List comprehensions

You met list comprehensions briefly in the control flow lesson. They're so useful for data work that they deserve a proper look here. A list comprehension builds a new list from an existing sequence in a single, readable line.

The basic form is `[expression for item in sequence]`:

```python
# Long way
squares = []
for n in range(6):
    squares.append(n ** 2)

# Comprehension — same result
squares = [n ** 2 for n in range(6)]
print(squares)   # [0, 1, 4, 9, 16, 25]
```

You can transform items in any way you like:

```python
names = ["ada", "alan", "grace"]
capitalized = [name.upper() for name in names]
print(capitalized)   # ['ADA', 'ALAN', 'GRACE']

prices = [10, 20, 30]
with_tax = [p * 1.2 for p in prices]
print(with_tax)   # [12.0, 24.0, 36.0]
```

### Filtering with a condition

Add `if condition` at the end to keep only certain items:

```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

evens = [n for n in numbers if n % 2 == 0]
print(evens)   # [2, 4, 6, 8, 10]

big = [n for n in numbers if n > 5]
print(big)     # [6, 7, 8, 9, 10]
```

You can transform and filter at the same time — "square each even number":

```python
result = [n ** 2 for n in numbers if n % 2 == 0]
print(result)   # [4, 16, 36, 64, 100]
```

Comprehensions are beloved in the Python and data-science world because they express a common idea — "make a new list by transforming/filtering an old one" — clearly and concisely. You'll read and write them constantly. The mental habit to build: whenever you catch yourself writing "make an empty list, loop, and append," ask whether a comprehension would be cleaner. Often it is.

> Don't overdo it, though. If a comprehension gets long or has multiple conditions and nested loops crammed in, it becomes hard to read. When that happens, a plain `for` loop is the friendlier choice. Clarity beats cleverness.

---

## Nested lists: a first look at matrices

A list can contain other lists. This gives you a **nested list**, which is how you represent a grid, table, or **matrix** in plain Python:

```python
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

print(matrix[0])       # [1, 2, 3]  — the first row
print(matrix[1][2])    # 6          — row 1, column 2
```

To reach a single element you use two indexes: the first picks the row, the second picks the column within that row. So `matrix[1][2]` means "row index 1, then column index 2" — the value 6.

To walk over every element, use nested loops (one for rows, one for columns):

```python
matrix = [[1, 2, 3], [4, 5, 6]]
for row in matrix:
    for value in row:
        print(value, end=" ")
    print()   # newline after each row
```

This 2D structure is a huge deal for AI. A grayscale image is a grid of pixel brightnesses — a matrix. A color image is a grid with three numbers per pixel. A dataset table is rows and columns. Even the weights inside a neural network are matrices. Plain nested lists *can* hold all this, but they're slow and clumsy for real math — which is exactly why NumPy exists. When you get to the NumPy lesson, you'll see this same 2D idea done properly and fast. Nested lists are the mental warm-up.

---

## `zip()` and `enumerate()`: two essential loop helpers

Two built-in functions make looping over data much nicer. You'll reach for them all the time.

### `enumerate()` — loop with an index

Often you want both the item *and* its position number while looping. Instead of managing a separate counter, `enumerate()` gives you both:

```python
fruits = ["apple", "banana", "cherry"]

for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")
# 0: apple
# 1: banana
# 2: cherry
```

Each time around, `enumerate` hands you a pair: the index and the item. This is cleaner and less error-prone than writing `for i in range(len(fruits))` and then indexing. You can start counting from a different number with `enumerate(fruits, start=1)` if you prefer 1-based labels.

### `zip()` — loop over several lists together

When you have two (or more) lists that line up — like names and their scores — `zip()` lets you walk through them in parallel, one matched pair at a time:

```python
names = ["Ada", "Alan", "Grace"]
scores = [95, 88, 92]

for name, score in zip(names, scores):
    print(f"{name} scored {score}")
# Ada scored 95
# Alan scored 88
# Grace scored 92
```

`zip` pairs up the first item of each list, then the second, and so on. If the lists are different lengths, it stops at the shortest one. This is perfect for data where related values live in separate lists — which happens a lot. For example, pairing up model predictions with the true answers to check how many match:

```python
predictions = [1, 0, 1, 1, 0]
actual      = [1, 0, 0, 1, 0]

correct = 0
for pred, true in zip(predictions, actual):
    if pred == true:
        correct += 1
print(f"Accuracy: {correct}/{len(actual)}")   # 4/5
```

That little loop — comparing predictions to the truth with `zip` — is the essence of measuring a model's accuracy. You just wrote a tiny piece of real machine learning evaluation.

---

## Why lists and arrays matter in AI

Everything in machine learning is data, and data lives in collections. The list is your first taste of how that works, and it leads directly to the tools you'll rely on for the rest of the course.

- **Data is stored in arrays.** A dataset of 10,000 images, a batch of training examples, a sequence of words — all of these are collections of values. Lists are the plain-Python version; NumPy arrays and PyTorch tensors are the fast, math-ready versions. But the mental model is the same: many values, accessed by index.

- **Indexing and slicing are universal.** The `[0]`, `[-1]`, `[2:5]` operations you learned on lists work identically on arrays and tensors. When you slice out the first 32 images of a batch, or grab one column of a dataset, you're using exactly this skill.

- **Order matters (usually).** In a time series, a sentence, or a sequence of video frames, the order carries meaning — which is why lists (ordered) fit better than sets (unordered) for most ML data.

- **Nested lists become matrices and tensors.** The 2D nested-list idea scales up to the multi-dimensional arrays that represent images, matrices of network weights, and batches of data. A tensor is really just a nested list done efficiently.

- **`zip` and `enumerate` show up in every training loop.** Pairing predictions with labels (zip) and tracking which batch you're on (enumerate) are everyday operations when training and evaluating models.

The humble list is the seed of it all. Understand it deeply, and NumPy arrays will feel like a natural, souped-up version of the same idea.

---

## Common mistakes with collections

**Mistake 1: `list.sort()` returns `None`.** `x = my_list.sort()` sets `x` to `None`, not the sorted list, because `.sort()` sorts in place. Use `sorted(my_list)` if you want a returned copy.

**Mistake 2: Trying to change a tuple.** `my_tuple[0] = 5` raises a `TypeError`. Tuples are immutable. Use a list if you need to change items.

**Mistake 3: Indexing a set.** `my_set[0]` is an error — sets have no order and no positions. Convert to a list first if you need indexing.

**Mistake 4: `.remove()` on a missing value.** `my_list.remove(x)` raises `ValueError` if `x` isn't there. Check with `if x in my_list` first.

**Mistake 5: Modifying a list while looping over it.** Adding or removing items from a list during a `for` loop over it causes skipped items and confusing bugs. Loop over a copy, or build a new list instead.

---

## Putting it all together

Here's a program that uses lists, a comprehension, `zip`, and set operations to analyze quiz results:

```python
students = ["Ada", "Alan", "Grace", "Katherine", "Alan"]
scores = [88, 72, 95, 60]   # note: fewer scores than the raw student list

# Remove duplicate student names using a set
unique_students = list(set(students))
print(f"Unique students: {len(unique_students)}")

# Pair the first four students with their scores
for name, score in zip(students, scores):
    print(f"{name}: {score}")

# Use a comprehension to find who passed (>= 70)
passers = [score for score in scores if score >= 70]
print(f"Passing scores: {passers}")

# Stats using built-in functions
print(f"Highest: {max(scores)}")
print(f"Lowest: {min(scores)}")
print(f"Average: {sum(scores) / len(scores):.1f}")
```

Trace it before running. You'll see set deduplication, `zip` pairing, a filtering comprehension, and the built-in `max`, `min`, and `sum` functions working on a list — a realistic slice of everyday data handling.

---

## List, Tuple, or Set?

```widget
{
  "type": "concept-sort",
  "title": "Which Collection Type Should You Use?",
  "categories": [
    { "name": "List [ ]", "color": "#5B5BD6" },
    { "name": "Tuple ( )", "color": "#22C55E" },
    { "name": "Set { }", "color": "#F97316" }
  ],
  "items": [
    { "text": "Store training losses in order — you'll append one per epoch", "category": "List [ ]" },
    { "text": "Return width, height, channels from an image shape function", "category": "Tuple ( )" },
    { "text": "Find all unique class labels in a dataset", "category": "Set { }" },
    { "text": "Check whether 'cat' is in the vocabulary (fast lookup)", "category": "Set { }" },
    { "text": "The (row, col) coordinates of a pixel — fixed pair", "category": "Tuple ( )" },
    { "text": "Batch of 32 input images you'll iterate over", "category": "List [ ]" },
    { "text": "Remove duplicate IDs from test data", "category": "Set { }" },
    { "text": "A sequence of tokens from a sentence, in order", "category": "List [ ]" }
  ]
}
```

---

## Summary

- A **list** (`[1, 2, 3]`) is an ordered, **mutable** collection. Index and slice it just like a string (starting at 0).
- Key list methods: `.append()` (add to end), `.insert()`, `.pop()` (remove & return), `.remove()` (by value), `.sort()`, `.reverse()`.
- `.sort()` sorts in place and returns `None` — use `sorted()` for a returned copy.
- A **tuple** (`(1, 2, 3)`) is an ordered but **immutable** collection. Use it for fixed groups of values, shapes, and returning multiple values from a function.
- A **set** (`{1, 2, 3}`) is unordered with **no duplicates**. Great for removing duplicates and for union/intersection/difference (`|`, `&`, `-`).
- **List comprehensions** (`[n**2 for n in range(6) if n % 2 == 0]`) build lists concisely by transforming and filtering.
- **Nested lists** represent 2D grids and matrices; reach an element with two indexes (`matrix[row][col]`).
- `enumerate()` gives you index + item while looping; `zip()` walks over several lists in parallel.
- In AI, all data lives in collections. Lists lead directly to NumPy arrays and tensors, where the same indexing, slicing, and looping ideas power everything.
