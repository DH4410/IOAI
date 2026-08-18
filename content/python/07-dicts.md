---
title: Dictionaries
track: python
order: 7
estimatedTime: 35
difficulty: beginner
---

# Dictionaries

Lists are great when your data is a plain sequence and you find things by position: "give me item number 3." But a lot of data isn't like that. Sometimes you want to look things up by a **name** or a **label**: "what's Ada's score?", "what does the word 'cat' map to?", "what's the setting for learning rate?" For this, Python gives you the **dictionary** — one of the most useful and most-used data structures in the entire language.

If lists are numbered rows, dictionaries are labeled drawers. You'll use them everywhere in real programs and constantly in AI, where they store settings, label mappings, counts, and results. Let's dig in.

---

## What is a dictionary?

A **dictionary** (`dict`) stores data as **key-value pairs**. Each **key** is a label, and each **value** is the data attached to that label. You look things up by key, not by position. Think of a real dictionary book: you look up a word (the key) to find its definition (the value).

You create a dictionary with curly braces `{}`, writing each pair as `key: value`:

```python
student = {
    "name": "Ada",
    "age": 20,
    "score": 95
}

print(student)
print(student["name"])   # Ada
print(student["score"])  # 95
```

To get a value, you put its key in square brackets: `student["name"]`. This is like list indexing, except instead of a number position you use the key.

Some things to notice:

- Keys are usually strings, but can be numbers, tuples, or other immutable types.
- Values can be *anything* — numbers, strings, lists, even other dictionaries.
- Each key must be **unique**. If you repeat a key, the last one wins.

```python
prices = {"apple": 1.20, "banana": 0.50, "cherry": 3.00}
print(prices["banana"])   # 0.50

# Keys can be numbers too
squares = {1: 1, 2: 4, 3: 9}
print(squares[2])         # 4
```

An empty dictionary is just `{}`:

```python
empty = {}
print(len(empty))   # 0
```

> A dictionary is unordered in meaning (you look up by key, not position), though modern Python does remember the insertion order when you loop. The key point is: you access values by their **key**, never by a numeric index. Writing `student[0]` would look for a key `0`, not the first item — and if there's no such key, it errors.

---

## Adding, updating, and deleting

Dictionaries are **mutable**, like lists — you can change them freely after creating them.

### Adding and updating

To add a new key-value pair, or to change an existing value, you just assign to the key. If the key doesn't exist yet, it's created; if it does, its value is replaced:

```python
student = {"name": "Ada", "age": 20}

student["score"] = 95        # add a new key
print(student)               # {'name': 'Ada', 'age': 20, 'score': 95}

student["age"] = 21          # update an existing key
print(student["age"])        # 21
```

The same syntax does both jobs. Python doesn't need separate "add" and "update" operations — assigning to a key just makes sure that key holds that value afterward.

### Deleting

To remove a pair, use `del`:

```python
student = {"name": "Ada", "age": 20, "score": 95}
del student["score"]
print(student)   # {'name': 'Ada', 'age': 20}
```

There's also `.pop(key)`, which removes a key **and returns its value** — handy when you want to grab a value on the way out:

```python
scores = {"Ada": 95, "Alan": 88}
alans_score = scores.pop("Alan")
print(alans_score)   # 88
print(scores)        # {'Ada': 95}
```

---

## The KeyError problem and `.get()`

Here's the most common dictionary pitfall. If you try to access a key that doesn't exist, Python raises a `KeyError` and stops:

```python
student = {"name": "Ada", "age": 20}
print(student["score"])   # KeyError: 'score'
```

There's no `"score"` key, so Python throws an error. This is a frequent source of crashes, especially when working with data that might have missing fields.

The safe way to look up a possibly-missing key is the `.get()` method. Instead of crashing, `.get()` returns `None` (or a default you choose) when the key is absent:

```python
student = {"name": "Ada", "age": 20}

print(student.get("score"))         # None — no crash
print(student.get("score", 0))      # 0 — your chosen default
print(student.get("name", "?"))     # Ada — the key exists, so you get its value
```

> Use `dict[key]` when you're sure the key exists (or you *want* it to crash if it doesn't, to catch a bug). Use `dict.get(key, default)` when the key might be missing and you'd rather have a fallback than a crash. This one habit prevents a huge fraction of real-world dictionary bugs.

You can also check for a key first with the `in` operator, which returns a boolean:

```python
student = {"name": "Ada", "age": 20}
print("name" in student)    # True
print("score" in student)   # False

if "score" in student:
    print(student["score"])
else:
    print("No score recorded")
```

Note that `in` checks the **keys**, not the values. `"Ada" in student` would be `False` here, because `"Ada"` is a value, not a key.

---

## Dictionary methods: keys, values, items

Three methods let you pull apart a dictionary's contents. They're essential for looping and inspecting.

- `.keys()` gives you all the keys.
- `.values()` gives you all the values.
- `.items()` gives you all the key-value pairs, each as a tuple.

```python
prices = {"apple": 1.20, "banana": 0.50, "cherry": 3.00}

print(list(prices.keys()))     # ['apple', 'banana', 'cherry']
print(list(prices.values()))   # [1.2, 0.5, 3.0]
print(list(prices.items()))    # [('apple', 1.2), ('banana', 0.5), ('cherry', 3.0)]
```

These are most useful for looping, which we'll cover next. One handy trick: since `.values()` gives you all the values, you can do math on them directly:

```python
prices = {"apple": 1.20, "banana": 0.50, "cherry": 3.00}
print(sum(prices.values()))    # 4.7 — total of all prices
print(max(prices.values()))    # 3.0 — most expensive
print(len(prices))             # 3 — number of items
```

---

## Iterating over dictionaries

Looping over dictionaries is something you'll do all the time. There are three natural ways, matching the three methods above.

**Loop over keys (the default).** If you loop over a dictionary directly, you get its keys:

```python
prices = {"apple": 1.20, "banana": 0.50}

for fruit in prices:
    print(fruit)              # apple, banana (just the keys)

# You can look up the value inside the loop
for fruit in prices:
    print(f"{fruit}: {prices[fruit]}")
```

**Loop over values** with `.values()`:

```python
for price in prices.values():
    print(price)              # 1.2, 0.5
```

**Loop over key-value pairs** with `.items()` — this is the most useful, because you get both at once:

```python
for fruit, price in prices.items():
    print(f"{fruit} costs ${price}")
# apple costs $1.2
# banana costs $0.5
```

The `.items()` loop unpacks each pair into two variables (`fruit` and `price`) automatically. This is the go-to way to process a whole dictionary, and it reads beautifully. Whenever you think "for each key and its value...", reach for `.items()`.

---

## Counting with dictionaries: a classic pattern

One of the most common uses of a dictionary is **counting** how many times each thing appears. This pattern shows up everywhere — counting words in a document, counting how many of each label are in a dataset, tallying survey responses.

The idea: use each item as a key, and its count as the value. When you see an item, add one to its count:

```python
words = ["cat", "dog", "cat", "bird", "cat", "dog"]

counts = {}
for word in words:
    if word in counts:
        counts[word] += 1     # seen before, increment
    else:
        counts[word] = 1      # first time, start at 1

print(counts)   # {'cat': 3, 'dog': 2, 'bird': 1}
```

There's an even cleaner version using `.get()` with a default of 0:

```python
words = ["cat", "dog", "cat", "bird", "cat", "dog"]

counts = {}
for word in words:
    counts[word] = counts.get(word, 0) + 1

print(counts)   # {'cat': 3, 'dog': 2, 'bird': 1}
```

Read that key line as: "take the current count (0 if we've never seen this word), add 1, and store it back." This little pattern is genuinely useful and worth memorizing. In NLP it's how you build **word frequency counts** — the foundation of understanding which words matter in a text.

---

## Nested dictionaries

A dictionary's values can themselves be dictionaries. This lets you represent structured, hierarchical data — like a record for each student:

```python
students = {
    "ada": {"age": 20, "score": 95},
    "alan": {"age": 22, "score": 88}
}

print(students["ada"])            # {'age': 20, 'score': 95}
print(students["ada"]["score"])   # 95  — chain the keys
```

To reach deep into a nested dictionary, you chain the keys: `students["ada"]["score"]` means "in students, find ada; in that record, find score." This is exactly like reaching into a nested list, but with names instead of numbers.

Nested dictionaries are how a lot of real data is shaped, especially data that comes from the web. The **JSON** format — which powers almost every web API and config file — maps directly onto Python's nested dictionaries and lists. When you fetch data from an online source in a competition, you'll very often get back nested dictionaries and navigate them exactly this way.

---

## Dictionary comprehensions

Just like list comprehensions build lists, **dictionary comprehensions** build dictionaries in one line. The syntax is `{key: value for item in sequence}`:

```python
# Build a dict mapping numbers to their squares
squares = {n: n ** 2 for n in range(1, 6)}
print(squares)   # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# Build a dict from two parallel lists using zip
names = ["Ada", "Alan", "Grace"]
scores = [95, 88, 92]
score_dict = {name: score for name, score in zip(names, scores)}
print(score_dict)   # {'Ada': 95, 'Alan': 88, 'Grace': 92}
```

That second example — zipping two lists into a dictionary — is a common and tidy way to pair up related data. You can filter too, keeping only some pairs:

```python
prices = {"apple": 1.20, "banana": 0.50, "cherry": 3.00}
cheap = {fruit: price for fruit, price in prices.items() if price < 2}
print(cheap)   # {'apple': 1.2, 'banana': 0.5}
```

Dictionary comprehensions keep your code compact when you're transforming or filtering key-value data.

---

## When to use a dict vs a list

Both lists and dictionaries store many values, so how do you choose? Here's the guiding question: **how do you want to find your data — by position, or by name?**

| Use a **list** when... | Use a **dict** when... |
|---|---|
| Order matters | You look things up by a label/key |
| You access items by position (0, 1, 2...) | You access items by name ("score", "age") |
| The data is a plain sequence | Each item has a meaningful identifier |
| You'll loop through everything in order | You need fast lookups by key |

Concretely: a list of test scores `[85, 92, 78]` is fine if you just need the numbers in order. But if you need to know *whose* score is whose, a dictionary `{"Ada": 85, "Alan": 92, "Grace": 78}` is far better — you can instantly ask for `scores["Ada"]` without hunting.

A subtle but important point: looking something up by key in a dictionary is **fast** no matter how big the dictionary is. Searching a list for a value, by contrast, gets slower as the list grows (Python has to check items one by one). So when you'll be doing lots of lookups by name, a dictionary isn't just cleaner — it's faster.

---

## Why dictionaries matter in AI

Dictionaries are everywhere in machine learning code. Here are the big four uses you'll meet again and again:

**1. Hyperparameter dictionaries.** The settings that control how a model trains — learning rate, batch size, number of layers — are almost always grouped in a dictionary. It keeps all your knobs in one labeled place:

```python
config = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "num_epochs": 50,
    "optimizer": "adam"
}
print(f"Training for {config['num_epochs']} epochs")
```

Changing an experiment is as easy as changing a value in this dict. Many ML tools even accept a config dictionary directly.

**2. Label maps.** Models work with numbers, but humans think in words. A dictionary translates between them. If your model predicts the number 2, a label map tells you that means "cat":

```python
label_map = {0: "dog", 1: "bird", 2: "cat", 3: "fish"}
prediction = 2
print(f"The model predicts: {label_map[prediction]}")   # cat
```

Every image classifier you build will have one of these to turn its numeric outputs back into human-readable class names.

**3. Word-to-number vocabularies.** In NLP, every word gets an ID number so the model can process it. That mapping is a dictionary: `{"the": 0, "cat": 1, "sat": 2, ...}`. Building and using this **vocabulary** is a core step in text processing, and it's literally a dict.

**4. Confusion matrices and results.** When you evaluate a model, you tally how often it confused one class for another. Counts, per-class accuracies, and other results are naturally stored in dictionaries — using the exact counting pattern you learned earlier in this lesson.

The counting pattern, the label map, the config dict — these aren't advanced topics you'll meet "someday." They're in the first real ML program you'll write. Every one of them is just a dictionary. Learn dicts well and this part of AI is already familiar.

---

## Common mistakes with dictionaries

**Mistake 1: `KeyError` from a missing key.** `d["missing"]` crashes if the key isn't there. Use `d.get("missing", default)` or check `if "missing" in d` first.

**Mistake 2: Using a numeric index like a list.** `d[0]` looks for a key `0`, not the first item. Dictionaries have no positional indexing.

**Mistake 3: Expecting `in` to check values.** `x in d` checks the **keys**, not the values. To check values, use `x in d.values()`.

**Mistake 4: Forgetting `.items()` when looping for both.** Looping over `d` directly gives only keys. To get keys *and* values together, loop over `d.items()`.

**Mistake 5: Duplicate keys.** `{"a": 1, "a": 2}` keeps only `{"a": 2}` — the later value silently overwrites the earlier one. Keys must be unique.

---

## Putting it all together

Here's a program that uses a dictionary to count word frequencies in a sentence, then reports the results — a mini version of real text analysis:

```python
sentence = "the cat sat on the mat and the cat is happy"

# Count how often each word appears
counts = {}
for word in sentence.split():
    counts[word] = counts.get(word, 0) + 1

# Report
print("Word counts:")
for word, count in counts.items():
    print(f"  {word}: {count}")

# Find the most common word
most_common = max(counts, key=counts.get)
print(f"Most common word: '{most_common}' ({counts[most_common]} times)")

# Total unique words
print(f"Unique words: {len(counts)}")
```

This ties together `.split()` from the strings lesson, the `.get()` counting pattern, `.items()` looping, and a neat use of `max` with `key=counts.get` to find the highest-count key. It's a real slice of NLP preprocessing. Trace it, then run it.

---

## Dict vs. List: Which Data Structure?

```widget
{
  "type": "concept-sort",
  "title": "Dictionary or List — Which Would You Use?",
  "categories": [
    { "name": "Dictionary (lookup by name/key)", "color": "#5B5BD6" },
    { "name": "List (access by position/order)", "color": "#22C55E" }
  ],
  "items": [
    { "text": "Store 1000 training losses — one per epoch", "category": "List (access by position/order)" },
    { "text": "Map class index 0→'cat', 1→'dog', 2→'bird'", "category": "Dictionary (lookup by name/key)" },
    { "text": "Count how many times each word appears in a text", "category": "Dictionary (lookup by name/key)" },
    { "text": "Store the batch of 32 images as tensors", "category": "List (access by position/order)" },
    { "text": "Config: {'lr': 0.001, 'epochs': 50, 'batch_size': 32}", "category": "Dictionary (lookup by name/key)" },
    { "text": "The sequence of tokens in a sentence", "category": "List (access by position/order)" },
    { "text": "Look up a word's embedding vector by its word string", "category": "Dictionary (lookup by name/key)" },
    { "text": "Track the 5 validation scores across CV folds", "category": "List (access by position/order)" }
  ]
}
```

---

## Summary

- A **dictionary** (`dict`) stores **key-value pairs**. Look up values by key: `d["name"]`, not by numeric position.
- Create with `{"key": value, ...}`. Keys must be unique and immutable; values can be anything.
- Add or update a value by assigning to its key: `d["new"] = 5`. Delete with `del d["key"]` or `d.pop("key")`.
- Accessing a missing key raises `KeyError`. Use `d.get(key, default)` for a safe fallback, or check `key in d` first.
- `.keys()`, `.values()`, and `.items()` give you the keys, values, and pairs. Loop with `for k, v in d.items():`.
- The **counting pattern** `d[x] = d.get(x, 0) + 1` tallies how often each item appears — a workhorse in text analysis.
- Values can be dictionaries (**nested dicts**); reach in by chaining keys. This mirrors the JSON data you'll get from web APIs.
- **Dict comprehensions** (`{k: v for ...}`) build dictionaries compactly, often from `zip`ped lists.
- Choose a **dict** when you look data up by name, a **list** when you access by position. Dict lookups stay fast even for huge dictionaries.
- In AI, dictionaries store hyperparameter configs, label maps (number → class name), word vocabularies, and evaluation results — you'll use them in your very first ML program.
