---
title: Strings and Text
track: python
order: 3
estimatedTime: 30
difficulty: beginner
---

# Strings and Text

In Python, text is stored in a type called a **string** (`str`). Strings show up everywhere in ML: reading file names, processing labels, printing results. This lesson covers what you actually need.

---

## 1. Creating Strings

A string is any sequence of characters wrapped in quotes:

```python
name = 'Ada'
city = "London"
```

Single `'` and double `"` quotes work the same. Use whichever avoids escaping:

```python
sentence = "It's a beautiful day"    # apostrophe inside, use double quotes
quote = 'She said "hello"'           # double quotes inside, use single quotes
```

For text that spans multiple lines, use triple quotes:

```python
message = """Line one,
Line two,
Line three."""
print(message)
```

**Quick check:** Why would `'It's fine'` cause an error?
> Python sees the apostrophe as the end of the string. Use `"It's fine"` instead.

---

## 2. Indexing and Slicing

Strings are sequences. Each character has an index starting at 0.

```python
word = 'Python'
print(word[0])      # 'P'
print(word[-1])     # 'n'  (last character)
print(word[0:3])    # 'Pyt'  (characters 0, 1, 2)
print(word[2:])     # 'thon'  (from index 2 to end)
print(word[:3])     # 'Pyt'  (from start to index 3)
```

Slicing syntax: `string[start:stop]`. The start is included, the stop is not.

```python
text = 'IOAI 2025'
print(text[5:])    # '2025'
print(len(text))   # 9  (number of characters)
```

**Quick check:** What does `'hello'[1:4]` return?
> `'ell'` (characters at index 1, 2, 3)

---

## 3. Common String Methods

Methods are functions that belong to a string. Call them with `.method()`:

```python
text = '  Hello, World!  '

print(text.strip())        # 'Hello, World!'  - removes leading/trailing spaces
print(text.lower())        # '  hello, world!  '
print(text.upper())        # '  HELLO, WORLD!  '
print(text.replace('World', 'Python'))   # '  Hello, Python!  '
print('hello'.capitalize())              # 'Hello'
```

Check and find:

```python
sentence = 'machine learning is great'

print('learning' in sentence)        # True
print(sentence.startswith('machine'))  # True
print(sentence.count('a'))           # 4
print(sentence.find('learning'))     # 8  (index where it starts, -1 if not found)
```

---

## 4. Split and Join

**Split** breaks a string into a list. **Join** glues a list back into a string.

```python
csv_line = 'apple,banana,cherry'
items = csv_line.split(',')
print(items)     # ['apple', 'banana', 'cherry']

sentence = 'the quick brown fox'
words = sentence.split()     # split on spaces (default)
print(words)     # ['the', 'quick', 'brown', 'fox']
```

```python
# Join: string.join(list)
print(', '.join(['cat', 'dog', 'bird']))   # 'cat, dog, bird'
print(' '.join(words))                      # 'the quick brown fox'
```

**Quick check:** You have a CSV string `'1,2,3,4'`. How do you get a list of the numbers?
> `'1,2,3,4'.split(',')` gives `['1', '2', '3', '4']`. Note: they are strings, not integers. Use `int(x)` to convert.

---

## 5. f-Strings (Formatted Output)

The best way to build strings with variables:

```python
name = 'Dima'
score = 95.7

print(f'Hello, {name}!')             # 'Hello, Dima!'
print(f'Score: {score:.1f}%')        # 'Score: 95.7%'
print(f'2 + 2 = {2 + 2}')           # '2 + 2 = 4'
```

Put `f` before the quote and use `{expression}` inside. The `:` inside the braces controls formatting:
- `:.2f` - 2 decimal places
- `:d` - integer
- `:>10` - right-align in 10 characters

```python
for i in range(5):
    loss = 1.0 / (i + 1)
    print(f'Epoch {i+1:2d}: loss = {loss:.4f}')
```

---

## 6. Strings in ML Context

A few patterns you will see often:

```python
# Reading a label from a file path
path = 'data/cats/image001.jpg'
label = path.split('/')[1]    # 'cats'

# Checking a class name
classes = ['cat', 'dog', 'bird']
if 'cat' in classes:
    print(f'Index: {classes.index("cat")}')  # 0

# Building a file path
base = 'results'
epoch = 10
filename = f'{base}/checkpoint_{epoch:03d}.pth'   # 'results/checkpoint_010.pth'
```

---

## String Operations Sorter

```widget
{
  "type": "concept-sort",
  "title": "Which String Method Does This?",
  "categories": [
    { "name": "Cleaning / normalizing", "color": "#5B5BD6" },
    { "name": "Splitting / joining", "color": "#22C55E" },
    { "name": "Searching / checking", "color": "#F97316" }
  ],
  "items": [
    { "text": "text.lower() — convert all characters to lowercase", "category": "Cleaning / normalizing" },
    { "text": "'word1 word2'.split() — break sentence into word list", "category": "Splitting / joining" },
    { "text": "'cat' in label — check if substring exists", "category": "Searching / checking" },
    { "text": "text.strip() — remove leading/trailing whitespace", "category": "Cleaning / normalizing" },
    { "text": "', '.join(['a', 'b', 'c']) — combine list into one string", "category": "Splitting / joining" },
    { "text": "label.startswith('pos') — check prefix", "category": "Searching / checking" },
    { "text": "text.replace('\\n', ' ') — swap newlines for spaces", "category": "Cleaning / normalizing" },
    { "text": "'a,b,c'.split(',') — split on comma delimiter", "category": "Splitting / joining" }
  ]
}
```

---

## Summary

| Operation | Example |
|---|---|
| Index | `s[0]`, `s[-1]` |
| Slice | `s[2:5]` |
| Strip / lower / upper | `s.strip()`, `s.lower()` |
| Split | `'a,b'.split(',')` -> `['a', 'b']` |
| Join | `','.join(['a', 'b'])` -> `'a,b'` |
| Check | `'x' in s`, `s.startswith('x')` |
| Format | `f'Hello {name}'` |

You do not need to memorize every method. When you need something specific (left-pad, replace multiple spaces, check if numeric), search for it. The patterns above cover most of what comes up in competition code.
