---
title: Strings and Text
track: python
order: 3
estimatedTime: 35
difficulty: beginner
---

# Strings and Text

Computers are famous for crunching numbers, but a huge amount of the world's data is **text**: names, messages, tweets, documents, code, DNA sequences, and the prompts you type into an AI chatbot. In Python, text is stored in a type called a **string** (`str`), and learning to work with strings is one of the most useful skills you'll build.

This lesson is longer than the others because strings have a lot of little tools. Don't try to memorize everything at once. Read through, run the examples, and come back to this page as a reference whenever you need it. By the end you'll be slicing text apart, gluing it together, and formatting it cleanly — the exact skills you'll need for Natural Language Processing (NLP), the branch of AI that deals with language.

---

## Creating strings

A string is any sequence of characters wrapped in quotes. Python gives you three ways to make one, and they exist for good reasons.

### Single and double quotes

Single quotes `'...'` and double quotes `"..."` work exactly the same. Pick whichever you like:

```python
name = 'Ada'
city = "London"
print(name)
print(city)
```

So why have both? Because sometimes your text *contains* a quote. If your string has an apostrophe, wrap it in double quotes. If it has double quotes inside, wrap it in single quotes:

```python
sentence = "It's a beautiful day"        # apostrophe inside, use double quotes
quote = 'She said "hello" to me'         # double quotes inside, use single quotes
print(sentence)
print(quote)
```

If you mismatch them, Python gets confused about where the string ends and throws a `SyntaxError`. So the outer quotes just need to be different from any quote inside the text.

### Triple quotes for multi-line text

For text that spans several lines, use three quotes (single or double) on each end:

```python
poem = """Roses are red,
Violets are blue,
Python is fun,
And so are you."""
print(poem)
```

Everything between the triple quotes is kept exactly as typed, including the line breaks. Triple-quoted strings are also used to write **docstrings** — documentation inside functions — which you'll meet in the functions lesson.

### The empty string

A string with nothing in it is still a string. It's called the **empty string**:

```python
blank = ""
print(len(blank))   # 0
```

Empty strings come up constantly when you're building up text piece by piece and starting from nothing.

---

## Strings are sequences: indexing

Here's the key idea that unlocks everything else: **a string is an ordered sequence of characters**. Each character has a position, and you can grab any character by its position number. That position number is called an **index**.

Python counts positions starting from **0**, not 1. This trips up every beginner, so let's be very explicit. Take the word `"Python"`:

```
 P   y   t   h   o   n
 0   1   2   3   4   5
```

The first character `P` is at index `0`. The second character `y` is at index `1`. And so on. To grab a character, put its index in square brackets:

```python
word = "Python"
print(word[0])   # P
print(word[1])   # y
print(word[2])   # t
print(word[5])   # n
```

> The number one thing to remember: **indexing starts at 0.** The first item is `[0]`, not `[1]`. This is true for strings, lists, arrays, and tensors — the entire Python and AI world counts from zero. Burn it into your memory now and you'll save yourself countless off-by-one bugs.

### Negative indexing — counting from the end

Python has a neat trick: negative indexes count backward from the end. `-1` is the last character, `-2` is the second-to-last, and so on:

```
 P    y    t    h    o    n
-6   -5   -4   -3   -2   -1
```

```python
word = "Python"
print(word[-1])   # n  (last character)
print(word[-2])   # o  (second to last)
print(word[-6])   # P  (same as word[0])
```

This is incredibly handy when you don't know how long a string is but you want the last character. `word[-1]` always gives you the last one, whether the word is 3 letters or 300.

### Going out of bounds

If you ask for an index that doesn't exist, Python raises an `IndexError`:

```python
word = "Python"
print(word[10])   # IndexError: string index out of range
```

There's no character at index 10 in a 6-letter word (the highest valid index is 5), so Python stops. When you see `IndexError`, you've asked for a position past the end of the string.

---

## Slicing: grabbing a piece of a string

Grabbing one character is useful, but often you want a whole chunk — a **substring**. This is called **slicing**, and the syntax is `string[start:stop]`.

The rule is: slicing includes the `start` index but **stops before** the `stop` index. The stop is *exclusive*. This feels strange at first but you'll get used to it.

```python
word = "Python"
print(word[0:3])   # Pyt  (indexes 0, 1, 2 — stops before 3)
print(word[2:5])   # tho  (indexes 2, 3, 4 — stops before 5)
```

Think of the numbers as the cuts between characters, and you're taking everything between the two cuts.

### Leaving out start or stop

If you leave out the start, Python assumes 0 (the beginning). If you leave out the stop, Python assumes the end:

```python
word = "Python"
print(word[:3])    # Pyt  (from the start up to index 3)
print(word[3:])    # hon  (from index 3 to the end)
print(word[:])     # Python  (the whole thing)
```

The pattern `word[3:]` reads as "everything from position 3 onward," and `word[:3]` reads as "everything up to position 3." These are extremely common.

### The step: slicing with a stride

Slicing has an optional third number: the **step**, written as `string[start:stop:step]`. The step controls how many characters to skip. A step of 2 takes every other character:

```python
word = "Python"
print(word[::2])   # Pto  (every 2nd character: P, t, o)
print(word[1::2])  # yhn  (every 2nd, starting at index 1)
```

And a step of `-1` reverses the string — a famous Python one-liner:

```python
word = "Python"
print(word[::-1])   # nohtyP  (reversed!)
```

That `[::-1]` trick for reversing shows up all the time. It reads as "whole string, but step backward."

> Slicing never causes an `IndexError`, even if the numbers go past the end. `"Python"[2:100]` just gives you everything from index 2 onward. Slicing is forgiving in a way that single-index access is not.

---

## Useful string functions and methods

Python comes with many built-in tools for working with strings. Some are functions you call *on* a string, like `len()`. Others are **methods** — functions that belong to the string and are called with a dot, like `text.upper()`. Let's go through the ones you'll use most.

### `len()` — how long is it?

`len()` gives you the number of characters in a string:

```python
print(len("Python"))   # 6
print(len("hi"))       # 2
print(len(""))         # 0
```

`len()` is one of the most-used functions in all of Python. You'll use it on strings, lists, arrays — anything with a length.

### `.upper()` and `.lower()` — changing case

These return a new string with all letters uppercased or lowercased:

```python
text = "Hello World"
print(text.upper())   # HELLO WORLD
print(text.lower())   # hello world
print(text)           # Hello World  (the original is unchanged!)
```

> Very important: string methods **never change the original string**. Strings are *immutable* — they can't be modified once created. Methods like `.upper()` return a brand-new string and leave the original alone. If you want to keep the result, you must store it: `text = text.upper()`.

Lowercasing is a real workhorse in NLP. When a computer compares words, `"Cat"` and `"cat"` look different because of the capital C. Converting everything to lowercase first — called **normalization** — makes the comparison fair. It's one of the first steps in almost every text-processing pipeline.

### `.strip()` — removing whitespace

`.strip()` removes spaces (and tabs and newlines) from the start and end of a string. This is essential when cleaning messy data, where text often has stray spaces:

```python
messy = "   hello   "
print(messy.strip())        # "hello"  (spaces gone)
print(len(messy))           # 11
print(len(messy.strip()))   # 5
```

There are also `.lstrip()` (left side only) and `.rstrip()` (right side only) if you need finer control.

### `.split()` — breaking a string into pieces

`.split()` chops a string into a **list** of smaller strings. By default it splits on spaces, which turns a sentence into a list of words:

```python
sentence = "the quick brown fox"
words = sentence.split()
print(words)        # ['the', 'quick', 'brown', 'fox']
print(len(words))   # 4
```

You can split on any character by passing it as an argument. Splitting on commas is common for data files:

```python
data = "Ada,15,London"
fields = data.split(",")
print(fields)   # ['Ada', '15', 'London']
```

Splitting text into words (or into smaller pieces) is the very first step in NLP — it's called **tokenization**, and modern AI language models do a fancy version of it before they can read anything.

### `.join()` — gluing pieces back together

`.join()` is the opposite of `.split()`. It takes a list of strings and glues them into one string, using the string you call it on as the "glue" between pieces:

```python
words = ['the', 'quick', 'brown', 'fox']
sentence = " ".join(words)
print(sentence)   # the quick brown fox

# Join with a different separator
print("-".join(words))   # the-quick-brown-fox
print("".join(words))    # thequickbrownfox
```

The syntax feels backward at first — you call `.join()` on the separator, not on the list. Read `" ".join(words)` as "join the words together with a space between each." Together, `.split()` and `.join()` let you take text apart, process the pieces, and put it back together.

### `.replace()` — swapping text

`.replace(old, new)` returns a new string with every occurrence of `old` swapped for `new`:

```python
text = "I like cats"
print(text.replace("cats", "dogs"))   # I like dogs
print("aaa".replace("a", "b"))        # bbb
```

### `.count()` and `.find()`

`.count()` tells you how many times something appears, and `.find()` tells you the index where it first appears (or `-1` if it's not there):

```python
text = "banana"
print(text.count("a"))    # 3
print(text.find("n"))     # 2  (first 'n' is at index 2)
print(text.find("z"))     # -1 (not found)
```

---

## The `in` operator — checking for substrings

To check whether one string appears inside another, use the `in` keyword. It gives you a boolean:

```python
text = "I love machine learning"
print("machine" in text)    # True
print("deep" in text)       # False
print("love" in text)       # True
```

This is clean and readable — it reads almost like English. You'll use `in` constantly to check things like "does this email contain the word 'free'?" (a spam-filter idea) or "does this review mention the product name?"

Be aware that `in` is case-sensitive: `"Machine" in text` would be `False` because of the capital M. This is exactly why lowercasing first (with `.lower()`) matters so much.

```python
text = "I Love Python"
print("love" in text)              # False (capital L in original)
print("love" in text.lower())      # True  (lowercased first)
```

---

## String multiplication and concatenation

Two operators that normally do math also work on strings, in intuitive ways.

**The `+` operator glues (concatenates) strings together:**

```python
first = "machine"
second = "learning"
print(first + " " + second)   # machine learning
```

Remember from the last lesson: `+` only works between two strings. Trying to add a number causes a `TypeError`. Convert the number first with `str()`, or use an f-string (coming up next).

**The `*` operator repeats a string:**

```python
print("ha" * 3)      # hahaha
print("=" * 20)      # ====================
print("-" * 10)      # ----------
```

String multiplication is genuinely useful for making separators and simple text layouts:

```python
print("=" * 30)
print("  Training Results")
print("=" * 30)
```

That prints a nice header with lines above and below it, without you having to count out thirty equals signs by hand.

---

## f-strings: the best way to format text

Very often you want to build a string that mixes fixed text with the values of variables — like `"Ada scored 92 points"` where the name and the number come from variables. The old way was clumsy, involving lots of `+` and `str()`. The modern way is the **f-string**, and it's wonderful.

Put the letter `f` right before the opening quote, then write `{variable}` anywhere inside the string. Python replaces each `{...}` with the value:

```python
name = "Ada"
score = 92
print(f"{name} scored {score} points")   # Ada scored 92 points
```

No `+`, no `str()` conversion — Python handles it all. This is the recommended way to build strings in modern Python, and you'll use it in almost every program from now on.

You can put any expression inside the braces, not just a variable name:

```python
a = 5
b = 3
print(f"{a} plus {b} equals {a + b}")   # 5 plus 3 equals 8

price = 9.5
quantity = 4
print(f"Total: {price * quantity} dollars")   # Total: 38.0 dollars
```

### Formatting numbers in f-strings

You can control how numbers display by adding a colon and a format code inside the braces. The most useful one is limiting decimal places, which matters a lot when printing things like accuracy or loss values in AI:

```python
accuracy = 0.87654321
print(f"Accuracy: {accuracy:.2f}")    # Accuracy: 0.88  (2 decimal places)
print(f"Accuracy: {accuracy:.4f}")    # Accuracy: 0.8765 (4 decimal places)

# Percentages
print(f"Accuracy: {accuracy:.1%}")    # Accuracy: 87.7%
```

The `.2f` means "show this as a float with 2 digits after the decimal point." When you're printing a model's accuracy every training step, you don't want to see `0.87654321987654` — `.2f` keeps your output clean and readable.

---

## Escape characters

What if you want a string to contain a character that's hard to type directly, like a newline, or a quote that matches your outer quotes? You use an **escape character**: a backslash `\` followed by a special letter. The backslash tells Python "the next character is special."

Here are the ones you'll actually use:

| Escape | Meaning | Example |
|---|---|---|
| `\n` | newline (line break) | `"line1\nline2"` |
| `\t` | tab | `"a\tb"` |
| `\\` | a literal backslash | `"C:\\Users"` |
| `\"` | a double quote | `"She said \"hi\""` |
| `\'` | a single quote | `'It\'s here'` |

Let's see them work:

```python
print("First line\nSecond line")
# First line
# Second line

print("Name:\tAda")
# Name:   Ada

print("Path: C:\\Users\\Ada")
# Path: C:\Users\Ada
```

The `\n` is the most important one. It's how you put a line break inside a single string. Notice that `print()` also adds a newline automatically at the end of every call — that's why each `print()` starts on a fresh line.

The double backslash `\\` matters for Windows file paths, which use single backslashes. Since a single `\` starts an escape, you need `\\` to get one literal backslash. (This is why you'll sometimes see paths written with forward slashes instead — Python accepts those too and they avoid the whole problem.)

> A common confusion: `\n` inside a string is a **single** newline character, even though it's typed as two characters (`\` and `n`). Once inside the string, Python treats the pair as one invisible line-break character. That's why `len("a\nb")` is 3, not 4.

---

## Why strings matter in AI (NLP)

Everything in this lesson is the groundwork for **Natural Language Processing** — the field of AI that teaches computers to understand and generate human language. Chatbots, translators, search engines, and text summarizers are all NLP.

But here's the catch: neural networks can only do math on numbers. They can't directly read the word `"cat"`. So the entire challenge of NLP starts with turning text into numbers, and every step uses the string skills you just learned:

1. **Normalization** — lowercasing (`.lower()`) and stripping (`.strip()`) so that `"Cat"`, `"cat"`, and `" cat "` are all treated the same.

2. **Tokenization** — splitting text into pieces (`.split()` is the simplest version) so each word or sub-word becomes a unit the model can handle. When you type a message to a large language model, the very first thing it does is chop your text into tokens.

3. **Cleaning** — using `.replace()` to remove junk, `.count()` to find patterns, and `in` to check for keywords. Real-world text is messy — full of typos, extra spaces, and weird symbols — and cleaning it is a big part of any NLP project.

4. **Building vocabulary** — counting how often each word appears (using splits and counts) to decide which words the model should know.

When you eventually train a language model, you'll be doing fancy versions of exactly these operations. The humble `.split()` and `.lower()` you learned today are the first rung on the ladder to understanding how AI reads. Strings really are the doorway to NLP.

---

## Common mistakes with strings

**Mistake 1: Forgetting that methods return a new string.**

```python
text = "hello"
text.upper()        # this result is thrown away!
print(text)         # still "hello"

text = text.upper() # store the result to keep it
print(text)         # "HELLO"
```

Because strings are immutable, `.upper()` can't change `text` in place. It hands you a new string, and if you don't catch it in a variable, it vanishes. This catches almost everyone.

**Mistake 2: Off-by-one with indexes.** Remember the last valid index is `len(s) - 1`, not `len(s)`. For `"Python"` (length 6), the last index is 5. Asking for `[6]` is an `IndexError`.

**Mistake 3: Mixing up `.split()` and `.join()`.** `.split()` turns a string into a list; `.join()` turns a list back into a string. And `.join()` is called on the *separator*, not the list: `" ".join(words)`, not `words.join(" ")`.

**Mistake 4: Forgetting `in` is case-sensitive.** `"cat" in "I have a Cat"` is `False`. Lowercase both sides first if case shouldn't matter.

---

## Putting it all together

Here's a small program that cleans and analyzes a piece of text, using tools from this whole lesson:

```python
review = "  The Movie Was AMAZING and I loved it  "

# Clean it up
cleaned = review.strip().lower()
print(f"Cleaned: '{cleaned}'")

# Split into words
words = cleaned.split()
print(f"Word count: {len(words)}")

# Check for a keyword
has_positive = "amazing" in cleaned
print(f"Mentions 'amazing': {has_positive}")

# Reverse the first word, just for fun
print(f"First word reversed: {words[0][::-1]}")

# Build a summary with an f-string
print(f"This review has {len(words)} words and the sentiment looks {'positive' if has_positive else 'unknown'}.")
```

Read through it and predict the output before running. You'll see stripping and lowercasing, splitting into words, `len()`, the `in` check, a reversing slice, and f-strings all working together — a miniature version of the text-cleaning you'll do in real NLP projects.

---

## Summary

- A **string** is text in quotes. Use single, double, or triple quotes; triple quotes span multiple lines.
- Strings are **sequences**: index them with `[i]` starting from **0**. Negative indexes count from the end (`[-1]` is last).
- **Slice** with `[start:stop:step]`. The stop is exclusive. `[::-1]` reverses a string.
- Key tools: `len()`, `.upper()`, `.lower()`, `.strip()`, `.split()`, `.join()`, `.replace()`, `.count()`, `.find()`.
- String methods **return a new string** — they never change the original (strings are immutable). Store the result.
- The `in` operator checks for substrings and returns a boolean. It's case-sensitive.
- `+` concatenates strings; `*` repeats them (`"=" * 20`).
- **f-strings** (`f"{name} scored {score}"`) are the modern way to mix variables into text. Use `{value:.2f}` to control decimals.
- Escape characters: `\n` (newline), `\t` (tab), `\\` (backslash), `\"` (quote).
- Strings are the foundation of **NLP**: normalization, tokenization, and cleaning all use these exact tools to turn text into something an AI model can work with.
