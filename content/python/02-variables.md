---
title: Variables and Data Types
track: python
order: 2
estimatedTime: 30
difficulty: beginner
---

# Variables and Data Types

In the last lesson you used Python like a calculator. You typed `2 + 3` and Python gave you `5`. That's useful, but there was a problem: the answer vanished. Once Python printed it, it was gone. If you wanted to use that `5` again later, you'd have to type the whole calculation again.

Variables fix this. A **variable** is a name that holds a value so you can use it later. This is the single most important idea in all of programming, and it stays important all the way up to training giant neural networks. So let's take our time and really understand it.

---

## What is a variable?

Think of a variable as a labeled box. You put a value inside the box, and you write a name on the outside. Later, whenever you say the name, Python looks inside the box and gives you what's there.

Here's how you create one:

```python
age = 15
print(age)
```

Run that. It prints `15`.

Let's break down what happened:

- `age` is the **name** of the variable (the label on the box).
- `=` is the **assignment operator**. It means "put the thing on the right into the box on the left."
- `15` is the **value** we stored.

> Important: In Python, `=` does **not** mean "equals" like in math. It means "assign." The value on the right gets stored in the name on the left. We'll use `==` (two equals signs) later when we actually want to check if two things are equal.

Once you've stored a value, you can use the name anywhere you'd use the value:

```python
age = 15
print(age)
print(age + 5)
print(age * 2)
```

This prints `15`, then `20`, then `30`. The variable `age` still holds `15` the whole time — we're just reading it and doing math with the result.

### Why bother with variables?

Imagine you're calculating the area of a circle with radius 7.

Without a variable:

```python
print(3.14159 * 7 * 7)
```

With a variable:

```python
radius = 7
pi = 3.14159
area = pi * radius * radius
print(area)
```

The second version is longer, but it's much clearer. Anyone reading it can see exactly what each number means. And if you want to change the radius, you change it in one place. This matters enormously when your programs get big. A machine learning script might have a hundred numbers in it. Naming them is how you stay sane.

---

## Variables can change

The word "variable" comes from "vary" — the value can change over time. When you assign a new value, the old one is thrown away.

```python
score = 0
print(score)   # 0

score = 10
print(score)   # 10

score = 25
print(score)   # 25
```

Each `=` replaces what was in the box. After the last line, `score` is `25`. The earlier values `0` and `10` are gone.

A very common pattern is updating a variable using its own current value:

```python
score = 10
score = score + 5
print(score)   # 15
```

Read `score = score + 5` from the right side first. Python computes `score + 5`, which is `10 + 5 = 15`. Then it stores `15` back into `score`. This "read the current value, change it, store it back" pattern is exactly what happens millions of times when a model is learning — the model's numbers get nudged a little on every step.

Python gives you a shortcut for this:

```python
score = 10
score += 5    # same as score = score + 5
print(score)  # 15

score -= 3    # same as score = score - 3
print(score)  # 12

score *= 2    # same as score = score * 2
print(score)  # 24
```

These are called **compound assignment operators**. You'll see `+=` constantly in real code, especially for counting things and accumulating totals.

---

## The four basic data types

Every value in Python has a **type**. The type tells Python what kind of thing the value is and what you're allowed to do with it. There are four types you need to know first.

### 1. Integers (`int`) — whole numbers

An integer is a whole number with no decimal point. It can be positive, negative, or zero.

```python
apples = 5
temperature = -8
count = 0
big_number = 1000000
```

You can do math with integers, and if the result is whole, it stays an integer:

```python
print(5 + 3)    # 8
print(10 - 4)   # 6
print(6 * 7)    # 42
```

### 2. Floats (`float`) — decimal numbers

A float is a number with a decimal point. "Float" is short for "floating-point number," which is just the technical name for how computers store decimals.

```python
price = 9.99
height = 1.75
pi = 3.14159
temperature = -2.5
```

Any time you divide with `/`, you get a float — even if the numbers divide evenly:

```python
print(10 / 2)   # 5.0  (a float, notice the .0)
print(10 / 3)   # 3.3333333333333335
```

That `5.0` instead of `5` is Python's way of telling you "this is a float now." It surprises a lot of beginners.

> Floats are not perfectly precise. Try `print(0.1 + 0.2)` — you'll get `0.30000000000000004`, not `0.3`. This isn't a bug in Python; it's how decimals work in binary on every computer. It matters a lot in AI, where you're adding up millions of tiny floats and small errors can build up. For now, just know that floats are approximate.

### 3. Strings (`str`) — text

A string is text: letters, words, sentences, symbols. You create a string by wrapping text in quotes. You can use single or double quotes — Python treats them the same.

```python
name = "Ada"
greeting = 'Hello there'
empty = ""
```

Strings get their own full lesson next. For now, just know that anything in quotes is text, and text behaves differently from numbers:

```python
print("5" + "3")   # 53  (glued together, NOT 8!)
print(5 + 3)       # 8   (added, because these are numbers)
```

That difference — `"5"` versus `5` — trips up almost everyone at some point. The quotes change everything. `"5"` is the character five, like a letter. `5` is the number five that you can do math with.

### 4. Booleans (`bool`) — true or false

A boolean has only two possible values: `True` or `False`. Note the capital first letter — Python is picky about this. `true` (lowercase) will cause an error.

```python
is_raining = True
is_sunny = False
passed_exam = True
```

Booleans come from asking yes/no questions:

```python
print(5 > 3)     # True
print(10 == 20)  # False
print(7 < 2)     # False
```

Booleans are the foundation of decision-making in programs. In the next lesson on control flow, you'll use them constantly to decide what your code does. And in AI, every yes/no prediction — "is this email spam?", "is this a cat?" — comes down to a boolean or a number that becomes one.

---

## Checking a type with `type()`

Python has a built-in function called `type()` that tells you the type of any value. This is genuinely useful when you're confused about why something isn't working.

```python
print(type(5))        # <class 'int'>
print(type(3.14))     # <class 'float'>
print(type("hello"))  # <class 'str'>
print(type(True))     # <class 'bool'>
```

You can check the type of a variable too:

```python
x = 42
print(type(x))   # <class 'int'>

x = "now I am text"
print(type(x))   # <class 'str'>
```

Notice that `x` changed type when we reassigned it. Python doesn't lock a variable to one type — the type is a property of the value, not the name. This is called **dynamic typing**, and it's one of the reasons Python is so quick to write.

When you're debugging a program that misbehaves, printing `type(something)` is one of the first things experienced programmers do. Often the bug is "oh, this is a string when I thought it was a number."

---

## Type conversion (casting)

Sometimes you have a value of one type and you need it as another. For example, when a user types a number into a program, Python receives it as a **string**, not a number. To do math with it, you have to convert it. This process is called **type conversion** or **casting**.

Python gives you conversion functions named after the types:

| Function | Converts to | Example | Result |
|---|---|---|---|
| `int()` | integer | `int("42")` | `42` |
| `float()` | float | `float("3.14")` | `3.14` |
| `str()` | string | `str(99)` | `"99"` |
| `bool()` | boolean | `bool(1)` | `True` |

Here they are in action:

```python
# String to number
text = "100"
number = int(text)
print(number + 50)   # 150

# Number to string
count = 7
message = "You have " + str(count) + " new messages"
print(message)

# Float to int (this CHOPS OFF the decimal, it does not round)
print(int(3.9))    # 3, not 4!
print(int(3.2))    # 3

# Int to float
print(float(5))    # 5.0
```

> Watch out: `int(3.9)` gives `3`, not `4`. Converting a float to an int always throws away the decimal part — it never rounds. If you want proper rounding, use the `round()` function instead: `round(3.9)` gives `4`.

### When conversion fails

You can only convert things that make sense. Converting the string `"hello"` to an integer is impossible, so Python raises an error:

```python
int("hello")   # ValueError: invalid literal for int() with base 10: 'hello'
```

But a string that looks like a number converts fine:

```python
print(int("42"))     # 42, works
print(float("3.14")) # 3.14, works
```

This is one of the most common sources of errors in real programs, especially when reading data from files. A dataset for a machine learning competition might have the number `age` stored as text like `"25"`, and you'll need to convert it before you can compute an average. We'll do exactly this when we get to pandas.

---

## Rules for naming variables

Python has strict rules about what a variable name can be. Break a rule and you get an error. Here they are:

**The hard rules (must follow, or Python errors):**

1. Names can contain letters, digits, and underscores (`_`).
2. Names **cannot** start with a digit. `2fast` is illegal; `fast2` is fine.
3. Names cannot contain spaces. Use underscores instead: `my_score`, not `my score`.
4. Names cannot be one of Python's reserved keywords like `if`, `for`, `class`, `True`, `def`, `import`. These words already mean something to Python.

```python
# Legal names
age = 15
player_score = 100
temperature2 = 20.5
_hidden = "ok"

# Illegal names (these cause errors)
# 2fast = 10        # can't start with a digit
# my score = 10     # no spaces allowed
# class = "Biology" # 'class' is a reserved keyword
```

**The style conventions (not enforced, but everyone follows them):**

- Use lowercase letters with underscores between words. This style is called **snake_case**: `learning_rate`, `num_epochs`, `hidden_layer_size`. This is the standard for Python and you'll see it everywhere in AI code.
- Make names descriptive. `x` tells you nothing; `student_count` tells you everything. Your future self will thank you.
- Avoid single letters except for short loops or well-known math symbols (like `x` and `y` for coordinates, or `n` for a count).

```python
# Bad — what do these even mean?
a = 60
b = 1.75
c = a / (b * b)

# Good — instantly readable
weight_kg = 60
height_m = 1.75
bmi = weight_kg / (height_m * height_m)
```

Both versions calculate the exact same thing. Python doesn't care which you write. But a human reading the second one understands it immediately. Good names are a gift to whoever reads your code next — and that person is usually you, a week later, having forgotten everything.

---

## Multiple assignment

Python lets you assign several variables in one line. This is a handy shortcut you'll see often.

```python
x, y, z = 1, 2, 3
print(x)   # 1
print(y)   # 2
print(z)   # 3
```

Python matches them up in order: the first name gets the first value, and so on. The number of names must match the number of values, or you get an error.

You can also give several variables the **same** value at once:

```python
a = b = c = 0
print(a, b, c)   # 0 0 0
```

This is a common way to reset several counters to zero at the start of a program.

One elegant trick multiple assignment enables is **swapping** two variables without needing a temporary one:

```python
left = "apple"
right = "banana"

left, right = right, left

print(left)    # banana
print(right)   # apple
```

In many other languages, swapping requires a third temporary variable. In Python, one line does it. This pattern shows up in sorting algorithms and data shuffling — both very common in machine learning when you randomize your training data.

---

## Constants (the UPPERCASE convention)

Sometimes you have a value that should never change while the program runs — like the number of classes in your dataset, or a fixed learning rate. Python doesn't have a way to truly lock a variable (you technically *can* still change it), but programmers have a convention to signal "don't touch this": write the name in **ALL CAPITAL LETTERS**.

```python
PI = 3.14159
MAX_PLAYERS = 4
LEARNING_RATE = 0.01
NUM_CLASSES = 10

# Later in the code...
area = PI * radius * radius
```

When you see an all-caps name, it's a message from the programmer: "this is a constant, a setting, a fixed value — I never intend to change it." Python won't stop you from changing it, but breaking the convention would confuse everyone (including you).

In AI code, constants are everywhere. You'll define things like `BATCH_SIZE = 32`, `NUM_EPOCHS = 100`, and `IMG_SIZE = 224` at the top of your scripts. Keeping these settings named and grouped makes experiments easy to adjust — you change one line at the top instead of hunting through hundreds of lines of code.

---

## Why types matter in AI

You might wonder why we're spending so much time on types. Here's the payoff.

In AI, all your data lives in structures called **tensors** — think of them as big grids of numbers. Every tensor has a type, called its **dtype** (data type). And the type is not a minor detail; it affects speed, memory, and even whether your model works at all.

Here are real examples of why type awareness matters:

- **Images** are usually stored as integers from 0 to 255 (one number per color per pixel). But neural networks want floats between 0 and 1. So the very first step of almost every image pipeline is converting integers to floats and dividing by 255. If you forget the conversion, your model gets garbage input.

- **Labels** (the correct answers, like "cat" or "dog") are often stored as integers: cat is `0`, dog is `1`. The model works with these integer codes, not the words.

- **Memory and speed**: A `float64` (64-bit float) uses twice the memory of a `float32`. Big models use `float32` or even `float16` to fit in memory and train faster. Choosing the right float type is a real decision AI engineers make.

- **Mixing types causes bugs**: If part of your data is a string like `"25"` and part is a number `25`, calculations break. A huge amount of "cleaning data" before training is just fixing types.

You don't need to memorize any of this now. The point is: the humble `int`, `float`, `str`, and `bool` you learned today are the foundation of everything. Tensors are just millions of typed numbers stacked together. Get comfortable with types now, and the AI parts will feel natural later.

---

## Common errors and how to fix them

Let's look at the two errors you'll hit most often with variables, so you recognize them when they appear.

### NameError — using a name that doesn't exist

If you use a variable before creating it (or you misspell its name), Python doesn't know what you mean and raises a `NameError`.

```python
print(scpre)   # NameError: name 'scpre' is not defined
```

Here we typed `scpre` instead of `score`. Python looked for a box labeled `scpre`, found nothing, and gave up. The fix is to check your spelling and make sure you assigned the variable before using it.

```python
score = 100
print(score)   # works fine now
```

> `NameError` almost always means one of two things: a typo, or you're using a variable before you created it. Read the name in the error message carefully — Python tells you exactly which name it couldn't find.

### TypeError — mixing incompatible types

Some operations don't make sense between certain types, and Python refuses to guess what you meant. Adding a number to a string is the classic example:

```python
age = 15
print("I am " + age + " years old")   # TypeError!
```

This fails with `TypeError: can only concatenate str (not "int") to str`. Python doesn't know whether you want to glue the number on as text or... it just refuses. The fix is to convert the number to a string first:

```python
age = 15
print("I am " + str(age) + " years old")   # works: I am 15 years old
```

Or, better, use an f-string (which you'll learn next lesson) so you don't have to convert manually:

```python
age = 15
print(f"I am {age} years old")   # works, and cleaner
```

`TypeError` is Python telling you "these two types don't go together in this operation." When you see it, look at what types you're combining and whether you need a conversion.

---

## Putting it all together

Let's write a small program that uses everything from this lesson. It calculates a student's average test score.

```python
# Constants (settings that don't change)
NUM_TESTS = 3

# The scores (integers)
test1 = 85
test2 = 92
test3 = 78

# Calculate the average (this will be a float because of /)
total = test1 + test2 + test3
average = total / NUM_TESTS

# Check the type
print(type(average))   # <class 'float'>

# Build a message (convert the number to a string to join it)
name = "Ada"
message = name + " scored an average of " + str(average)
print(message)

# A boolean: did they pass? (passing is 60 or above)
passed = average >= 60
print("Passed:", passed)
```

Trace through it and predict the output before you run it. You should see the type, then the message with the average, then `Passed: True`. Every concept from this lesson is in there: integers, a float from division, a constant, string conversion, and a boolean from a comparison.

---

## Python Data Types Sorter

```widget
{
  "type": "concept-sort",
  "title": "What Type Would Python Give This Value?",
  "categories": [
    { "name": "int", "color": "#5B5BD6" },
    { "name": "float", "color": "#F97316" },
    { "name": "str", "color": "#22C55E" },
    { "name": "bool", "color": "#EF4444" }
  ],
  "items": [
    { "text": "42", "category": "int" },
    { "text": "3.14", "category": "float" },
    { "text": "'hello'", "category": "str" },
    { "text": "True", "category": "bool" },
    { "text": "10 / 2   (division always returns this)", "category": "float" },
    { "text": "5 == 5   (a comparison result)", "category": "bool" },
    { "text": "str(100)   (converting a number to text)", "category": "str" },
    { "text": "10 // 3   (floor division)", "category": "int" }
  ]
}
```

---

## Summary

- A **variable** is a named box that stores a value so you can reuse it. Create one with `name = value`.
- `=` means "assign" (store the right side into the left), not "equals." That's `==`.
- The four basic types are **int** (whole numbers), **float** (decimals), **str** (text in quotes), and **bool** (`True`/`False`).
- `type(x)` tells you the type of a value — great for debugging.
- Convert between types with `int()`, `float()`, `str()`, `bool()`. Converting a float to an int chops off the decimal (it doesn't round).
- Name rules: letters, digits, underscores; can't start with a digit; no spaces; no keywords. Use `snake_case` and descriptive names.
- Assign several variables at once with `x, y = 1, 2`. Swap with `a, b = b, a`.
- Constants are written in `UPPERCASE` by convention to signal "don't change this."
- Types are the foundation of AI: tensors are just huge grids of typed numbers, and choosing the right type affects speed, memory, and correctness.
- `NameError` = the name doesn't exist (typo or used too early). `TypeError` = you combined types that don't mix (convert first).
