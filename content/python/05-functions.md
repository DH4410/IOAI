---
title: Functions
track: python
order: 5
estimatedTime: 40
difficulty: beginner
---

# Functions

Imagine you're writing a program and you need to calculate the average of some numbers in five different places. Would you copy-paste the same calculation five times? What happens when you find a bug in it — do you fix it five times and hope you didn't miss one? This is exactly the problem **functions** solve.

A function is a named, reusable piece of code. You write it once, give it a name, and then use it as many times as you like just by saying its name. Functions are how programmers avoid repeating themselves, keep code organized, and build big programs out of small, understandable pieces. In AI, entire models are built from functions and function-like objects — so this lesson is a big step toward real ML code.

You've already been using functions the whole time: `print()`, `len()`, `int()`, `type()` are all functions. Now you'll learn to build your own.

---

## Defining your first function

You create a function with the `def` keyword (short for "define"). Here's the simplest possible function:

```python
def greet():
    print("Hello!")
    print("Welcome to Python.")

greet()   # this runs the function
greet()   # you can call it again
```

Let's break down the parts:

- `def` tells Python "I'm defining a function."
- `greet` is the **name** you're giving it (same naming rules as variables).
- The empty parentheses `()` are where inputs go (none here, for now).
- The colon `:` starts the function's body, just like with `if` and `for`.
- The indented lines are the **body** — the code that runs when you call the function.

Defining a function does **not** run it. The lines inside just sit there, ready. The function only runs when you **call** it, by writing its name followed by parentheses: `greet()`. In the example above we call it twice, so both messages print twice.

> Key mental model: defining a function is like writing down a recipe. Calling the function is like actually cooking the recipe. Writing the recipe doesn't make dinner — you have to cook it. And you can cook the same recipe as many times as you want.

---

## Parameters and arguments: giving functions input

A function that always does exactly the same thing is limited. The real power comes from feeding a function different inputs each time. You do this with **parameters**.

A parameter is a variable listed inside the parentheses when you define the function. It acts as a placeholder for a value that will be supplied when the function is called:

```python
def greet(name):
    print(f"Hello, {name}!")

greet("Ada")     # Hello, Ada!
greet("Alan")    # Hello, Alan!
greet("Grace")   # Hello, Grace!
```

Here `name` is a parameter. When you call `greet("Ada")`, the value `"Ada"` gets stored in `name` for that run of the function. Call it with a different value and you get a different result. One function, endless uses.

There's a bit of vocabulary worth getting straight:

- A **parameter** is the name in the definition (`name` above). It's the placeholder.
- An **argument** is the actual value you pass in when calling (`"Ada"` above). It's what fills the placeholder.

People use these words a little loosely, and that's fine, but knowing the difference helps when reading error messages.

You can have several parameters, separated by commas. They fill in order:

```python
def introduce(name, age, city):
    print(f"{name} is {age} years old and lives in {city}.")

introduce("Ada", 20, "London")
# Ada is 20 years old and lives in London.
```

The first argument fills the first parameter, and so on. Order matters here — `introduce(20, "Ada", "London")` would put the number where the name should be and make a mess.

---

## Return values: getting output back

Printing is nice for showing things to a human, but usually you want a function to **compute a value and hand it back** so the rest of your program can use it. That's what `return` does.

```python
def add(a, b):
    return a + b

result = add(3, 5)
print(result)          # 8
print(add(10, 20))     # 30
print(add(1, 2) + add(3, 4))   # 3 + 7 = 10
```

When Python hits `return`, it stops the function and sends the value back to wherever the function was called. That returned value can be stored in a variable, printed, or used in more math — just like any other value.

### The crucial difference between `print` and `return`

Beginners mix these up constantly, so let's be clear. Compare:

```python
def add_print(a, b):
    print(a + b)      # shows the number, hands back nothing

def add_return(a, b):
    return a + b      # hands the number back

x = add_print(3, 5)   # prints 8, but x is None!
y = add_return(3, 5)  # prints nothing, but y is 8

print(x)   # None
print(y)   # 8
```

`print` displays something on the screen and is gone — the value can't be reused. `return` gives the value back to your program so you can keep working with it. A function with no `return` automatically returns a special value called `None`, which means "nothing."

> Rule of thumb: use `return` when the caller needs the result to do more work (which is almost always in real code), and use `print` only when you literally just want to show something to a person. For anything a model computes, you want `return`.

### `return` ends the function immediately

As soon as `return` runs, the function stops — any code after it is skipped:

```python
def check_positive(n):
    if n > 0:
        return "positive"
    return "not positive"   # only reached if n <= 0

print(check_positive(5))    # positive
print(check_positive(-3))   # not positive
```

This "return early" pattern is clean and common. You can return from several different spots in a function, and whichever one runs first wins.

---

## Default parameter values

Sometimes a parameter usually has the same value, and you don't want to type it every time. You can give a parameter a **default value** in the definition. If the caller doesn't supply that argument, the default is used:

```python
def greet(name, greeting="Hello"):
    print(f"{greeting}, {name}!")

greet("Ada")                    # Hello, Ada!   (uses default)
greet("Alan", "Good morning")   # Good morning, Alan!  (overrides default)
```

The `greeting="Hello"` says "if nobody gives me a greeting, use 'Hello'." Defaults make functions flexible without forcing callers to specify everything.

This shows up constantly in AI libraries. When you create a model, you might write `create_model(layers=3)` and let dozens of other settings (learning rate, activation function, etc.) fall back to sensible defaults. You only override the ones you care about.

> One rule: parameters **with** defaults must come **after** parameters without them. `def f(a, b=2)` is fine; `def f(a=1, b)` is a `SyntaxError`. Python needs the required ones first so it knows how to match arguments up.

---

## `*args` and `**kwargs`: flexible numbers of arguments

What if you don't know how many arguments a function will get? Say you want a function that adds up *any* number of values. Python has two special tools for this.

### `*args` — any number of positional arguments

Putting a `*` before a parameter name collects all extra positional arguments into a **tuple** (a list-like thing you'll meet properly soon):

```python
def add_all(*numbers):
    total = 0
    for n in numbers:
        total += n
    return total

print(add_all(1, 2))          # 3
print(add_all(1, 2, 3, 4))    # 10
print(add_all(5))             # 5
print(add_all())              # 0
```

Inside the function, `numbers` is a collection of everything that was passed, and you can loop over it. The name `args` is just a convention — the `*` is what does the work. Now one function handles two arguments or twenty.

### `**kwargs` — any number of named arguments

Two stars `**` collect extra **keyword** (named) arguments into a **dictionary** (also coming soon). Keyword arguments are the ones you pass with `name=value`:

```python
def describe(**info):
    for key, value in info.items():
        print(f"{key}: {value}")

describe(name="Ada", age=20, city="London")
# name: Ada
# age: 20
# city: London
```

Here `info` becomes a dictionary `{"name": "Ada", "age": 20, "city": "London"}`. Again, `kwargs` ("keyword arguments") is just the traditional name; the `**` is the magic.

You'll see `*args` and `**kwargs` all over real AI code, especially in library functions that need to accept many optional settings and pass them along. You don't have to master them now — just recognize the syntax when you see it and know it means "accept a flexible number of arguments."

---

## Variable scope: local vs global

Here's a concept that quietly causes many bugs if you don't understand it: **scope**. Scope is about *where* a variable exists and can be seen.

Variables created **inside** a function are **local** to that function. They exist only while the function runs, and the outside world can't see them:

```python
def calculate():
    result = 42       # local variable
    print(result)     # works — we're inside

calculate()           # prints 42
print(result)         # NameError! 'result' doesn't exist out here
```

The variable `result` lives and dies inside `calculate()`. Once the function finishes, it's gone. Trying to use it outside gives a `NameError`. This is actually a good thing — it means functions are self-contained and their internal variables can't accidentally clash with the rest of your program.

Variables created **outside** all functions are **global** — every function can read them:

```python
message = "Hello from outside"   # global variable

def show():
    print(message)   # can read the global just fine

show()   # Hello from outside
```

But there's a catch: a function can **read** a global variable, but if you try to **assign** to a name inside a function, Python treats it as a new local variable, not the global one:

```python
count = 0

def increment():
    count = count + 1   # ERROR: Python thinks count is local here

increment()   # UnboundLocalError
```

This confuses everyone the first time. To actually modify a global from inside a function, you'd use the `global` keyword — but honestly, doing so is usually a sign you should restructure your code. The cleaner approach is to pass values in as parameters and get results back with `return`:

```python
def increment(count):
    return count + 1

count = 0
count = increment(count)   # count is now 1
count = increment(count)   # count is now 2
print(count)               # 2
```

> Best practice: keep functions self-contained. Take what you need as parameters, return what you produce, and avoid reaching out to modify global variables. Code written this way is far easier to understand, test, and debug — which is exactly why it's the standard style in AI codebases.

---

## Lambda functions: tiny one-line functions

Sometimes you need a function so small and simple that writing a full `def` feels like overkill. For these, Python offers **lambda functions** — anonymous, one-line functions.

The syntax is `lambda parameters: expression`. The value of the expression is automatically returned (no `return` keyword needed):

```python
# A normal function
def square(x):
    return x * x

# The same thing as a lambda
square_lambda = lambda x: x * x

print(square(5))         # 25
print(square_lambda(5))  # 25
```

Both do exactly the same thing. The lambda is just more compact. Lambdas can take several parameters too:

```python
add = lambda a, b: a + b
print(add(3, 4))   # 7
```

Where lambdas really shine is when you need to hand a small function to *another* function. For example, sorting a list of names by their length:

```python
names = ["Ada", "Katherine", "Al", "Grace"]
names_sorted = sorted(names, key=lambda name: len(name))
print(names_sorted)   # ['Al', 'Ada', 'Grace', 'Katherine']
```

Here `sorted` needs to know *how* to measure each item, and the lambda tells it "measure by length." Writing a whole `def` for such a throwaway helper would be clumsy. You'll see this `key=lambda ...` pattern often when sorting and processing data. Don't overuse lambdas, though — if the logic is more than a simple expression, a normal `def` with a clear name is more readable.

---

## Recursion: functions that call themselves

A function is allowed to call itself. This is called **recursion**, and while it can feel mind-bending at first, it's an elegant way to solve problems that break down into smaller versions of themselves.

The classic example is the **factorial**: $n! = n \times (n-1) \times (n-2) \times \dots \times 1$. Notice that $n! = n \times (n-1)!$ — the factorial of $n$ is defined in terms of a smaller factorial. That's the recursive structure:

```python
def factorial(n):
    if n <= 1:          # base case: stop recursing
        return 1
    return n * factorial(n - 1)   # recursive case

print(factorial(5))   # 120  (5 * 4 * 3 * 2 * 1)
```

Every recursive function needs two parts:

1. A **base case** — a condition where it stops calling itself and returns a plain value. Here it's `n <= 1`. Without a base case, the function calls itself forever (like an infinite loop) and crashes.
2. A **recursive case** — where it calls itself with a smaller input, moving toward the base case.

Trace `factorial(3)`:
- `factorial(3)` returns `3 * factorial(2)`
- `factorial(2)` returns `2 * factorial(1)`
- `factorial(1)` hits the base case, returns `1`
- Now it unwinds: `2 * 1 = 2`, then `3 * 2 = 6`

> Every recursion needs a base case, or it never stops and Python eventually raises a `RecursionError`. Think of the base case as the "off switch." Anything you can do with recursion you can also do with a loop, so don't force it — but for naturally nested problems (like trees, or exploring branching possibilities), recursion can be the clearest tool. You'll see recursive structure again in decision trees, a classic ML model.

---

## Docstrings: documenting your functions

When you write a function, it's good practice to explain what it does right at the top, using a **docstring**. A docstring is a string (usually triple-quoted) placed as the first line inside the function body:

```python
def calculate_bmi(weight_kg, height_m):
    """Calculate Body Mass Index from weight in kg and height in meters.

    Returns the BMI as a float.
    """
    return weight_kg / (height_m ** 2)

print(calculate_bmi(70, 1.75))   # 22.86...
```

The docstring doesn't affect what the function does — Python ignores it when running. It's documentation for humans. But it's not just a comment: tools can read docstrings automatically. In fact, when you call `help(calculate_bmi)`, Python shows you the docstring. Every function in NumPy, pandas, and PyTorch has a docstring, which is how you can look up what any library function does without leaving your code.

Get in the habit of writing a one-line docstring for any function that isn't totally obvious. Say what it does, what it takes, and what it returns. Your future self and your teammates will thank you.

---

## Why functions matter in AI

Functions aren't just tidy — they're the fundamental building block of every serious program, and AI is no exception. Here's why they matter so much for machine learning:

**Reusability.** A model has to run the same calculations over and over — on the training data, the validation data, and the test data. You write the logic once as a function and reuse it everywhere. No copy-paste, no inconsistencies.

**Building blocks.** A neural network is literally a stack of functions, each transforming its input a little. You'll write functions like `forward()` (push data through the network) and `train_step()` (do one update), and compose them into a full training pipeline. Big AI systems are just many small functions wired together.

**Fixing bugs in one place.** When your data-cleaning logic lives in a single function, you fix a bug once and every part of your program that uses it is fixed instantly. This reliability is priceless when your experiments take hours to run.

**Parameters = experiments.** Because functions take parameters, you can run experiments just by changing arguments: `train(learning_rate=0.01)` versus `train(learning_rate=0.001)`. The function stays the same; the settings vary. This is how researchers systematically hunt for the best model.

**Everything is a function or built from one.** When you get to PyTorch, you'll see that even a whole neural network is essentially a big callable object with a `forward` function inside. The activation functions, loss functions, and optimizers all have "function" right in their names. The mental model you build here — inputs go in, a value comes out — is the mental model of all of machine learning. A model *is* a function that maps inputs to predictions.

Master functions now, and a huge amount of AI code will read naturally: it's functions calling functions, all the way down.

---

## Common mistakes with functions

**Mistake 1: Confusing `print` and `return`.** A function that `print`s a value returns `None`. If you need to use the result later, you must `return` it. `x = my_func()` only gives `x` a useful value if the function returns something.

**Mistake 2: Forgetting the parentheses when calling.** Writing `greet` refers to the function object; writing `greet()` actually runs it. This is a subtle but common slip.

**Mistake 3: Trying to use a local variable outside its function.** Local variables vanish when the function ends. Return them if you need them outside.

**Mistake 4: Wrong argument order.** Arguments fill parameters in order. `introduce(20, "Ada")` puts the number where the name belongs. Match the order, or use keyword arguments (`introduce(name="Ada", age=20)`).

**Mistake 5: No base case in recursion.** A recursive function with no stopping condition calls itself forever and crashes with `RecursionError`.

---

## Putting it all together

Here's a program that uses functions with parameters, defaults, return values, and a docstring to analyze exam scores:

```python
def average(numbers):
    """Return the average of a list of numbers."""
    return sum(numbers) / len(numbers)

def grade(score, pass_mark=60):
    """Return 'Pass' or 'Fail' for a score, with a default pass mark of 60."""
    if score >= pass_mark:
        return "Pass"
    return "Fail"

def report(name, scores):
    """Print a full report for one student."""
    avg = average(scores)
    status = grade(avg)
    print(f"{name}: average {avg:.1f} — {status}")
    return avg

# Use the functions
report("Ada", [95, 88, 92])
report("Alan", [55, 48, 60])

# Reuse average() on its own
class_scores = [78, 85, 90, 65]
print(f"Class average: {average(class_scores):.1f}")
```

Notice how `report` calls `average` and `grade` — small functions combining into a bigger one. That composition is the heart of good programming. Trace the output before running, then check.

---

## Function Concepts Sorter

```widget
{
  "type": "concept-sort",
  "title": "Which Python Function Concept Is This?",
  "categories": [
    { "name": "Parameters & return", "color": "#5B5BD6" },
    { "name": "Scope", "color": "#F97316" },
    { "name": "Advanced (args/lambda/recursion)", "color": "#22C55E" }
  ],
  "items": [
    { "text": "def add(a, b=0): — b has a default value", "category": "Parameters & return" },
    { "text": "Variables defined inside a function vanish when it ends", "category": "Scope" },
    { "text": "return result — hands a value back to the caller", "category": "Parameters & return" },
    { "text": "sorted(names, key=lambda x: x.lower())", "category": "Advanced (args/lambda/recursion)" },
    { "text": "def f(*args): — collect any number of positional arguments", "category": "Advanced (args/lambda/recursion)" },
    { "text": "A function calling itself with a smaller input until a base case", "category": "Advanced (args/lambda/recursion)" },
    { "text": "A function with no return statement gives back None", "category": "Parameters & return" },
    { "text": "Reading a global variable inside a function is fine; modifying it is risky", "category": "Scope" }
  ]
}
```

---

## Summary

- A **function** is reusable, named code. Define it with `def name(...):` and run it by calling `name(...)`.
- Defining ≠ running. A function only executes when you **call** it. You can call it as many times as you like.
- **Parameters** are placeholders in the definition; **arguments** are the actual values you pass. They fill in order.
- `return` hands a value back to the caller so it can be reused. `print` only shows something on screen. A function with no `return` gives back `None`. As soon as `return` runs, the function stops.
- **Default parameters** (`greeting="Hello"`) let callers skip arguments. Defaults must come after non-default parameters.
- `*args` collects extra positional arguments into a tuple; `**kwargs` collects extra named arguments into a dict.
- **Scope**: variables made inside a function are local and vanish when it ends. Prefer passing values in as parameters and returning results, rather than modifying globals.
- **Lambda** functions are tiny one-line functions, handy as the `key=` in sorting.
- **Recursion** is a function calling itself; it always needs a base case to stop.
- **Docstrings** (triple-quoted strings at the top of a function) document what it does; `help()` reads them.
- In AI, functions are the core building block: models are stacks of functions, experiments are just different arguments, and a model itself is essentially a function from inputs to predictions.
