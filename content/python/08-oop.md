---
title: Classes and Object-Oriented Programming
track: python
order: 8
estimatedTime: 45
difficulty: intermediate
---

# Classes and Object-Oriented Programming

You've come a long way: variables, strings, control flow, functions, and collections. Now we reach a bigger idea that ties everything together — **Object-Oriented Programming**, or **OOP**. This is the concept that lets you build large, organized programs, and it's absolutely essential for AI, because the most important AI library, PyTorch, is built entirely around classes. When you eventually write `class MyModel(nn.Module):`, you'll be using exactly what this lesson teaches.

OOP feels abstract at first. Don't worry if it takes a couple of reads. We'll build it up slowly with concrete examples, and by the end you'll see why it exists and why AI code is written this way.

---

## Why does OOP exist? The problem it solves

Imagine you're modeling a dog in a program. A dog has some **data** — a name, an age, a breed — and it can **do things** — bark, sit, eat. So far you'd store the data in variables and write functions for the actions:

```python
dog_name = "Rex"
dog_age = 3

def bark(name):
    print(f"{name} says Woof!")

bark(dog_name)   # Rex says Woof!
```

This works for one dog. But what about a hundred dogs? You'd have `dog1_name`, `dog2_name`, `dog3_age`... a tangled mess of loosely related variables, and functions that you have to carefully feed the right pieces to. The data and the actions that go together are scattered all over your program. It becomes impossible to manage.

**OOP's big idea:** bundle the data and the actions that belong together into a single unit called an **object**. A dog object holds its own name and age *and* knows how to bark. Everything about a dog lives in one tidy package. When you have a hundred dogs, you have a hundred neat objects, each managing itself, instead of a swamp of variables.

This bundling is called **encapsulation**, and it's the heart of OOP. It's how you keep big programs organized and understandable.

---

## Classes and objects: blueprint vs thing

Two words you must keep straight:

- A **class** is a **blueprint** — a template that describes what a type of thing looks like and what it can do. `Dog` (the general idea of a dog) is a class.
- An **object** (also called an **instance**) is an actual **thing** built from that blueprint. Rex, a specific dog, is an object.

The analogy people use: a class is like the blueprint for a house, and objects are the actual houses built from it. One blueprint, many houses. Each house is separate — repainting one doesn't affect the others — but they all share the same design.

You've been using objects all along without knowing it. A string is an object (it has methods like `.upper()`); a list is an object (it has `.append()`). The `str` and `list` types are classes. Now you'll build your own.

---

## Your first class

You define a class with the `class` keyword. Here's a minimal `Dog` class:

```python
class Dog:
    def bark(self):
        print("Woof!")

# Create an object (instance) from the class
my_dog = Dog()
my_dog.bark()    # Woof!
```

Let's unpack this:

- `class Dog:` starts the blueprint. Class names conventionally use `CapitalizedWords` (called PascalCase), unlike variables and functions which use `snake_case`. This visual difference helps you spot classes instantly.
- Inside, `bark` is a **method** — a function that belongs to the class. Methods define what objects can *do*.
- `my_dog = Dog()` creates an object. The parentheses are you "calling" the class to build a new instance.
- `my_dog.bark()` calls the method on that specific object, using dot notation.

We'll explain the mysterious `self` parameter very soon — hold that thought.

---

## `__init__`: setting up a new object

An empty dog isn't very useful. We want each dog to have its own name and age from the moment it's created. That's the job of a special method called `__init__` (that's two underscores on each side — say it "dunder init," short for "double underscore").

`__init__` runs **automatically** every time you create an object. It's where you set up the object's starting data:

```python
class Dog:
    def __init__(self, name, age):
        self.name = name    # store the name on this object
        self.age = age      # store the age on this object

# When we create a Dog, __init__ runs with the arguments we pass
rex = Dog("Rex", 3)
bella = Dog("Bella", 5)

print(rex.name)    # Rex
print(rex.age)     # 3
print(bella.name)  # Bella
```

Notice how `Dog("Rex", 3)` passes `"Rex"` and `3` straight into `__init__`. Those get stored on the object. Now `rex` and `bella` are two separate objects, each carrying its own data. `rex.name` is `"Rex"`, `bella.name` is `"Bella"` — completely independent, just like separate houses from one blueprint.

The variables stored on an object (`self.name`, `self.age`) are called **instance attributes**. They're the object's personal data. You read them with dot notation: `rex.name`.

> `__init__` is often called the "constructor" because it constructs (sets up) a new object. You never call it directly — Python calls it for you whenever you write `Dog(...)`. Almost every class you write will have an `__init__`.

---

## `self`: the most confusing part, explained clearly

Now let's tackle `self`, which trips up every single beginner. Look at those method definitions — `def __init__(self, ...)`, `def bark(self)` — they all have `self` as the first parameter. But when you call them, you don't pass a `self`: you write `rex.bark()`, not `rex.bark(rex)`. What's going on?

Here's the key insight: **`self` refers to the specific object the method is being called on.** When you write `rex.bark()`, Python automatically passes `rex` in as `self` behind the scenes. So inside the method, `self` *is* `rex`. When you call `bella.bark()`, `self` *is* `bella`.

This is how one method can work on many objects. The method code is written once (in the class), but `self` tells it *which* object's data to use each time:

```python
class Dog:
    def __init__(self, name):
        self.name = name

    def introduce(self):
        print(f"Hi, I'm {self.name}")

rex = Dog("Rex")
bella = Dog("Bella")

rex.introduce()    # Hi, I'm Rex    (self is rex, so self.name is "Rex")
bella.introduce()  # Hi, I'm Bella  (self is bella, so self.name is "Bella")
```

Same `introduce` method, two different results — because `self` points to a different object each time. Read `self.name` as "*this particular object's* name."

The rules to remember:

- Every method's first parameter is `self` (you write it in the definition).
- You never pass `self` yourself — Python fills it in from whatever object you called the method on.
- Inside a method, use `self.something` to read or set that object's own data.
- Forgetting `self` in a method definition is one of the most common beginner errors, and gives a confusing "takes 0 positional arguments but 1 was given" message.

> Think of `self` as the word "my" from the object's point of view. When Rex runs `introduce`, `self.name` means "my name," and Rex's name is "Rex." It's the object talking about itself.

---

## Methods that use the object's data

Methods become powerful when they combine the object's own data with logic. Let's build a more complete class — a bank account — where methods change and read the object's state:

```python
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount
        print(f"Deposited {amount}. New balance: {self.balance}")

    def withdraw(self, amount):
        if amount > self.balance:
            print("Insufficient funds!")
        else:
            self.balance -= amount
            print(f"Withdrew {amount}. New balance: {self.balance}")

# Use it
account = BankAccount("Ada", 100)
account.deposit(50)    # Deposited 50. New balance: 150
account.withdraw(30)   # Withdrew 30. New balance: 120
account.withdraw(200)  # Insufficient funds!
print(account.balance) # 120
```

See how `deposit` and `withdraw` read and update `self.balance` — the object's own money. Each account keeps track of its own balance. The methods bundle the *behavior* (deposit, withdraw) right alongside the *data* (balance) it acts on. That's encapsulation in action, and it's exactly why OOP keeps big programs manageable: everything about a bank account lives inside the `BankAccount` class.

Notice too that `withdraw` uses an `if` statement — methods are just functions, so everything you learned about control flow, loops, and logic works inside them.

---

## Class attributes vs instance attributes

There are two kinds of attributes, and the difference matters.

- **Instance attributes** (`self.name`) belong to each individual object. Every object has its own copy. Rex's name and Bella's name are separate.
- **Class attributes** are defined directly in the class body (not inside `__init__`) and are **shared by all objects** of that class. There's just one copy, common to everyone.

```python
class Dog:
    species = "Canis familiaris"   # class attribute — shared by ALL dogs

    def __init__(self, name):
        self.name = name           # instance attribute — unique per dog

rex = Dog("Rex")
bella = Dog("Bella")

print(rex.name)       # Rex     (unique)
print(bella.name)     # Bella   (unique)
print(rex.species)    # Canis familiaris  (shared)
print(bella.species)  # Canis familiaris  (same shared value)
```

Use a **class attribute** for something that's the same for every object — like the species of all dogs, or a fixed configuration value. Use an **instance attribute** for something that varies per object — like each dog's name. A common real use of class attributes is a shared counter or constant, or default settings that apply across all instances.

---

## Inheritance: building on existing classes

Here's where OOP gets really powerful. **Inheritance** lets you create a new class based on an existing one. The new class (the **child** or **subclass**) automatically gets all the attributes and methods of the existing one (the **parent** or **base class**), and can add its own or change things.

Why is this useful? It avoids repetition. Suppose you have an `Animal` class with an `eat` method, and you want `Dog` and `Cat` classes. Both animals eat, so instead of writing `eat` twice, you write it once in `Animal` and have `Dog` and `Cat` inherit it:

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def eat(self):
        print(f"{self.name} is eating.")

class Dog(Animal):          # Dog inherits from Animal
    def speak(self):
        print(f"{self.name} says Woof!")

class Cat(Animal):          # Cat inherits from Animal
    def speak(self):
        print(f"{self.name} says Meow!")

rex = Dog("Rex")
rex.eat()      # Rex is eating.   (inherited from Animal!)
rex.speak()    # Rex says Woof!   (Dog's own method)

whiskers = Cat("Whiskers")
whiskers.eat()    # Whiskers is eating.  (inherited)
whiskers.speak()  # Whiskers says Meow!
```

The `class Dog(Animal):` syntax means "Dog is a kind of Animal." `Dog` gets `__init__` and `eat` for free from `Animal`, and adds its own `speak`. We didn't rewrite the eating logic — we reused it. This is the DRY principle: **Don't Repeat Yourself**.

Inheritance models "is-a" relationships: a Dog *is an* Animal, a Cat *is an* Animal. Whenever you have several types that share common behavior but differ in specifics, inheritance lets you put the shared part in a parent and the specifics in children.

> This is the single most important OOP concept for AI. In PyTorch, every neural network you build inherits from a base class called `nn.Module`. You write `class MyModel(nn.Module):`, and your model automatically gets all of PyTorch's machinery for tracking parameters, moving to a GPU, saving, and loading — for free, via inheritance. You just add the parts specific to your model. Everything in this section is preparing you for that exact pattern.

---

## `__str__` and `__repr__`: making objects printable

By default, printing an object gives you something ugly and unhelpful:

```python
class Dog:
    def __init__(self, name):
        self.name = name

rex = Dog("Rex")
print(rex)   # <__main__.Dog object at 0x7f8b1c0d3a90>  — useless!
```

That memory-address gibberish tells you nothing. To make your objects print nicely, define the special method `__str__`, which returns the string you want shown when the object is printed:

```python
class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return f"Dog(name={self.name}, age={self.age})"

rex = Dog("Rex", 3)
print(rex)   # Dog(name=Rex, age=3)   — much better!
```

Now `print(rex)` shows a readable description. `__str__` is meant for a friendly, human-readable message.

There's a close cousin, `__repr__`, which is meant for a more precise, developer-facing representation (ideally one you could copy-paste to recreate the object). If you only define one, define `__repr__`, because Python falls back to it in more situations (like when an object appears inside a list). For most learning purposes, a `__str__` is enough to make debugging pleasant. These "dunder" methods (`__init__`, `__str__`, `__repr__`) are how you hook your class into Python's built-in behaviors.

---

## Why OOP matters in AI: PyTorch is all classes

Let's make the payoff concrete. Here's roughly what a real PyTorch model looks like (simplified — you won't run this now, but read it):

```python
# This is what real PyTorch code looks like
class MyNeuralNet(nn.Module):        # inherits from nn.Module
    def __init__(self):
        super().__init__()           # set up the parent's machinery
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):            # a method describing the computation
        x = self.layer1(x)
        x = self.layer2(x)
        return x

model = MyNeuralNet()                # create an instance (an object)
output = model(input_data)           # use it
```

Look at how much of this you now recognize:

- `class MyNeuralNet(nn.Module):` — a class using **inheritance** from `nn.Module`.
- `def __init__(self):` — the **constructor**, setting up the model's layers as **instance attributes** (`self.layer1`, `self.layer2`).
- `def forward(self, x):` — a **method** that uses `self` to access the layers it set up.
- `model = MyNeuralNet()` — creating an **object** from the class.

Every single one of those concepts is from this lesson. The layers are objects too (`nn.Linear` is a class). Training loops, datasets, optimizers, loss functions — in PyTorch they're all classes and objects. This is why OOP isn't optional for AI: the entire framework is built on it. Once classes click for you, PyTorch code stops looking like magic and starts looking familiar. You'll read `class Model(nn.Module):` and think, "ah, a subclass with an `__init__` and a method — I know exactly what that is."

That's the goal. You don't need to master every corner of OOP today. You need the core: classes are blueprints, objects are instances, `__init__` sets up data, `self` means "this object," methods bundle behavior with data, and inheritance lets one class build on another. With those, the AI road ahead is open.

---

## Common mistakes with classes

**Mistake 1: Forgetting `self` in a method definition.** `def bark():` should be `def bark(self):`. Without `self`, calling the method gives a confusing argument-count error.

**Mistake 2: Forgetting `self.` when accessing attributes.** Inside a method, `name` refers to a local variable, but `self.name` refers to the object's attribute. To use the object's data, you need the `self.` prefix.

**Mistake 3: Confusing the class with an object.** `Dog` is the blueprint; `Dog()` creates an actual object. You call methods on objects (`rex.bark()`), not usually on the class itself.

**Mistake 4: Calling `__init__` directly.** You never write `rex.__init__(...)` yourself. Python runs `__init__` automatically when you create the object with `Dog(...)`.

**Mistake 5: Putting per-object data in a class attribute.** If you define `name = ""` in the class body instead of setting `self.name` in `__init__`, all objects share one name. Per-object data belongs in `__init__` as `self.name`.

---

## Putting it all together

Here's a complete example using `__init__`, methods, a class attribute, inheritance, and `__str__` — a small model of students and honor students:

```python
class Student:
    school = "IOAI Academy"          # class attribute — shared by all

    def __init__(self, name):
        self.name = name
        self.grades = []             # each student has their own list

    def add_grade(self, grade):
        self.grades.append(grade)

    def average(self):
        if not self.grades:
            return 0
        return sum(self.grades) / len(self.grades)

    def __str__(self):
        return f"{self.name} (avg: {self.average():.1f})"

class HonorStudent(Student):         # inherits everything from Student
    def add_grade(self, grade):
        # Honor students get a 5-point bonus on each grade
        self.grades.append(grade + 5)

# Regular student
ada = Student("Ada")
ada.add_grade(80)
ada.add_grade(90)
print(ada)                # Ada (avg: 85.0)
print(ada.school)         # IOAI Academy  (class attribute)

# Honor student inherits everything but overrides add_grade
alan = HonorStudent("Alan")
alan.add_grade(80)        # stored as 85 (bonus)
alan.add_grade(90)        # stored as 95 (bonus)
print(alan)               # Alan (avg: 90.0)  — uses inherited __str__ and average
```

Trace through it. `HonorStudent` reuses `__init__`, `average`, and `__str__` from `Student` (inheritance), but replaces `add_grade` with its own bonus version. That mix of reuse and customization is the whole point of OOP — and it's exactly how you'll build models on top of PyTorch's base classes.

---

## Sort OOP Concepts

```widget
{
  "type": "concept-sort",
  "title": "Class Attribute or Instance Attribute?",
  "categories": [
    { "name": "Class attribute (shared by all instances)", "color": "#5B5BD6" },
    { "name": "Instance attribute (unique per object)", "color": "#F97316" }
  ],
  "items": [
    { "text": "Defined directly in the class body (outside __init__)", "category": "Class attribute (shared by all instances)" },
    { "text": "self.x = ... in __init__", "category": "Instance attribute (unique per object)" },
    { "text": "Changing it affects all objects of that class", "category": "Class attribute (shared by all instances)" },
    { "text": "Each object stores its own copy", "category": "Instance attribute (unique per object)" },
    { "text": "Dog.species = 'Canis lupus' (same for all dogs)", "category": "Class attribute (shared by all instances)" },
    { "text": "self.name = name (each dog has its own name)", "category": "Instance attribute (unique per object)" }
  ]
}
```

---

## Summary

- **OOP** bundles related data and behavior into **objects**, keeping large programs organized. This bundling is called **encapsulation**.
- A **class** is a blueprint; an **object** (instance) is a thing built from it. Define a class with `class Name:` (PascalCase names).
- `__init__` is the constructor: it runs automatically when you create an object and sets up its data (instance attributes) via `self`.
- `self` refers to the specific object a method is called on. It's how one method works on many objects. You always write `self` as the first parameter but never pass it yourself.
- **Instance attributes** (`self.x`) are unique per object; **class attributes** (defined in the class body) are shared by all objects.
- **Inheritance** (`class Child(Parent):`) lets a new class reuse and extend an existing one — the key to avoiding repetition and the basis of PyTorch models.
- `__str__` (and `__repr__`) make your objects print as readable text instead of memory addresses.
- **AI connection:** PyTorch is built entirely on classes. Your models will be subclasses of `nn.Module` with an `__init__` and a `forward` method — exactly the pattern from this lesson. Master classes, and PyTorch code becomes readable.
