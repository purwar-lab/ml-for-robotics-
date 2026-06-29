# Chapter 1: Python Programming Fundamentals

---

**Python** — A readable programming language used heavily in machine learning, robotics tooling, data analysis, and automation.

---

## Why Python & Setting Up Colab


---

### Why Python & Setting Up Colab
**Concept.** Python is the default language of machine learning because the syntax is readable and the ecosystem is huge. NumPy, Pandas, scikit-learn, TensorFlow, PyTorch, OpenCV, and nearly every robotics ML example you find online will have Python support.
**C++**
Same idea in C++
```cpp
#include <iostream>
int main() {
  std::cout << "Hello, Robot!" << std::endl;
  return 0;
}
```
**Python**
Your first Python cell
```python
print("Hello, Robot!")
```
!!! tip "Mini Exercise"
    Run `print("Hello, Robot!")` in Colab. Then change the message to include your name and your robot's name.
!!! warning "Common Mistake"
    Do not worry about installing Python on your laptop yet. For this course, run the cells in Colab so everyone starts from the same environment.

---

## Variables & Data Types


---

### Python vs Other Languages: Three Things That Will Surprise You

#### No curly braces: Python uses indentation
In C, C++, or Java, blocks of code are wrapped in curly braces `{ }`. In Python, indentation — the spaces at the start of a line — is the structure. A block starts when indentation increases and ends when it goes back.

C++ uses braces
```cpp
if (battery < 10) {
    std::cout << "Low battery";
    returnToBase();
}
```

Python uses indentation
```python
if battery < 10:
    print("Low battery")
    return_to_base()
```

**Rule:** Always use 4 spaces for each level of indentation. Do not mix spaces and tabs. Python will throw an `IndentationError`.

!!! warning "IndentationError is common"
    If Python says `unexpected indent` or `expected an indented block`, look at the spaces at the start of the offending line.

#### No semicolons at the end of lines
Python does not need a semicolon at the end of each statement. One line equals one statement. Just press Enter and start the next line.

#### No type declarations: Python figures out the type itself
In C++ you write `int wheels = 4;`. In Python you write `wheels = 4`. Python looks at the value `4` and knows it is an integer automatically. This is called dynamic typing. You never need to write the type name when creating a variable. However, Python still has types. You just do not have to declare them. You can always check with `print(type(wheels))`, which prints `<class 'int'>`.

**Variable** — A named container in memory that holds a value. Think of it as a labelled box for robot data.


### Variables & Data Types
**Concept.** A variable is a named box that holds a value. Python has four beginner-friendly basic types: `int` for whole numbers, `float` for decimals, `str` for text, and `bool` for true/false values.
[Open in Colab →](https://colab.research.google.com/github/purwar-lab/ml-for-robotics-/blob/main/notebooks/ch1-variables.ipynb)
!!! info "Colab notebook is pre-filled"
    The Colab notebook contains every code cell from this lesson in order, exactly as shown on this page. It opens pre-filled so you can run cells top to bottom, modify them, and save your own copy to Google Drive with **File → Save a copy in Drive**
Robot sensor variables
```python
# Robot sensor variables
distance = 2.45          # float - distance in meters
is_obstacle = True       # bool - obstacle detected?
robot_name = "ARIA"      # str - robot's name
wheels = 4               # int - number of wheels

print(type(distance))
print(type(is_obstacle))
print(f"{robot_name} sees an obstacle at {distance:.1f}m: {is_obstacle}")
```
| Line | What it does |
| --- | --- |
| `distance = 2.45` | Creates a variable named `distance` and stores a decimal number, which Python calls a `float`. |
| `is_obstacle = True` | Stores a boolean value. Booleans are either `True` or `False`. |
| `robot_name = "ARIA"` | Stores text as a string. Strings are wrapped in quotes. |
| `wheels = 4` | Stores a whole number, which Python calls an `int`. |
| `print(type(distance))` | Prints the Python type of the value stored in `distance`. |
| `f"{robot_name} ..."` | Uses an f-string to insert variable values into a readable message. |
Type conversion changes one type into another: `int("42")`, `str(3.14)`, and `float("2.0")`. An f-string lets you insert variables into a string: `f"The sensor reads {value:.2f} meters"`.
!!! tip "Tip"
    Python is case-sensitive. `Distance` and `distance` are different variables.
!!! warning "Common Mistake"
    Do not mix types unexpectedly. `"5" + 5` will crash. Use `int("5") + 5` instead.
!!! tip "Mini Exercise"
    Declare 3 variables for a robot: speed as a float, battery level as an int from 0-100, and current task as a string. Print a status report using an f-string.

---

## Data Structures


---

**Data Structure** — A container pattern for organizing related values, such as lists of readings or dictionaries of robot state.


### Data Structures: Lists, Tuples, Sets, Dictionaries
**Concept.** Data structures are containers. They let you keep related values together instead of creating a separate variable for everything.
[Open in Colab →](https://colab.research.google.com/github/purwar-lab/ml-for-robotics-/blob/main/notebooks/ch1-data-structures.ipynb)
!!! info "Colab notebook is pre-filled"
    The Colab notebook contains every code cell from this lesson in order, exactly as shown on this page. It opens pre-filled so you can run cells top to bottom, modify them, and save your own copy to Google Drive with **File → Save a copy in Drive**

#### Lists
A list is ordered and mutable. Use it for values that arrive over time, like ultrasonic sensor readings.
Key operations: indexing with `my_list[0]`, negative indexing with `my_list[-1]`, slicing with `my_list[1:3]`, and methods like `.append()`, `.remove()`, `.pop()`, and `.sort()`. Use `len(my_list)` for length.
Lists: sensor readings over time
```python
sensor_readings = [1.2, 0.9, 1.5, 2.1, 0.3]
sensor_readings.append(1.8)          # add new reading

print(f"Latest reading: {sensor_readings[-1]}m")
print(f"Closest obstacle: {min(sensor_readings):.1f}m")
print(f"Total readings: {len(sensor_readings)}")
print(f"Middle slice: {sensor_readings[1:3]}")
```
| Line | What it does |
| --- | --- |
| `sensor_readings = [1.2, 0.9, 1.5, 2.1, 0.3]` | Creates a list. Square brackets `[ ]` tell Python this is a list, and commas separate the values. |
| `sensor_readings.append(1.8)` | `.append()` is a list method. It adds one item to the end of the existing list. |
| `sensor_readings[-1]` | Uses a negative index. `-1` means the last item, `-2` means the second-to-last item, and so on. |
| `min(sensor_readings)` | Calls Python's built-in `min()` function and returns the smallest value in the list. |
| `len(sensor_readings)` | Returns the number of items in the list. |
| `sensor_readings[1:3]` | Creates a slice. It starts at index `1` and stops before index `3`. |
!!! tip "Mini Exercise"
    Create a list of 5 robot speeds. Find the max, min, and average with `sum()` and `len()`.

#### Tuples
A tuple is ordered and immutable. Use it when data should not change, like a fixed home coordinate.
Tuples: fixed robot configuration
```python
home_position = (0.0, 0.0)
wheel_diameter = (0.12,)       # single-element tuple needs a comma

x, y = home_position           # tuple unpacking
print(f"Home is at x={x}, y={y}")
```
| Line | What it does |
| --- | --- |
| `home_position = (0.0, 0.0)` | Creates a tuple. Parentheses `( )` group values that should stay together. |
| `wheel_diameter = (0.12,)` | Creates a one-item tuple. The comma is required; without it, Python treats the parentheses like normal grouping. |
| `x, y = home_position` | Uses tuple unpacking. Python assigns the first value to `x` and the second value to `y`. |
| `print(f"Home is at x={x}, y={y}")` | Uses an f-string to insert the unpacked values into a readable message. |
!!! tip "Tip"
    If you try to change a tuple element, Python throws a `TypeError`. That is the point: it protects fixed data.

#### Sets
A set is unordered and keeps unique elements only. It is useful when a camera reports objects and you only care which object types appeared.
Key operations: `.add()`, `.remove()`, membership with `in`, union with `|`, intersection with `&`, and difference with `-`.
Sets: unique objects seen by a robot
```python
objects_seen = {"chair", "table", "door", "chair", "wall"}
print(objects_seen)               # duplicates removed automatically
objects_seen.add("human")
print(f"Unique objects: {len(objects_seen)}")

room_a_objects = {"chair", "table", "lamp"}
room_b_objects = {"table", "plant", "lamp"}
shared = room_a_objects & room_b_objects
print(f"Objects in both rooms: {shared}")
```
| Line | What it does |
|:---|---|
| `objects_seen = {"chair", "table", "door", "chair", "wall"}` | Creates a set. Duplicate values are automatically kept only once. |
| `objects_seen.add("human")` | Adds one new value to the set. |
| `len(objects_seen)` | Counts how many unique objects are in the set. |
| `room_a_objects & room_b_objects` | Computes the intersection: only the objects that appear in both sets. |

#### Dictionaries
A dictionary stores key-value pairs. Use it for named robot state, sensor packets, and configuration settings.
Key operations: access with `dict[key]`, safer access with `.get(key, default)`, and inspection with `.keys()`, `.values()`, and `.items()`.
Dictionaries: robot state packet
```python
robot_state = {
    "name": "ARIA",
    "battery": 87,
    "speed": 1.2,
    "task": "mapping",
    "obstacles_detected": 3
}

print(f"Battery: {robot_state['battery']}%")
robot_state["battery"] -= 5
robot_state["location"] = (3.1, 4.2)

camera_status = robot_state.get("camera", "not installed")
print(f"Camera: {camera_status}")

for key, value in robot_state.items():
    print(f"  {key}: {value}")
```
When you call `robot_state.items()`, Python gives you back each key-value pair as a small tuple: `("name", "ARIA")`, `("battery", 87)`, and so on. The line `for key, value in robot_state.items():` uses tuple unpacking. Each pair is split into `key` and `value` automatically on every loop iteration.

| Line | What it does |
|:-|-|
| `robot_state = { ... }` | Creates a dictionary. Each entry has a key on the left and a value on the right. |
| `robot_state['battery']` | Looks up the value stored under the key `"battery"`. This crashes if the key does not exist. |
| `robot_state["battery"] -= 5` | Subtracts 5 from the existing battery value and stores the result back in the dictionary. |
| `robot_state["location"] = (3.1, 4.2)` | Adds a new key-value pair. The location value is a tuple. |
| `robot_state.get("camera", "not installed")` | Safely asks for a key. If `"camera"` is missing, Python returns the default value instead of crashing. |
| `for key, value in robot_state.items():` | Loops through key-value pairs. `.items()` returns pairs, and tuple unpacking splits each pair into two variables. |
!!! warning "Common Mistake"
    `robot_state["camera"]` crashes if the key does not exist. Use `.get("camera", "not installed")` when a key is optional.
!!! tip "Mini Exercise"
    Create a dictionary representing a robot with at least 5 attributes. Decrease the battery by 10 and print all keys that have numeric values.

---

## Control Flow


---

**Control Flow** — The part of a program that decides which code runs based on conditions.


### Control Flow: if / elif / else
**Concept.** Control flow lets your program make decisions. Comparisons include `==`, `!=`, `<`, `>`, `<=`, and `>=`. Logical operators include `and`, `or`, and `not`.
Battery and obstacle decisions
```python
battery = 23
distance_to_obstacle = 0.4

if battery < 10:
    print("CRITICAL: Return to base immediately!")
elif battery < 25:
    print("WARNING: Low battery. Finish current task and return.")
    if distance_to_obstacle < 0.5:
        print("ALSO: Obstacle too close - slow down!")
else:
    print("Battery OK. Continuing mission.")

safe_zones = {"base", "charging_dock", "lab_entrance"}
current_zone = "charging_dock"
if current_zone in safe_zones:
    print(f"{current_zone} is a safe zone. Power down.")
```
| Line | What it does |
| --- | --- |
| `battery = 23` | Stores the current battery level as an integer. |
| `distance_to_obstacle = 0.4` | Stores a sensor reading as a float in meters. |
| `if battery < 10:` | Starts the first decision branch. The indented line below it runs only if this condition is true. |
| `elif battery < 25:` | Checks a second condition only if the first condition was false. |
| `if distance_to_obstacle < 0.5:` | Starts a nested `if`. It is indented one extra level because it belongs inside the low-battery branch. |
| `else:` | Runs only when the earlier battery conditions were false. |
| `safe_zones = {"base", "charging_dock", "lab_entrance"}` | Creates a set of allowed safe-zone names. |
| `if current_zone in safe_zones:` | Uses `in` to test whether the current zone appears in the set. |
!!! warning "Nested indentation"
    The obstacle check is indented one more level because it only runs inside the `elif battery < 25` branch. If it lined up with `elif`, it would be a separate decision instead.
!!! tip "Tip"
    Use `elif` instead of multiple `if` blocks when conditions are mutually exclusive. It is clearer and avoids accidental double responses.
!!! tip "Mini Exercise"
    Write a function that takes a distance reading in meters and prints `STOP` if it is under 0.3m, `SLOW DOWN` if it is under 0.8m, and `ALL CLEAR` otherwise.

---

## Loops


---

**Loop** — A programming structure that repeats work over a collection or until a condition changes.


### Loops: for and while
**Concept.** Loops repeat work. Use `for` when you know what collection you are walking through. Use `while` when you repeat until a condition changes.
You will also see `continue` to skip the rest of the current loop iteration and `zip()` to walk through two lists together, such as timestamps and readings.
[Open in Colab →](https://colab.research.google.com/github/purwar-lab/ml-for-robotics-/blob/main/notebooks/ch1-loops.ipynb)
!!! info "Colab notebook is pre-filled"
    The Colab notebook contains every code cell from this lesson in order, exactly as shown on this page. It opens pre-filled so you can run cells top to bottom, modify them, and save your own copy to Google Drive with **File → Save a copy in Drive**

#### For loops
For loops with enumerate and zip
```python
sensor_log = [1.5, 0.8, 2.3, 0.2, 1.9]
timestamps = ["00:00", "00:05", "00:10", "00:15", "00:20"]

print("Scanning sensor log...")
for i, reading in enumerate(sensor_log):
    if reading < 0.5:
        print(f"  Reading #{i}: {reading:.1f}m - DANGER")
    else:
        print(f"  Reading #{i}: {reading:.1f}m - OK")

print("\nReadings with timestamps:")
for time, reading in zip(timestamps, sensor_log):
    print(f"  {time}: {reading:.1f}m")
```
| Line | What it does |
| --- | --- |
| `sensor_log = [...]` | Stores the readings in a list so the loop can visit them one at a time. |
| `timestamps = [...]` | Stores matching time labels in a second list. |
| `for i, reading in enumerate(sensor_log):` | `enumerate()` returns two values each time: the index and the reading. Python unpacks them into `i` and `reading`. |
| `if reading < 0.5:` | Checks the current reading and chooses the danger message when the obstacle is close. |
| `for time, reading in zip(timestamps, sensor_log):` | `zip()` walks through two lists together and returns pairs like `("00:00", 1.5)`. |

#### While loops
While loop: battery until destination
```python
battery = 100
distance_traveled = 0.0

while battery > 20:
    battery -= 5
    distance_traveled += 0.3
    if distance_traveled >= 1.2:
        print(f"Destination reached! Battery remaining: {battery}%")
        break
else:
    print(f"Battery too low! Only traveled {distance_traveled:.1f}m")
```
| Line | What it does |
| --- | --- |
| `battery = 100` | Starts the simulated battery at 100 percent. |
| `while battery > 20:` | Repeats the indented block as long as the condition remains true. |
| `battery -= 5` | Subtracts 5 from the battery each loop. This moves the loop toward stopping. |
| `distance_traveled += 0.3` | Adds another 0.3 meters to the simulated trip. |
| `break` | Exits the loop immediately when the destination is reached. |
| `else:` | Runs only if the `while` loop ended naturally, not if it ended with `break`. |

#### List Comprehensions: a One-Line Loop
A list comprehension is a shorthand way to build a new list from an existing one, all in one line. It looks unusual at first, but it follows a fixed pattern you can memorize.
```
new_list = [ expression   for item in existing_list   if condition ]
```

The expanded equivalent:

```
new_list = []
for item in existing_list:
    if condition:
        new_list.append(expression)
```

- **Expression.** The value to put into the new list. In the long version, this is the value passed into `.append()`.
- **Existing list.** The collection Python loops through. In the long version, it appears after `for item in`.
- **Condition.** The optional test that decides whether an item is included. If the condition is false, that item is skipped.
List comprehensions: a one-line loop
```python
# Long version (normal for loop)
raw_readings = [1.5, -0.2, 0.9, 2.1, -0.1]

clean_readings_long = []               # start with empty list
for r in raw_readings:                 # go through every item
    if r > 0:                          # only keep positive values
        clean_readings_long.append(r)  # add to new list

print("Long version:", clean_readings_long)

# Short version (list comprehension - identical result)
clean_readings_short = [r for r in raw_readings if r > 0]
#                        keep r   for every r       only if r > 0

print("Short version:", clean_readings_short)

# Another example: convert all readings from meters to centimeters
in_cm = [r * 100 for r in clean_readings_short]
print("In cm:", in_cm)
```
| Line | What it does |
| --- | --- |
| `clean_readings_long = []` | Creates an empty list that the long loop will fill. |
| `for r in raw_readings:` | Visits each raw reading one at a time. |
| `if r > 0:` | Filters out invalid negative readings. |
| `clean_readings_long.append(r)` | Adds the valid reading to the new list. |
| `[r for r in raw_readings if r > 0]` | Does the same loop, filter, and append operation in one line. |
| `[r * 100 for r in clean_readings_short]` | Builds a new list by converting each meter reading to centimeters. There is no filter this time. |
!!! tip "Reading list comprehensions"
    When you see `[something for x in something_else]`, that is a list comprehension. Read it as: build a list by taking `x` from `something_else`, and optionally filtering. Once you spot the pattern, it becomes easy to read.
!!! warning "Common Mistake"
    A `while` loop can run forever if the condition never changes. Make sure something inside the loop moves the program toward stopping.
!!! tip "Mini Exercise"
    Use a for loop to count how many motor temperatures are above 80°C and calculate the average temperature. Then rewrite the filtering step as a list comprehension.

---

## Functions


---

**Function** — A reusable block of code with a name, inputs, and often a returned result.


### Functions
**Concept.** Functions package reusable behavior. They make code easier to test, read, and repair. A function can take parameters, use default values, and return a result.
[Open in Colab →](https://colab.research.google.com/github/purwar-lab/ml-for-robotics-/blob/main/notebooks/ch1-functions.ipynb)
!!! info "Colab notebook is pre-filled"
    The Colab notebook contains every code cell from this lesson in order, exactly as shown on this page. It opens pre-filled so you can run cells top to bottom, modify them, and save your own copy to Google Drive with **File → Save a copy in Drive**

#### Anatomy of a Function

```python
def analyze_sensor_data(readings, danger_threshold=0.5):
    """
    Analyze a list of sensor readings.
    Returns a summary dictionary.
    """
    clean = [r for r in readings if r > 0]
    return {"min": min(clean)}
```

- **`def` keyword.** Tells Python that the next block defines a function.
- **Function name.** This is the name you call later, such as `analyze_sensor_data(data)`.
- **Required parameter.** `readings` has no default, so the caller must provide it.
- **Optional parameter.** `danger_threshold` can be provided by the caller, but it also has a fallback value.
- **Default value.** If the caller does not pass `danger_threshold`, Python uses `0.5`.
- **Docstring.** Triple-quoted text that documents what the function does.
- **Function body.** The indented code that runs when the function is called.
- **Return line.** Sends a value back to whoever called the function.
Function: analyze robot sensor readings
```python
def analyze_sensor_data(readings, danger_threshold=0.5, unit="m"):
    """
    Analyze a list of sensor readings and return a summary.
    """
    if not readings:
        return {"error": "No readings provided"}

    clean = [r for r in readings if r > 0]

    return {
        "min": min(clean),
        "max": max(clean),
        "average": sum(clean) / len(clean),
        "danger_count": sum(1 for r in clean if r < danger_threshold),
        "unit": unit
    }

data = [1.2, 0.3, 0.8, 0.4, 2.1, 0.1, -0.5, 1.6]
result = analyze_sensor_data(data, danger_threshold=0.5)

print(f"Min distance:  {result['min']:.2f}{result['unit']}")
print(f"Max distance:  {result['max']:.2f}{result['unit']}")
print(f"Average:       {result['average']:.2f}{result['unit']}")
print(f"Danger events: {result['danger_count']}")
```
| Line | What it does |
| --- | --- |
| `def analyze_sensor_data(...):` | Defines a reusable function with three parameters. `danger_threshold` and `unit` have default values. |
| `"""Analyze ..."""` | Documents the function with a docstring. Python ignores it during normal execution, but tools can display it as help text. |
| `if not readings:` | Checks for an empty list. Empty lists count as false in Python. |
| `clean = [r for r in readings if r > 0]` | Uses a list comprehension to keep only positive readings. |
| `return { ... }` | Returns a dictionary containing several named summary values. |
| `sum(1 for r in clean if r < danger_threshold)` | Counts readings below the danger threshold by adding one for each matching reading. |
| `result = analyze_sensor_data(...)` | Calls the function and stores its returned dictionary in `result`. |

#### Commenting
Python has two ways to write comments. Comments are ignored by Python. They exist only for humans reading the code.
Single-line comment: anything after a `#` on a line is a comment. Multi-line comment: wrap text in triple quotes `""" ... """`. When used at the start of a function, this is called a docstring.
Commenting and docstrings
```python
# This is a single-line comment. Python ignores everything after the #.

distance = 2.45   # you can also put a comment at the end of a line

"""
This is a multi-line comment (also called a docstring when it appears
at the start of a function or file).
Python does not execute any of this text.
"""

def get_battery_warning(level):
    """
    Return a warning string based on battery level.
    level: int from 0 to 100
    Returns: str
    """
    if level < 10:
        return "CRITICAL"
    elif level < 25:
        return "LOW"
    return "OK"

print(get_battery_warning(8))
print(get_battery_warning(50))
```
| Line | What it does |
| --- | --- |
| `# This is a single-line comment` | Starts a comment. Python skips the rest of the line. |
| `distance = 2.45   # ...` | Stores a number, then adds a human note after the code. |
| `""" ... """` | Creates a multi-line string. At the top of a function, this becomes a docstring. |
| `def get_battery_warning(level):` | Defines a function with one required parameter. |
| `return "CRITICAL"` | Stops the function and sends the string back to the caller. |

#### Returning Multiple Values
Most languages need special syntax to return more than one value from a function. In Python, you just separate values with a comma. Python automatically packs them into a tuple and unpacks them on the other side.
Returning multiple values
```python
def analyze_motor(temperature, rpm):
    """
    Check motor health. Returns two values: a status string and a boolean.
    """
    is_overheating = temperature > 80

    if temperature > 90:
        status = "SHUTDOWN"
    elif temperature > 80:
        status = "WARNING"
    else:
        status = "OK"

    return status, is_overheating   # returning TWO values, separated by a comma

# Method 1: capture both values into two variables
motor_status, overheating = analyze_motor(85, 3000)
print(f"Status: {motor_status}")          # WARNING
print(f"Overheating: {overheating}")      # True

# Method 2: capture as a single tuple (Python packs them automatically)
result = analyze_motor(65, 2800)
print(f"Full result tuple: {result}")     # ('OK', False)
print(f"Just the status: {result[0]}")    # OK
```
| Line | What it does |
| --- | --- |
| `return status, is_overheating` | Returns two values. Python wraps them in a tuple automatically, such as `("WARNING", True)`. |
| `motor_status, overheating = analyze_motor(...)` | Uses tuple unpacking. Python splits the tuple and assigns the first value to `motor_status` and the second to `overheating`. |
| `result = analyze_motor(65, 2800)` | Stores the whole returned tuple in one variable. |
| `result[0]` | Reads the first item from the tuple. |

#### Positional vs Named Arguments
Positional vs named arguments
```python
def set_robot_speed(speed, unit="m/s", silent=False):
    """Set robot speed. unit defaults to m/s, silent defaults to False."""
    if not silent:
        print(f"Setting speed to {speed} {unit}")
    return speed

# POSITIONAL: arguments matched by their position in the definition
set_robot_speed(1.5)
set_robot_speed(2.0, "km/h")

# NAMED: arguments matched by name, order does not matter
set_robot_speed(speed=1.5, silent=True)
set_robot_speed(silent=True, speed=1.5)
set_robot_speed(1.5, silent=True, unit="mph")

# DEFAULT VALUES: if you do not pass an argument, Python uses the default.
```
| Line | What it does |
| --- | --- |
| `speed` | Required parameter. The caller must provide this. Omitting it causes a `TypeError`. |
| `unit="m/s"` | Optional parameter with a default value. If the caller does not pass it, Python uses `"m/s"`. |
| `silent=False` | Another optional parameter. It defaults to `False`. |
| `set_robot_speed(2.0, "km/h")` | Passes positional arguments. Python matches them left to right. |
| `set_robot_speed(silent=True, speed=1.5)` | Passes named arguments. Order does not matter because the names identify the parameters. |
| `set_robot_speed(1.5, silent=True, unit="mph")` | Mixes styles. Positional arguments must come first, then named arguments. |
!!! tip "Parameter order rule"
    Required parameters with no default must always come before optional parameters with defaults in the function definition.

#### `*args` and `**kwargs`
You do not need to write functions with `*args` or `**kwargs` yourself yet. But you will see them when you look at library documentation. Here is just enough to recognize them.
*args and **kwargs
```python
# *args collects any number of positional arguments into a tuple
def print_all_readings(*args):
    """Accept any number of readings."""
    for reading in args:
        print(f"  {reading}m")

print_all_readings(1.2, 0.9, 1.5, 2.1)

# **kwargs collects any number of named arguments into a dictionary
def configure_robot(**kwargs):
    """Accept any named settings."""
    for key, value in kwargs.items():
        print(f"  {key} = {value}")

configure_robot(speed=1.5, task="mapping", battery=87)
```
| Line | What it does |
| --- | --- |
| `def print_all_readings(*args):` | Collects any number of extra positional arguments into a tuple named `args`. |
| `for reading in args:` | Loops through every value that was passed in. |
| `def configure_robot(**kwargs):` | Collects any number of named arguments into a dictionary named `kwargs`. |
| `kwargs.items()` | Returns key-value pairs so the loop can unpack each setting name and value. |
!!! tip "Names are convention"
    The names `*args` and `**kwargs` are convention, not keywords. The `*` and `**` are what matter. You could write `*values` or `**settings`, but use `*args` and `**kwargs` so other developers recognize the pattern.
!!! tip "Scope"
    Variables created inside a function are local. Code outside the function cannot see them unless the function returns them.
!!! warning "Common Mistake"
    Printing a value is not the same as returning it. If another part of your program needs the result, use `return`.
!!! tip "Mini Exercise"
    Write `convert_speed(value, from_unit, to_unit)` to convert between `m/s`, `km/h`, and `mph`. Return the converted value and print a formatted message.

---

## Classes


---

**Class** — A blueprint that bundles related data (attributes) and the functions that act on it (methods) into one object.


### Classes
**Concept.** A function packages behavior. A **class** packages behavior *and* the data that behavior works on, so they travel together as one object. Every robot project later in this course is built from classes --- `Tracker`, `Commander`, `Telemetry`, and `MobileVideoStream` are all classes. This lesson teaches the basics so those files read like something you already understand.

#### Why Not Just Functions?
Imagine controlling a robot with plain functions. Where does the robot's current battery level live? You would have to pass it into every function and hand the updated value back out again, every single time. A class fixes this: the object *remembers* its own data between calls, and its methods read and update that data directly.

#### Anatomy of a Class
Defining and using a class
```python
class Robot:
    """A robot that remembers its own name and battery level."""

    def __init__(self, name, battery=100):
        self.name = name            # attribute stored on this object
        self.battery = battery      # attribute with a default value

    def drive(self, minutes):
        """Driving drains the battery."""
        self.battery -= minutes * 2
        print(f"{self.name} drove {minutes} min, battery now {self.battery}%")

    def charge(self):
        """Refill the battery to full."""
        self.battery = 100
        print(f"{self.name} is fully charged")


# Create two independent objects from the same blueprint
scout = Robot("Scout")             # battery uses the default, 100
rover = Robot("Rover", battery=40)

scout.drive(10)                    # Scout drove 10 min, battery now 80%
rover.drive(10)                    # Rover drove 10 min, battery now 20%

print(scout.battery)               # 80   each object keeps its own data
print(rover.battery)               # 20

rover.charge()                     # Rover is fully charged
print(rover.battery)               # 100
```
| Line | What it does |
| --- | --- |
| `class Robot:` | Defines a new class (a blueprint). By convention class names use CapitalizedWords. |
| `def __init__(self, name, battery=100):` | The constructor. Python runs it automatically every time you create an object. `battery` has a default value. |
| `self.name = name` | Stores data *on the object* as an attribute. `self` is the object being built. |
| `def drive(self, minutes):` | A method --- a function that belongs to the class. Its first parameter is always `self`. |
| `self.battery -= minutes * 2` | Reads and updates the object's own attribute. No need to pass the battery in and out. |
| `scout = Robot("Scout")` | Creates an object (an *instance*) from the blueprint. `__init__` runs here. |
| `scout.drive(10)` | Calls a method on that object. Python passes `scout` in as `self` automatically. |
| `scout.battery` | Reads an attribute back off the object. |

#### Each Object Keeps Its Own Data
`scout` and `rover` are built from the same class, but they are independent objects. Driving `scout` does not touch `rover`'s battery --- each instance carries its own attributes. That is the whole point: one blueprint, many objects, each with its own state.
!!! tip "self is automatic"
    You write `def drive(self, minutes)` with `self`, but you call `scout.drive(10)` without it. Python fills in `self` for you --- it is just the object on the left of the dot.

#### The Pattern You Will See in Every Project
Classes shine when an object must *remember* something between calls. A controller, a sensor stream, a network connection --- all of them hold state. Here is a tiny preview of the structure Project 1's PID controller uses.
State that persists between calls
```python
# This is the same pattern Project 1 uses for its PID controller:
# settings and state live on the object and persist between calls.
class SimpleController:
    def __init__(self, gain):
        self.gain = gain            # set once, remembered for every call
        self.last_error = 0         # state that persists between steps

    def step(self, error):
        change = error - self.last_error
        self.last_error = error      # remember it for next time
        return self.gain * error + change


controller = SimpleController(gain=0.5)
print(controller.step(10))         # last_error was 0
print(controller.step(8))          # last_error is now 10
print(controller.step(8))          # last_error is now 8
```
Each call to `step()` remembers the previous error, because that value lives on the object. Plain functions cannot do this without extra bookkeeping.
!!! tip "Where this goes"
    In Project 1 you open `shared.py` and see `class Commander`, `class Telemetry`, and `class MobileVideoStream`. Each one bundles a socket or a thread together with the methods that use it --- exactly the pattern above, just with real hardware behind it.
!!! tip "Mini Exercise"
    Write a `Thermostat` class with a `target` attribute and a method `reading(temp)` that returns `"HEAT"`, `"COOL"`, or `"OK"` by comparing `temp` to `self.target`. Create two thermostats with different targets and confirm they behave independently.

---

## Libraries & Imports


---

**Library** — A package of reusable code written by other developers so you do not rebuild common tools from scratch.


### Libraries & Imports
**Concept.** A library is code someone else packaged so you do not have to rebuild everything from scratch. In Colab, common ML libraries are already installed. For anything missing, you can run `!pip install package_name`.

#### Installing and Importing Libraries
Python comes with a set of built-in tools called the standard library: things like `math`, `random`, `datetime`, and `os`. But the tools we need for machine learning, such as NumPy, Pandas, Matplotlib, and scikit-learn, are third-party packages. Other developers wrote them and published them for free.
All Python packages live on a public registry called [PyPI](https://pypi.org). When you run `!pip install numpy`, `pip`, Python's package manager, downloads NumPy from PyPI and installs it.
The `!` at the start of a line in a Colab cell means: run this as a shell command, not as Python. You run it in a Colab code cell, not in a Python file.
Colab package install command
```python
!pip install numpy pandas matplotlib scikit-learn
```
After running that cell, those packages are available for the rest of your Colab session. You do not need to install them again until you start a new session.
!!! tip "Most common packages are already installed"
    Most packages come pre-installed in Google Colab. You only need `!pip install` for packages that are less common. If you try to import something and see `ModuleNotFoundError`, that is your signal to run `!pip install package-name` first.
The install name and the import name are sometimes different. Here are the package names used in this course.
| What we call it | pip install command | import statement |
| --- | --- | --- |
| NumPy | `!pip install numpy` | `import numpy as np` |
| Pandas | `!pip install pandas` | `import pandas as pd` |
| Matplotlib | `!pip install matplotlib` | `import matplotlib.pyplot as plt` |
| Scikit-learn | `!pip install scikit-learn` | `from sklearn import ...` |
| OpenCV | `!pip install opencv-python` | `import cv2` |
| TensorFlow | `!pip install tensorflow` | `import tensorflow as tf` |
| OpenAI Gym | `!pip install gymnasium` | `import gymnasium as gym` |
**Note:** The install name `scikit-learn` is different from the import name `sklearn`. This is the most common source of confusion. Always look up the correct import name in the library's official documentation.
NumPy Arrays and math Pandas Data tables Matplotlib Plots and charts Scikit-learn Classic ML TensorFlow/Keras Neural networks
Standard ML import block
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sensor_array = np.array([1.2, 0.9, 1.5, 2.1, 0.3, 1.8])

print(f"Mean: {np.mean(sensor_array):.3f}")
print(f"Std Dev: {np.std(sensor_array):.3f}")
print(f"Readings above 1.0m: {np.sum(sensor_array > 1.0)}")

sensor_array_cm = sensor_array * 100
print(f"In cm: {sensor_array_cm}")
```
!!! tip "Robotics Connection"
    Robotics data is often arrays: images, lidar scans, sensor logs, joint angles, and motor current. NumPy is the shared foundation underneath most of those tools.
!!! warning "Common Mistake"
    `import numpy` works, but the standard nickname is `import numpy as np`. Most tutorials assume that alias.
!!! tip "Mini Exercise"
    Create a NumPy array of 10 motor current readings. Print the average, maximum, and how many readings are above a safety threshold you choose.

---

## Checkpoint Exercise: Sensor Data Parser


---

###  Checkpoint Exercise: Sensor Data Parser
**The task.** A robot has been collecting data for 30 minutes. You are given a raw log as a list of dictionaries. Parse it, clean it, and produce a summary report.
[Open in Colab →](https://colab.research.google.com/github/purwar-lab/ml-for-robotics-/blob/main/notebooks/ch1-checkpoint.ipynb)
!!! info "Colab notebook is pre-filled"
    The Colab notebook contains every code cell from this lesson in order, exactly as shown on this page. It opens pre-filled so you can run cells top to bottom, modify them, and save your own copy to Google Drive with **File → Save a copy in Drive**
**Run this cell first.** It creates the simulated robot log you will analyze. You do not need to understand every line right now. Just run it and move on.
Cell 1: create simulated robot log
```python
robot_log = [
    {"time": "00:00", "distance": 1.2, "motor_temp": 54.2, "battery": 100, "task": "mapping"},
    {"time": "05:00", "distance": -1,  "motor_temp": 58.8, "battery": 94,  "task": "mapping"},
    {"time": "10:00", "distance": 0.4, "motor_temp": 66.1, "battery": 87,  "task": "avoidance"},
    {"time": "15:00", "distance": 2.2, "motor_temp": 63.4, "battery": 80,  "task": "delivery"},
    {"time": "20:00", "distance": -1,  "motor_temp": 71.0, "battery": 73,  "task": "delivery"},
    {"time": "25:00", "distance": 1.1, "motor_temp": 78.5, "battery": 66,  "task": "return"},
    {"time": "30:00", "distance": 0.9, "motor_temp": 74.0, "battery": 60,  "task": "return"},
]
```
The variable `robot_log` now contains 30 minutes of sensor data. Your job is to analyze it. Try the exercise yourself before opening the solution.
!!! tip "Exercise"
    Write analysis code that does the following: count total entries and sensor errors, filter out `distance == -1` entries, find the highest motor temperature and its task, calculate average battery drain per minute, use a set to find unique tasks, and print a formatted report using f-strings.
!!! tip "Self-check"
    [ ] Did you use a function? [ ] Did you use a list comprehension? [ ] Did you use a dictionary for the summary? [ ] Is your output clearly formatted?
Show Solution


Checkpoint solution
```python
robot_log = [
    {"time": "00:00", "distance": 1.2, "motor_temp": 54.2, "battery": 100, "task": "mapping"},
    {"time": "05:00", "distance": -1,  "motor_temp": 58.8, "battery": 94,  "task": "mapping"},
    {"time": "10:00", "distance": 0.4, "motor_temp": 66.1, "battery": 87,  "task": "avoidance"},
    {"time": "15:00", "distance": 2.2, "motor_temp": 63.4, "battery": 80,  "task": "delivery"},
    {"time": "20:00", "distance": -1,  "motor_temp": 71.0, "battery": 73,  "task": "delivery"},
    {"time": "25:00", "distance": 1.1, "motor_temp": 78.5, "battery": 66,  "task": "return"},
    {"time": "30:00", "distance": 0.9, "motor_temp": 74.0, "battery": 60,  "task": "return"},
]

def summarize_robot_log(log):
    total_entries = len(log)
    sensor_error_count = sum(1 for row in log if row["distance"] == -1)
    clean_log = [row for row in log if row["distance"] != -1]

    hottest = max(log, key=lambda row: row["motor_temp"])
    minutes = int(log[-1]["time"].split(":")[0]) - int(log[0]["time"].split(":")[0])
    battery_drain = log[0]["battery"] - log[-1]["battery"]
    average_drain = battery_drain / minutes
    unique_tasks = {row["task"] for row in log}

    return {
        "total_entries": total_entries,
        "sensor_errors": sensor_error_count,
        "valid_entries": len(clean_log),
        "hottest_temp": hottest["motor_temp"],
        "hottest_task": hottest["task"],
        "average_drain": average_drain,
        "unique_tasks": unique_tasks
    }

summary = summarize_robot_log(robot_log)

print("ROBOT LOG SUMMARY")
print("-----------------")
print(f"Total entries:       {summary['total_entries']}")
print(f"Sensor errors:       {summary['sensor_errors']}")
print(f"Valid entries:       {summary['valid_entries']}")
print(f"Highest temperature: {summary['hottest_temp']:.1f} C during {summary['hottest_task']}")
print(f"Battery drain/min:   {summary['average_drain']:.2f}%")
print(f"Tasks performed:     {', '.join(sorted(summary['unique_tasks']))}")
```
