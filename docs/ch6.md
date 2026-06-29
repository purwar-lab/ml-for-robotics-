# Chapter 6: Neural Networks

---

## What Is a Neural Network?

<video controls height="100%"><source src="../original/tf_playground.mp4" type="video/mp4"></video>

---

In Chapter 3 you trained a Random Forest to predict machine failure. The Random Forest learned a set of if-then rules from the data. Clear, interpretable, fast to train. For many problems it is the right tool and you should use it.
A neural network learns differently. Instead of rules, it learns numbers --- thousands or millions of adjustable numbers called weights. Training is the process of tuning those numbers until the network's output matches the correct answer. The rules are never written down explicitly. The pattern lives entirely in the weights.
Think of learning to ride a bike. A rulebook says: lean left to turn left, pedal faster to accelerate, squeeze brakes to stop. A neural network is more like your body after hours of practice --- it just knows, without being able to explain every adjustment it makes. That inability to explain is both the power and the limitation.
A chain of matrix multiplications and simple math functions. That is all. The word "neural" is a metaphor. There is no brain, no understanding, no consciousness --- just math that happens to learn patterns from examples.
Use neural networks when the relationship between inputs and outputs is too complex for simple rules --- images, audio, text, complex sensor combinations. For clean tabular data like the Chapter 3 failure dataset, Random Forest is simpler and often just as accurate. You will return to this question in lesson 6.12.
!!! tip "What comes next"
    You will build intuition for neural networks in the next few lessons using a free interactive tool called TensorFlow Playground --- no code required yet.

---

## Meet TensorFlow Playground


---

TensorFlow Playground is a neural network that runs live in your browser. You set the architecture, hit play, and watch the network train in real time. The decision boundary --- the boundary between classes --- updates every few milliseconds as the network learns. This makes concepts that are invisible in code completely visible.
[Open in Colab →](https://playground.tensorflow.org)

### What you are looking at
TensorFlow Playground shows the dataset, input features, network layers, output boundary, and loss graph in one live interface.
- **Data panel (left):** choose your dataset and how noisy it is.
- **Features (above network):** which input columns the network sees.
- **Hidden layers (center):** the heart of the network --- add layers and neurons by clicking + and -.
- **Output panel (right):** the decision boundary --- blue = predicted class 1, orange = predicted class -1. Darker color = higher confidence.
- **Loss graph (top right):** how wrong the network is over time.
- **Training controls (top):** learning rate, activation, batch size, and the play/pause/reset buttons.

### Your first 60 seconds in Playground
Do these steps before reading any further:
Open Playground at the link above. Select the Circle dataset. Look for four small icons below "DATA" on the left — pick the one showing two concentric circles. Click the play button ▶ at the top left. Watch for 10 seconds. Click the reset button ↺ and watch again from the start.
!!! tip "Write your answer before moving on"
    What shape is the decision boundary trying to form? Is it a straight line or a curve? Does it succeed? Write down what you see --- you will compare this to what a single neuron can do in the next lesson.

---

## Anatomy of a Neuron


---

Before understanding a network of hundreds of neurons, understand one. A single neuron is just arithmetic --- three steps, no mystery.
**The exam scorer**

Imagine scoring a student's performance using three things: homework (weight 0.5), exam score (weight 0.4), attendance (weight 0.1). You multiply each by its weight, add them up, and get a score. A neuron does exactly this — it is a weighted sum of its inputs. The only extra step is squashing the result through a function so it stays in a useful range.

### Three steps every neuron takes

**Weighted sum** `z = w1*x1 + w2*x2 + ... + wn*xn + b`

Each input `x` is multiplied by its weight `w`. All results are added together. The bias `b` is added last — it shifts the output regardless of inputs.

**Activation function** `a = f(z)`

The function `f` squashes `z` into a useful range. Without it, stacking neurons is useless — the whole network collapses into one weighted sum, which is just a straight line.

**Pass the result forward**

`a` becomes an input to the next layer's neurons. This is how information flows from left to right through the network.
!!! tip "Try this before running"
    The code below computes what one neuron outputs given three sensor readings. Before running, predict: if temperature (`x[0]`) is very high (`0.9` out of `1.0`) and its weight is `0.8`, will the output confidence be above 50% or below 50%?
single_neuron_sensor_confidence.py
```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Sensor readings normalized from 0 to 1: temp, vibration, speed
x = np.array([0.9, 0.7, 0.5])

# Weights and bias. During real training, the network learns these.
w = np.array([0.8, 0.3, -0.2])
b = -0.4

z = np.dot(w, x) + b
a = sigmoid(z)

print(f"Weighted sum z: {z:.4f}")
print(f"After sigmoid:  {a:.4f}")
print(f"Interpretation: {a*100:.1f}% confidence of abnormality")

# Try changing w[1]. What happens if vibration matters more than temperature?
```
!!! tip "Exercise"
    Run the code, then try each change and note what happens:

    1. Change `w[0]` from `0.8` to `-0.8`. What happens to confidence? Why does making a weight negative reverse its effect?
    2. Change `b` from `-0.4` to `0.4`. What happens? The bias shifts the baseline — what does a positive bias mean in terms of the neuron's "default assumption"?
    3. Set all weights to `0.0`. What is the output and why?

### Weights are not set by you
In the code above, the weights are hand-picked numbers. In a real network, training finds these numbers automatically by trying to minimize the loss. The network starts with random weights and adjusts them thousands of times until the output is correct. You will see this process in lesson 6.6.
!!! tip "TensorFlow Playground connection"
    Every connection between neurons in TensorFlow Playground is one weight --- hover over any connection and you can see its current value updating as the network trains.

---

##  Playground: One Neuron Fails


---

A single neuron draws one straight line. That is its only tool. Some problems can be solved with a straight line. Others cannot. This experiment shows you both --- and the moment you add a second layer, the impossible becomes trivial.
[Open in Colab →](https://playground.tensorflow.org)

### Part 1 --- Set up the impossible problem
Open Playground: https://playground.tensorflow.org

- **Dataset:** click the XOR dataset. It is the checkerboard pattern: orange top-right and bottom-left, blue top-left and bottom-right.
- **Features:** make sure only `X1` and `X2` are checked.
- **Hidden layers:** click the minus button until there are zero hidden layers — just inputs going straight to output.
- **Learning rate:** set to `0.03`.
- **Activation:** set to `ReLU`.
- **Train:** click ▶ and watch for 30 seconds.
!!! tip "Question"
    What accuracy does it reach? Does the boundary ever successfully separate the orange from the blue? Why do you think a straight line cannot solve this problem? Think geometrically: is there any straight line that puts all orange on one side and all blue on the other?

### Part 2 --- Add one hidden layer

- Click ↺ to reset.
- Add one hidden layer with 2 neurons: click `+` next to Hidden Layers, then set the layer to 2 neurons.
- Keep everything else the same.
- Click ▶ to train.
- Watch until the test loss drops below `0.05`.
!!! tip "Question"
    How long did it take? What does the decision boundary look like now? The two neurons together created a boundary that one neuron could not. Watch the two neurons on the left --- each one is drawing its own line. Their combination creates the curved boundary you see on the right.

### Why layers work
Each neuron in the hidden layer learns a different view of the data --- one might learn "is this point in the top half?" while another learns "is this point in the right half?" Neither answer alone is enough. But combined, they can describe "top-right AND bottom-left," which is exactly what XOR needs.
This is why deep networks can learn complex patterns. Each layer transforms the data into a new representation. By the final layer a pattern that was impossible to separate has been transformed into something simple. The network learned to see the data differently at each stage.
!!! tip "Challenge"
    After solving XOR with 2 neurons in one layer, try to solve the spiral dataset, the hardest one with two interleaved spirals. How many hidden layers and neurons does it take? Can you solve it with just 1 hidden layer?

---

## Layers: The Fix


---

Lesson 6.3 showed the problem empirically --- one neuron fails on XOR, a hidden layer fixes it. This lesson explains why mathematically without requiring calculus.
**The committee vote**

One judge scores a gymnastics routine: 7.2. That is a single number capturing a complex performance. Three judges each notice different things — difficulty, execution, artistry. Their combined scores are richer than any single judge's opinion. A hidden layer is a committee of neurons, each noticing a different aspect of the input. Their combined output gives the next layer richer information to work with.

### What a layer actually does to data
Each layer takes the outputs from the previous layer and transforms them. The first hidden layer transforms raw input features into a new representation. The second layer transforms that representation into an even more abstract one. By the final layer the data has been transformed enough that a single neuron can make the final decision.
In Playground you can see this happening. Click on any neuron in a hidden layer during training. The small square inside it shows what that neuron has learned to detect --- a gradient of orange and blue showing which parts of the input space activate it strongly. Each neuron is a feature detector. Together they decompose the problem into parts that the output neuron can combine.
!!! tip "Try this before running"
    Before running, predict the parameter count for a network with `2 inputs -> 3 neurons -> 3 neurons -> 1 output`. Then run `count_params([2, 3, 3, 1])` to check.
count_network_parameters.py
```python
def count_params(layer_sizes):
    total = 0
    for i in range(len(layer_sizes) - 1):
        weights = layer_sizes[i] * layer_sizes[i + 1]
        biases = layer_sizes[i + 1]
        params = weights + biases
        print(f"Layer {i+1}: {layer_sizes[i]} -> {layer_sizes[i+1]}"
              f" ({weights} weights + {biases} biases = {params})")
        total += params
    print(f"Total parameters: {total:,}")

count_params([3, 4, 4, 1])
print()
count_params([3, 64, 64, 1])
print()
count_params([3, 4, 4, 4, 1])
```
!!! tip "Counting rule"
    Every connection is one weight, and every neuron has one bias. That is why bigger layers quickly create many more trainable numbers.

### How deep should you go?

| Architecture | Parameters | Good for |
|---|---|---|
| `2->4->1` | 13 | Very simple patterns |
| `2->16->16->1` | 321 | Most common starting point |
| `2->64->64->64->1` | 8,513 | Complex patterns, needs more data |
| `2->256->256->1` | 66,305 | Often overkill for small datasets |
!!! tip "Practical default"
    Start with two hidden layers of 16 neurons each. Only go bigger if the network clearly cannot learn the pattern after proper tuning.

---

## How Networks Learn


---

You have seen what a network is and that adding layers helps. Now: how does a network actually improve? How do 41 random numbers become 41 carefully tuned numbers that solve a problem?
**The blindfolded hiker**

You are blindfolded on a hilly landscape trying to find the lowest valley. You cannot see anything but you can feel the slope under your feet. Each step you take one step in the downhill direction. After enough steps you reach a valley. This is gradient descent — navigating a landscape you cannot see by following the local slope downward.

### The loss --- your altitude
The height in the hiking analogy is the loss. It measures how wrong the network is right now. When predictions are far from the true labels, loss is high. When predictions are close, loss is low. Training is the process of walking downhill on the loss surface until you reach a valley --- a set of weights where the network is as accurate as possible.

### The five steps of training

**Forward pass** — Feed a batch of training examples through the network left to right. Every neuron computes its weighted sum and activation. The final layer produces a prediction for each example in the batch.

**Compute loss** — Compare predictions to true labels. The loss is a single number — the average error across the batch. Large loss means the network is very wrong. Small loss means it is close. In Playground you see this number updating live in the top right.

**Backward pass (backpropagation)** — Calculate how much each weight contributed to the loss. This uses the chain rule from calculus — the network works backward from the output, spreading blame to each weight proportionally. You do not need to understand the calculus. The key insight is: every weight gets a score saying "increase me" or "decrease me."

**Update weights** — Move every weight a small step in the direction that reduces loss: `new_weight = old_weight - learning_rate * gradient`. The learning rate controls how big the step is. Too large: overshoots the valley. Too small: takes forever.

**Repeat** — Do this for every batch in the training set. One complete pass through all training data = one epoch. Typical networks train for 10 to 500 epochs.
!!! tip "Try this before running"
    Before running, predict --- what shape will the loss curve have over 50 epochs? Flat? Always going down? Bumpy? Sketch your prediction on paper, then run.
healthy_training_curve.py
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
epochs = np.arange(1, 51)
loss = 1.2 * np.exp(-0.08 * epochs) + 0.1 + np.random.randn(50) * 0.02

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(epochs, loss, color="#6366f1", linewidth=2, label="Training loss")
ax.axhline(y=0.1, color="#30363d", linestyle="--", label="Irreducible error")
ax.set_xlabel("Epoch", color="#8b949e")
ax.set_ylabel("Loss", color="#8b949e")
ax.set_title("Healthy Training Curve", color="#e6edf3")
ax.set_facecolor("#0d1117")
fig.patch.set_facecolor("#0d1117")
ax.tick_params(colors="#8b949e")
for spine in ax.spines.values():
    spine.set_color("#30363d")
ax.legend(facecolor="#161b22", labelcolor="white")
plt.tight_layout()
plt.show()
```
!!! tip "Question"
    The curve has small random wiggles but a clear downward trend. The wiggles come from randomness in which training examples each batch contains. What do you think would happen to the wiggles if you used all training data in every batch instead of small random batches? Would they get bigger or smaller? Answer: smaller --- but each step would be much slower to compute.

---

##  Playground: Watch It Learn


---

Lesson 6.5 described gradient descent in words. Now watch it happen. Playground shows the loss graph updating in real time and the decision boundary changing as weights shift. You will run three experiments --- each breaks one thing and shows exactly why that setting matters.
[Open in Colab →](https://playground.tensorflow.org)

### Experiment 1 --- A healthy training run
Set up this exact configuration:

- **Dataset:** Circle, the two concentric rings.
- **Features:** `X1` and `X2` only.
- **Hidden layers:** 1 layer, 4 neurons.
- **Activation:** `ReLU`.
- **Learning rate:** `0.03`.
- **Batch size:** `10`.
- **Regularization:** None.
- **Train:** click ▶ and watch until test loss is below `0.05`.
!!! tip "Watch three things simultaneously"
    Each of these is showing you a different view of the same process.

### Experiment 2 --- Learning rate too high

- Click ↺ to reset.
- Change only the learning rate to `3`. That is 100x larger than the healthy run.
- Click ▶ and watch for 10 seconds.
!!! tip "Question"
    What happens to the loss graph? Does it decrease smoothly or oscillate wildly? What does the decision boundary look like? This is the hiking analogy --- taking steps so large you leap over the valley and land on the other side.

### Experiment 3 --- Learning rate too low

- Click ↺ to reset.
- Change only the learning rate to `0.0001`. That is 300x smaller than the healthy run.
- Click ▶ and watch for 30 seconds.
!!! tip "Question"
    How far does the loss decrease after 30 seconds compared to Experiment 1? The network is learning --- just extremely slowly. This is why `0.0001` is rarely used as a starting learning rate.

### Experiment 4 --- Batch size effect

- Click ↺ to reset and restore learning rate to `0.3`.
- Change batch size to `1`. The network now trains from one example at a time.
- Click ▶ and watch the loss graph specifically.
!!! tip "Question"
    The loss graph is now much noisier. Why? Each step is based on only one example instead of 10. One example gives a noisy estimate of the true gradient. How does this compare to the hiking analogy --- is this like taking careful measured steps or like stumbling in a random direction each time?
!!! tip "What you just learned"
    Learning rate and batch size both affect how gradient descent walks down the loss surface. Learning rate controls step size. Batch size controls how accurately the slope is measured at each step. Neither has a universally correct value --- they depend on your specific problem, dataset size, and architecture.
!!! tip "Transition"
    The next lesson moves from Playground to a real experiment: training a network from scratch in Python to learn the sin function. You will control all three knobs --- learning rate, epochs, and architecture --- and watch the effect directly in the training curves and the learned function shape.

---

##  Experiment: Teach a Network to Learn sin(x)


---

###  Experiment: Teach a Network to Learn sin(x)
You know what sin(x) looks like: a smooth wave that rises and falls between -1 and +1. A neural network knows nothing. It starts with random weights and has no concept of waves, curves, or math.
In this experiment you will watch it figure out sin(x) from scratch, purely by being shown `(x, sin(x))` pairs thousands of times and correcting itself each time.
The twist: we train on x from 0 to 2π, then test on 0 to 4π, a range the network has never seen. This directly shows whether the network truly learned the sin function or just memorized the training data.
!!! warning "Run this experiment in Google Colab"
    The training loop is intentionally written from scratch in NumPy. It is too slow for the browser runtime, so use the notebook.
[Open in Colab →](https://colab.research.google.com/github/purwar-lab/ml-for-robotics-/blob/main/notebooks/ch6-sin-experiment.ipynb)
**Challenge 1: Minimum viable network** Find the smallest `HIDDEN_DIM1` and `HIDDEN_DIM2` that still accurately approximates sin(x) on the training range. How few neurons can learn a sine wave?

**Challenge 2: Generalization limits** Set `TRAIN_RANGE=2`, `TEST_RANGE=8`. Does the network generalize to 4x the training range? What about 6x?

**Challenge 3: Too many epochs** Set `EPOCHS=50000`. Does accuracy keep improving? What happens to the generalization plot? This is overfitting in action.

**Challenge 4: Learning rate extremes** Try `LEARNING_RATE=0.5` and `LEARNING_RATE=0.000001`. What does the loss curve look like in each case?

#### Cell 1: Setup and Data Generation
Cell 1: Setup and data generation
```python
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------
# 1. Generate Dataset: Approximating sin(x)
# -----------------------------------------
np.random.seed(42)
# Create 100 data points between 0 and 2 * pi
x = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)
y = np.sin(x)  # True function values
```
| Line | Meaning |
|---|---|
| `np.linspace(0, 2*np.pi, 1000)` | Creates 1000 evenly spaced x values from 0 to 2π. This is the training range: the network only sees this interval. |
| `.reshape(-1, 1)` | Neural networks expect a 2D array where each row is one example. `-1` means Python figures out that dimension automatically, giving shape `(1000, 1)`. |
| `y = np.sin(x)` | The true labels. For each x value, the correct output is sin(x). The network learns this mapping from scratch. |

#### Cell 2: Network Architecture
The architecture is: `1 input -> 20 neurons -> 10 neurons -> 1 output`.
Why these sizes? This is a 1D regression problem: one number in, one number out. The hidden layers need enough neurons to represent a curved wave. Too few neurons and the network cannot capture the shape of sin(x).
Weight initialization matters enormously. We multiply by `0.1` to start with small weights rather than large random ones. Large initial weights can make gradients unstable in the first few training steps.
Cell 2: Network architecture
```python
# -----------------------------------------
# 2. Define the Neural Network Architecture
# -----------------------------------------
# Network dimensions: 1 input, hidden neurons in first layer, hidden neurons in second layer, 1 output
input_dim = 1
hidden_dim1 = 20 # First hidden layer size
hidden_dim2 = 10 # Second hidden layer size
output_dim = 1

# Initialize weights and biases (small random values and zeros)
W1 = np.random.randn(input_dim, hidden_dim1) * 0.1
b1 = np.zeros((1, hidden_dim1))
W2 = np.random.randn(hidden_dim1, hidden_dim2) * 0.1 # Weights for second hidden layer
b2 = np.zeros((1, hidden_dim2)) # Biases for second hidden layer
W3 = np.random.randn(hidden_dim2, output_dim) * 0.1 # Weights for output layer
b3 = np.zeros((1, output_dim)) # Biases for output layer
```
| Parameter | Shape | Meaning |
|---|---|---|
| `W1` | `(1, 20)` | 1 input connects to 20 hidden neurons |
| `W2` | `(20, 10)` | 20 first-layer neurons connect to 10 second-layer neurons |
| `W3` | `(10, 1)` | 10 second-layer neurons connect to 1 output |
| `b1` | `(1, 20)` | one bias per neuron in layer 1 |
| **Total** | `271` | `1x20 + 20 + 20x10 + 10 + 10x1 + 1` parameters |

#### Cell 3: Activation Functions
Why ReLU for hidden layers but linear for the output?
ReLU gives the network the ability to learn non-linear shapes. Without it, the 20-neuron hidden layer collapses to the same as one neuron: just a weighted sum, which is a straight line. No straight line fits sin(x).
Linear output is used because sin(x) can be any value between -1 and +1. Sigmoid would clamp outputs to 0-1. ReLU could not go negative. Linear output lets the network predict any real number.
Cell 3: Activation functions
```python
# -----------------------------------------
# 3. Define Activation Functions and Their Derivatives
# -----------------------------------------
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def linear(x):
    # Identity function for regression output
    return x

def linear_derivative(x):
    # Derivative of a linear function is 1
    return 1
```

#### Cell 4: Adam Optimizer and Training Loop
This is the most important cell. It performs the forward pass, computes loss, sends the gradient backward through every layer, and updates every weight with Adam.
| Adam parameter | Meaning |
|---|---|
| `beta1 = 0.9` | Momentum: keep 90% of the previous gradient direction and mix in 10% of the new gradient. |
| `beta2 = 0.999` | Tracks a moving average of squared gradients, which Adam uses to scale each parameter update. |
| `epsilon = 1e-8` | A tiny number that prevents division by zero. |

| Forward-pass line | Meaning |
|---|---|
| `z1 = np.dot(x, W1) + b1` | Weighted sum for layer 1. |
| `a1 = relu(z1)` | Apply non-linearity. |
| `z2 = np.dot(a1, W2) + b2` | Weighted sum for layer 2. |
| `a2 = relu(z2)` | Second hidden activation. |
| `z3 = np.dot(a2, W3) + b3` | Final weighted sum. |
| `a3 = linear(z3)` | Regression output with no activation. |

| Backprop line | Meaning |
|---|---|
| `d_loss_a3 = 2*(a3-y)/len(x)` | Gradient of MSE with respect to the prediction. It points uphill, so updates move the opposite way. |
| `delta3 = d_loss_a3 * linear_derivative(a3)` | Chain rule at the output layer. Since the derivative of a linear function is 1, this is included for clarity. |
| `dW3 = np.dot(a2.T, delta3)` | Gradient for the third-layer weights. |
| `delta2 = np.dot(delta3, W3.T) * relu_derivative(z2)` | Error signal propagated backward through layer 3, then through layer 2 ReLU. |

| Adam update line | Meaning |
|---|---|
| `m_W1 = beta1*m_W1 + (1-beta1)*dW1` | Update the moving average of the gradient. This is Adam momentum. |
| `v_W1 = beta2*v_W1 + (1-beta2)*(dW1**2)` | Update the moving average of squared gradients. Large recent gradients slow future updates. |
| `m_W1_hat = m_W1 / (1-beta1**epoch)` `v_W1_hat = v_W1 / (1-beta2**epoch)` | Bias correction. Early moment estimates start near zero, so Adam corrects them. |
| `W1 -= learning_rate * m_W1_hat / (np.sqrt(v_W1_hat) + epsilon)` | The actual adaptive update: learning rate scaled by momentum and squared-gradient history. |
Cell 4: Adam optimizer and training loop
```python
# -------------------------------
# 4. Define the Loss Function (MSE)
# -------------------------------
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# -------------------------------
# 5. Training the Neural Network with Adam Optimizer
# -------------------------------
# Adam optimizer parameters
adam_learning_rate = 0.001 # Base learning rate for Adam
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# Initialize Adam's first and second moment vectors for each parameter
m_W1, v_W1 = np.zeros_like(W1), np.zeros_like(W1)
m_b1, v_b1 = np.zeros_like(b1), np.zeros_like(b1)
m_W2, v_W2 = np.zeros_like(W2), np.zeros_like(W2) # Moments for W2
m_b2, v_b2 = np.zeros_like(b2), np.zeros_like(b2) # Moments for b2
m_W3, v_W3 = np.zeros_like(W3), np.zeros_like(W3) # Moments for W3
m_b3, v_b3 = np.zeros_like(b3), np.zeros_like(b3) # Moments for b3

epochs = 5000       # Number of training epochs
loss_history = []

for epoch in range(1, epochs + 1): # Start epoch from 1 for Adam bias correction
    # ---- Forward Pass ----
    z1 = np.dot(x, W1) + b1        # Linear combination for first hidden layer
    a1 = relu(z1)                 # Activation using relu
    z2 = np.dot(a1, W2) + b2        # Linear combination for second hidden layer
    a2 = relu(z2)                 # Activation using relu for second hidden layer
    z3 = np.dot(a2, W3) + b3        # Linear combination for output layer
    a3 = linear(z3)               # Output (prediction)

    # ---- Compute Loss ----
    loss = mse_loss(y, a3) # Use a3 for loss computation
    loss_history.append(loss)

    # ---- Backward Pass ----
    # Compute gradient of loss with respect to output predictions
    d_loss_a3 = 2 * (a3 - y) / len(x)

    # For the output layer:
    delta3 = d_loss_a3 * linear_derivative(a3)
    dW3 = np.dot(a2.T, delta3)
    db3 = np.sum(delta3, axis=0, keepdims=True)

    # Backpropagate to second hidden layer
    delta2 = np.dot(delta3, W3.T) * relu_derivative(z2)
    dW2 = np.dot(a1.T, delta2)
    db2 = np.sum(delta2, axis=0, keepdims=True)

    # Backpropagate to first hidden layer
    delta1 = np.dot(delta2, W2.T) * relu_derivative(z1)
    dW1 = np.dot(x.T, delta1)
    db1 = np.sum(delta1, axis=0, keepdims=True)

    # ---- Update parameters using Adam ----
    # Update W1
    m_W1 = beta1 * m_W1 + (1 - beta1) * dW1
    v_W1 = beta2 * v_W1 + (1 - beta2) * (dW1 ** 2)
    m_W1_hat = m_W1 / (1 - beta1 ** epoch)
    v_W1_hat = v_W1 / (1 - beta2 ** epoch)
    W1 -= adam_learning_rate * m_W1_hat / (np.sqrt(v_W1_hat) + epsilon)

    # Update b1
    m_b1 = beta1 * m_b1 + (1 - beta1) * db1
    v_b1 = beta2 * v_b1 + (1 - beta2) * (db1 ** 2)
    m_b1_hat = m_b1 / (1 - beta1 ** epoch)
    v_b1_hat = v_b1 / (1 - beta2 ** epoch)
    b1 -= adam_learning_rate * m_b1_hat / (np.sqrt(v_b1_hat) + epsilon)

    # Update W2
    m_W2 = beta1 * m_W2 + (1 - beta1) * dW2
    v_W2 = beta2 * v_W2 + (1 - beta2) * (dW2 ** 2)
    m_W2_hat = m_W2 / (1 - beta1 ** epoch)
    v_W2_hat = v_W2 / (1 - beta2 ** epoch)
    W2 -= adam_learning_rate * m_W2_hat / (np.sqrt(v_W2_hat) + epsilon)

    # Update b2
    m_b2 = beta1 * m_b2 + (1 - beta1) * db2
    v_b2 = beta2 * v_b2 + (1 - beta2) * (db2 ** 2)
    m_b2_hat = m_b2 / (1 - beta1 ** epoch)
    v_b2_hat = v_b2 / (1 - beta2 ** epoch)
    b2 -= adam_learning_rate * m_b2_hat / (np.sqrt(v_b2_hat) + epsilon)

    # Update W3
    m_W3 = beta1 * m_W3 + (1 - beta1) * dW3
    v_W3 = beta2 * v_W3 + (1 - beta2) * (dW3 ** 2)
    m_W3_hat = m_W3 / (1 - beta1 ** epoch)
    v_W3_hat = v_W3 / (1 - beta2 ** epoch)
    W3 -= adam_learning_rate * m_W3_hat / (np.sqrt(v_W3_hat) + epsilon)

    # Update b3
    m_b3 = beta1 * m_b3 + (1 - beta1) * db3
    v_b3 = beta2 * v_b3 + (1 - beta2) * (db3 ** 2)
    m_b3_hat = m_b3 / (1 - beta1 ** epoch)
    v_b3_hat = v_b3 / (1 - beta2 ** epoch)
    b3 -= adam_learning_rate * m_b3_hat / (np.sqrt(v_b3_hat) + epsilon)

    # Optionally print the loss every 500 epochs
    if epoch % 500 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {loss:.5f}")

# -------------------------------
# 6. Visualizing Training Loss and Predictions
# -------------------------------
# Plot the loss curve
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.title("Training Loss (MSE) with Adam (2 Hidden Layers, ReLU)")
plt.xlabel("Epoch")
plt.ylabel("Loss")

# Plot the actual sine function vs. the network's predictions
plt.subplot(1, 2, 2)
plt.scatter(x, y, label="Actual sin(x)", color='blue')
plt.scatter(x, a3, label="Predicted", color='red', marker='x') # Use a3 for predictions
plt.title("Neural Network Regression with Adam (2 Hidden Layers, ReLU)")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.legend()
plt.tight_layout()
plt.show()
```

#### Cell 5: Generalization Test from 0 to 4π
The network was trained on x from 0 to 2π. Now we ask it to predict sin(x) for x from 0 to 4π: values it has never seen.
This is the most important test in the chapter. Good generalization means the prediction follows sin(x) for the full 4π range. Bad generalization means it matches well from 0 to 2π then diverges beyond 2π.
Cell 5a: Wider range evaluation
```python
# -----------------------------------------------------
# 7. Evaluate and Visualize Predictions on a Wider Range
# -----------------------------------------------------

# Create a new, wider range for evaluation
x_plot = np.linspace(0, 4 * np.pi, 1000).reshape(-1, 1)

# True sine values for the wider range
y_true_plot = np.sin(x_plot)

# Perform forward pass with the trained weights on the new range
z1_plot = np.dot(x_plot, W1) + b1
a1_plot = relu(z1_plot)
z2_plot = np.dot(a1_plot, W2) + b2
a2_plot = relu(z2_plot)
z3_plot = np.dot(a2_plot, W3) + b3
y_pred_plot = linear(z3_plot)

print("Generated predictions for the wider range [0, 4 * pi].")
```
Cell 5b: Wider range plot
```python
# Plot the actual sine function vs. the network's predictions for the wider range
plt.figure(figsize=(10, 6))
plt.scatter(x_plot, y_true_plot, label="Actual sin(x) (Wider Range)", color='blue', alpha=0.7, s=5)
plt.scatter(x_plot, y_pred_plot, label="Predicted (Wider Range)", color='red', marker='.', alpha=0.7, s=5)
plt.title("Neural Network Regression on Wider Range [0, 4π]")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```
!!! tip "Challenge"
    Can you make it generalize even further? Try training on 0 to 2π and testing on 0 to 6π. What is the smallest network that still generalizes correctly? Does adding more epochs help or hurt beyond the training range?

#### Cell 6: Learning Rate Comparison
The three learning rates, `0.0001`, `0.001`, and `0.01`, are all reasonable values. But they produce very different behavior.
Before running, predict what will happen: Is `0.0001` fast or slow? Is `0.01` stable or unstable? Which reaches the lowest loss after 5000 epochs?
Cell 6a: Learning-rate training helper
```python
def train_with_learning_rate(learning_rate, epochs, x, y, input_dim, hidden_dim1, hidden_dim2, output_dim):
    # Re-initialize weights and biases for each run
    W1 = np.random.randn(input_dim, hidden_dim1) * 0.1
    b1 = np.zeros((1, hidden_dim1))
    W2 = np.random.randn(hidden_dim1, hidden_dim2) * 0.1
    b2 = np.zeros((1, hidden_dim2))
    W3 = np.random.randn(hidden_dim2, output_dim) * 0.1
    b3 = np.zeros((1, output_dim))

    # Adam optimizer parameters (re-initialize moments)
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    m_W1, v_W1 = np.zeros_like(W1), np.zeros_like(W1)
    m_b1, v_b1 = np.zeros_like(b1), np.zeros_like(b1)
    m_W2, v_W2 = np.zeros_like(W2), np.zeros_like(W2)
    m_b2, v_b2 = np.zeros_like(b2), np.zeros_like(b2)
    m_W3, v_W3 = np.zeros_like(W3), np.zeros_like(W3)
    m_b3, v_b3 = np.zeros_like(b3), np.zeros_like(b3)

    loss_history_lr = []

    for epoch in range(1, epochs + 1):
        # Forward Pass
        z1 = np.dot(x, W1) + b1
        a1 = relu(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = relu(z2)
        z3 = np.dot(a2, W3) + b3
        a3 = linear(z3)

        # Compute Loss
        loss = mse_loss(y, a3)
        loss_history_lr.append(loss)

        # Backward Pass
        d_loss_a3 = 2 * (a3 - y) / len(x)
        delta3 = d_loss_a3 * linear_derivative(a3)
        dW3 = np.dot(a2.T, delta3)
        db3 = np.sum(delta3, axis=0, keepdims=True)

        delta2 = np.dot(delta3, W3.T) * relu_derivative(z2)
        dW2 = np.dot(a1.T, delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)

        delta1 = np.dot(delta2, W2.T) * relu_derivative(z1)
        dW1 = np.dot(x.T, delta1)
        db1 = np.sum(delta1, axis=0, keepdims=True)

        # Update parameters using Adam
        # Update W1
        m_W1 = beta1 * m_W1 + (1 - beta1) * dW1
        v_W1 = beta2 * v_W1 + (1 - beta2) * (dW1 ** 2)
        m_W1_hat = m_W1 / (1 - beta1 ** epoch)
        v_W1_hat = v_W1 / (1 - beta2 ** epoch)
        W1 -= learning_rate * m_W1_hat / (np.sqrt(v_W1_hat) + epsilon)

        # Update b1
        m_b1 = beta1 * m_b1 + (1 - beta1) * db1
        v_b1 = beta2 * v_b1 + (1 - beta2) * (db1 ** 2)
        m_b1_hat = m_b1 / (1 - beta1 ** epoch)
        v_b1_hat = v_b1 / (1 - beta2 ** epoch)
        b1 -= learning_rate * m_b1_hat / (np.sqrt(v_b1_hat) + epsilon)

        # Update W2
        m_W2 = beta1 * m_W2 + (1 - beta1) * dW2
        v_W2 = beta2 * v_W2 + (1 - beta2) * (dW2 ** 2)
        m_W2_hat = m_W2 / (1 - beta1 ** epoch)
        v_W2_hat = v_W2 / (1 - beta2 ** epoch)
        W2 -= learning_rate * m_W2_hat / (np.sqrt(v_W2_hat) + epsilon)

        # Update b2
        m_b2 = beta1 * m_b2 + (1 - beta1) * db2
        v_b2 = beta2 * v_b2 + (1 - beta2) * (db2 ** 2)
        m_b2_hat = m_b2 / (1 - beta1 ** epoch)
        v_b2_hat = v_b2 / (1 - beta2 ** epoch)
        b2 -= learning_rate * m_b2_hat / (np.sqrt(v_b2_hat) + epsilon)

        # Update W3
        m_W3 = beta1 * m_W3 + (1 - beta1) * dW3
        v_W3 = beta2 * v_W3 + (1 - beta2) * (dW3 ** 2)
        m_W3_hat = m_W3 / (1 - beta1 ** epoch)
        v_W3_hat = v_W3 / (1 - beta2 ** epoch)
        W3 -= learning_rate * m_W3_hat / (np.sqrt(v_W3_hat) + epsilon)

        # Update b3
        m_b3 = beta1 * m_b3 + (1 - beta1) * db3
        v_b3 = beta2 * v_b3 + (1 - beta2) * (db3 ** 2)
        m_b3_hat = m_b3 / (1 - beta1 ** epoch)
        v_b3_hat = v_b3 / (1 - beta2 ** epoch)
        b3 -= learning_rate * m_b3_hat / (np.sqrt(v_b3_hat) + epsilon)

    return loss_history_lr, W1, b1, W2, b2, W3, b3
```
Cell 6b: Compare learning rates
```python
# Define the learning rates to test
learning_rates_to_test = [0.0001, 0.001, 0.01]
epochs_for_test = 5000 # Increased epochs for longer training

plt.figure(figsize=(10, 6))

for lr in learning_rates_to_test:
    print(f"Training with learning rate: {lr}")
    loss_history_current_lr, W1_final, b1_final, W2_final, b2_final, W3_final, b3_final = train_with_learning_rate(lr, epochs_for_test, x, y, input_dim, hidden_dim1, hidden_dim2, output_dim)
    plt.plot(loss_history_current_lr, label=f'LR = {lr}')

plt.title("Training Loss Curves for Different Learning Rates (Adam Optimizer)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()


# Plot the predicted sine function for each learning rate
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label="Actual sin(x)", color='blue', alpha=0.7, s=5)

for lr in learning_rates_to_test:
    # Retrain to get the final weights for this LR (or store them from the previous loop)
    _, W1_final, b1_final, W2_final, b2_final, W3_final, b3_final = train_with_learning_rate(lr, epochs_for_test, x, y, input_dim, hidden_dim1, hidden_dim2, output_dim)

    # Perform forward pass with the trained weights
    z1_pred = np.dot(x, W1_final) + b1_final
    a1_pred = relu(z1_pred)
    z2_pred = np.dot(a1_pred, W2_final) + b2_final
    a2_pred = relu(z2_pred)
    z3_pred = np.dot(a2_pred, W3_final) + b3_final
    y_pred_lr = linear(z3_pred)

    plt.plot(x, y_pred_lr, label=f'Predicted (LR = {lr})', linestyle='--')

plt.title("Predicted Sine Function for Different Learning Rates")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.legend()
plt.grid(True)
plt.show()
```
!!! tip "What you should observe"
    `0.0001` converges very slowly. `0.001`, Adam's default, converges cleanly and usually follows sin(x) closely. `0.01` often improves quickly at first but can be noisier or overshoot the minimum.

#### Cell 7: Your Turn
This final cell lets you modify architecture, learning rate, epochs, training range, and test range in one place. It trains a fresh network and shows both the loss curve and the generalization plot.
Cell 7: Student experiment cell
```python
import numpy as np
import matplotlib.pyplot as plt

# -- YOUR TURN - CHANGE THESE AND RE-RUN ----------------------------
HIDDEN_DIM1   = 20      # try: 2, 5, 20, 100
HIDDEN_DIM2   = 10      # try: 2, 5, 10, 50
LEARNING_RATE = 0.001   # try: 0.0001, 0.001, 0.01, 0.1
EPOCHS        = 3000    # try: 100, 500, 3000, 10000
TRAIN_RANGE   = 2       # multiples of pi - try: 1, 2, 4
TEST_RANGE    = 4       # multiples of pi - try: 4, 6, 8
# -------------------------------------------------------------------

np.random.seed(42)
x = np.linspace(0, TRAIN_RANGE * np.pi, 1000).reshape(-1, 1)
y = np.sin(x)

input_dim = 1
output_dim = 1

W1 = np.random.randn(input_dim, HIDDEN_DIM1) * 0.1
b1 = np.zeros((1, HIDDEN_DIM1))
W2 = np.random.randn(HIDDEN_DIM1, HIDDEN_DIM2) * 0.1
b2 = np.zeros((1, HIDDEN_DIM2))
W3 = np.random.randn(HIDDEN_DIM2, output_dim) * 0.1
b3 = np.zeros((1, output_dim))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def linear(x):
    return x

def linear_derivative(x):
    return 1

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

m_W1, v_W1 = np.zeros_like(W1), np.zeros_like(W1)
m_b1, v_b1 = np.zeros_like(b1), np.zeros_like(b1)
m_W2, v_W2 = np.zeros_like(W2), np.zeros_like(W2)
m_b2, v_b2 = np.zeros_like(b2), np.zeros_like(b2)
m_W3, v_W3 = np.zeros_like(W3), np.zeros_like(W3)
m_b3, v_b3 = np.zeros_like(b3), np.zeros_like(b3)

loss_history = []

for epoch in range(1, EPOCHS + 1):
    z1 = np.dot(x, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = relu(z2)
    z3 = np.dot(a2, W3) + b3
    a3 = linear(z3)

    loss = mse_loss(y, a3)
    loss_history.append(loss)

    d_loss_a3 = 2 * (a3 - y) / len(x)
    delta3 = d_loss_a3 * linear_derivative(a3)
    dW3 = np.dot(a2.T, delta3)
    db3 = np.sum(delta3, axis=0, keepdims=True)

    delta2 = np.dot(delta3, W3.T) * relu_derivative(z2)
    dW2 = np.dot(a1.T, delta2)
    db2 = np.sum(delta2, axis=0, keepdims=True)

    delta1 = np.dot(delta2, W2.T) * relu_derivative(z1)
    dW1 = np.dot(x.T, delta1)
    db1 = np.sum(delta1, axis=0, keepdims=True)

    m_W1 = beta1 * m_W1 + (1 - beta1) * dW1
    v_W1 = beta2 * v_W1 + (1 - beta2) * (dW1 ** 2)
    W1 -= LEARNING_RATE * (m_W1 / (1 - beta1 ** epoch)) / (np.sqrt(v_W1 / (1 - beta2 ** epoch)) + epsilon)

    m_b1 = beta1 * m_b1 + (1 - beta1) * db1
    v_b1 = beta2 * v_b1 + (1 - beta2) * (db1 ** 2)
    b1 -= LEARNING_RATE * (m_b1 / (1 - beta1 ** epoch)) / (np.sqrt(v_b1 / (1 - beta2 ** epoch)) + epsilon)

    m_W2 = beta1 * m_W2 + (1 - beta1) * dW2
    v_W2 = beta2 * v_W2 + (1 - beta2) * (dW2 ** 2)
    W2 -= LEARNING_RATE * (m_W2 / (1 - beta1 ** epoch)) / (np.sqrt(v_W2 / (1 - beta2 ** epoch)) + epsilon)

    m_b2 = beta1 * m_b2 + (1 - beta1) * db2
    v_b2 = beta2 * v_b2 + (1 - beta2) * (db2 ** 2)
    b2 -= LEARNING_RATE * (m_b2 / (1 - beta1 ** epoch)) / (np.sqrt(v_b2 / (1 - beta2 ** epoch)) + epsilon)

    m_W3 = beta1 * m_W3 + (1 - beta1) * dW3
    v_W3 = beta2 * v_W3 + (1 - beta2) * (dW3 ** 2)
    W3 -= LEARNING_RATE * (m_W3 / (1 - beta1 ** epoch)) / (np.sqrt(v_W3 / (1 - beta2 ** epoch)) + epsilon)

    m_b3 = beta1 * m_b3 + (1 - beta1) * db3
    v_b3 = beta2 * v_b3 + (1 - beta2) * (db3 ** 2)
    b3 -= LEARNING_RATE * (m_b3 / (1 - beta1 ** epoch)) / (np.sqrt(v_b3 / (1 - beta2 ** epoch)) + epsilon)

x_test = np.linspace(0, TEST_RANGE * np.pi, 1000).reshape(-1, 1)
y_test = np.sin(x_test)

z1_test = np.dot(x_test, W1) + b1
a1_test = relu(z1_test)
z2_test = np.dot(a1_test, W2) + b2
a2_test = relu(z2_test)
z3_test = np.dot(a2_test, W3) + b3
y_pred_test = linear(z3_test)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
ax1.plot(loss_history)
ax1.set_title("Training Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("MSE")

ax2.plot(x_test, y_test, label="Actual sin(x)")
ax2.plot(x_test, y_pred_test, label="Network prediction", linestyle="--")
ax2.axvline(TRAIN_RANGE * np.pi, color="gray", linestyle=":", label="End of training range")
ax2.set_title(f"Train: 0 to {TRAIN_RANGE}pi | Test: 0 to {TEST_RANGE}pi")
ax2.set_xlabel("x")
ax2.set_ylabel("sin(x)")
ax2.legend()
plt.tight_layout()
plt.show()

print(f"Final training loss: {loss_history[-1]:.6f}")
print(f"Total parameters: {input_dim*HIDDEN_DIM1 + HIDDEN_DIM1 + HIDDEN_DIM1*HIDDEN_DIM2 + HIDDEN_DIM2 + HIDDEN_DIM2*output_dim + output_dim}")
```

---

## Activation Functions


---

### Activation Functions
Activation functions create non-linearity. Without them, stacking layers still behaves like one linear model.
**Sigmoid**
Outputs 0 to 1. Good for binary classifier output layers. Can suffer vanishing gradients.
**ReLU**
Outputs 0 to infinity. Default choice for hidden layers because it trains fast.
**Tanh**
Outputs -1 to 1. Similar to sigmoid but zero-centered.
activation_function_comparison.py
```python
import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-5, 5, 300)
sigmoid = 1 / (1 + np.exp(-z))
relu = np.maximum(0, z)
tanh = np.tanh(z)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.patch.set_facecolor("#0d1117")

for ax, fn, name, color, note in zip(
    axes,
    [sigmoid, relu, tanh],
    ["Sigmoid", "ReLU", "Tanh"],
    ["#6366f1", "#f97316", "#10b981"],
    ["Output: 0 to 1\nGood for output layer",
     "Output: 0 to infinity\nDefault for hidden layers",
     "Output: -1 to 1\nZero-centered sigmoid"],
):
    ax.plot(z, fn, color=color, linewidth=2.5)
    ax.axhline(0, color="#30363d", linewidth=0.8)
    ax.axvline(0, color="#30363d", linewidth=0.8)
    ax.set_title(name, color="#e6edf3", fontsize=13)
    ax.set_xlabel("z", color="#8b949e")
    ax.text(0.05, 0.08, note, transform=ax.transAxes, color="#8b949e", fontsize=8.5)
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")

plt.tight_layout()
plt.show()
```
!!! tip "Practical rule"
    Hidden layers: ReLU. Binary output: sigmoid. Multi-class output: softmax. Regression output: linear.

---

## Overfitting and How to Fight It


---

Imagine you are studying for an exam. You have 10 past papers. You memorize every single answer to every single question on all 10 papers. On exam day the teacher uses a new paper with slightly different questions. You fail - not because you are bad at the subject, but because you memorized instead of understanding.
Overfitting is exactly this. The network memorizes the training data - every detail, every quirk, every piece of noise - instead of learning the underlying pattern. It performs brilliantly on training examples and terribly on new ones. The training loss drops to near zero. The validation loss gets worse. The network has learned nothing useful.
Underfitting is the opposite problem. The network is too simple to capture the pattern at all. Both training and validation loss stay high. The network is not even good at the training data. In between these two failure modes is the good fit - where the network has learned the real pattern and generalizes to new data.

### Seeing the three cases
Run the code below. It generates simulated training curves for all three cases side by side. You are looking at loss over epochs: purple is training loss, orange is validation loss.
!!! tip "Try this before running"
    Before running, predict what the overfitting curve looks like. If the network is memorizing the training data, what happens to training loss over time? What about validation loss? Write your prediction, then run.
underfit_goodfit_overfit_curves.py
```python
import numpy as np
import matplotlib.pyplot as plt

epochs = np.arange(1, 101)
train_under = 0.9 - 0.2 * np.log(epochs / 10 + 1) + np.random.randn(100) * 0.01
val_under = train_under + 0.05 + np.random.randn(100) * 0.01
train_good = 0.8 * np.exp(-0.05 * epochs) + 0.12 + np.random.randn(100) * 0.01
val_good = 0.8 * np.exp(-0.04 * epochs) + 0.15 + np.random.randn(100) * 0.015
train_over = 0.8 * np.exp(-0.08 * epochs) + 0.02 + np.random.randn(100) * 0.005
val_over = 0.8 * np.exp(-0.05 * epochs) + 0.15 + 0.004 * epochs + np.random.randn(100) * 0.015

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
fig.patch.set_facecolor("#0d1117")

for ax, (tl, vl), title in zip(
    axes,
    [(train_under, val_under), (train_good, val_good), (train_over, val_over)],
    ["Underfitting", "Good Fit", "Overfitting"],
):
    ax.plot(epochs, tl, color="#6366f1", linewidth=2, label="Train")
    ax.plot(epochs, vl, color="#f97316", linewidth=2, linestyle="--", label="Val")
    ax.set_title(title, color="#e6edf3", fontsize=12)
    ax.set_xlabel("Epoch", color="#8b949e")
    ax.set_ylabel("Loss", color="#8b949e")
    ax.set_ylim(0, 1)
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.legend(facecolor="#161b22", labelcolor="white", fontsize=9)

plt.tight_layout()
plt.show()
```

### How to read each curve
**Underfitting panel:** Both training and validation loss are high and barely decreasing. The network is too simple - it does not have enough parameters to represent the pattern in the data. The cure is a larger network, more epochs, or a higher learning rate. This looks like a student who did not study enough.
**Good Fit panel:** Training loss decreases and levels off. Validation loss follows closely and also levels off at a slightly higher value. The gap between train and validation loss is small and stable. This is what you want. The network learned the real pattern, not the noise.
**Overfitting panel:** Training loss keeps decreasing toward zero - the network is getting better and better on the training data. But validation loss starts decreasing, then turns around and starts going up. This is the signature of overfitting. The moment the validation curve turns upward is when the network stopped learning patterns and started memorizing training examples.
!!! warning "Watch the validation loss"
    The validation loss turning upward is the most important signal in all of machine learning. When you see this in your own training runs, stop training immediately and use the weights from before the upturn.

### Why does overfitting happen?
A network with many parameters has enormous capacity. Given enough training time, it can memorize every individual training example perfectly - including the random noise in the data. A network with 10,000 parameters trained on 100 examples is almost guaranteed to overfit. The network has 100 examples to explain with 10,000 numbers. It has no difficulty at all finding weights that fit the training data exactly.
Overfitting gets worse as your network gets bigger relative to your dataset size. A good rule of thumb: you need at least 10 to 100 training examples per parameter. A 41-parameter network needs at least a few hundred training examples. A network with 100,000 parameters needs tens of thousands.
!!! info "The train/val split exists for this reason"
    Remember splitting data 80/20 in Chapter 3? The 20% validation set exists specifically to catch overfitting. If you only looked at training loss, you would never know the network was memorizing instead of learning. Validation loss on data the network has never seen is the only honest measure of whether training is working.

### Three ways to fight overfitting
**Dropout**
During training, dropout randomly switches off a percentage of neurons on each forward pass. The network cannot rely on any single neuron because it might be off in the next pass. This forces the network to learn redundant, robust representations instead of fragile memorized paths.
Think of it like training a sports team where random players sit out each practice. The team cannot rely on one star player - everyone has to be capable.
The strength is one number - the fraction of neurons to drop. Around 30% is common for hidden layers, and you never apply it to the output layer.
**Early Stopping**
Train until validation loss starts getting worse, then stop and keep the weights from the best validation epoch. This is the simplest and often most effective technique.
You give it a patience - how many epochs to wait without improvement before giving up. Wait too little and you stop on a temporary bump; wait too long and you waste time memorizing. When it stops, you go back to the best-ever weights, not the final ones.
It is just an automatic version of the rule from the warning above: watch the validation loss, and when it turns upward, go back to where it was lowest.
**More Data**
The most reliable cure. More diverse training examples means the network has to learn the real pattern - it cannot memorize 300 unique examples the way it could memorize 50.
When you cannot collect more real data, data augmentation creates variations of existing examples: flip, rotate, add noise, scale. This is why YOLO training uses augmentation - it multiplies the effective dataset size without collecting new images.
Sometimes more data is not possible. Then dropout and early stopping are your best tools.
!!! info "The code comes next"
    Dropout and early stopping are each a single line once you are using a real framework. You will write them for the first time in the next lesson, **6.10 From NumPy to Keras**. For now, focus on what each one does and why - the how is one lesson away.

### Watch overfitting happen in Playground

Open TensorFlow Playground: https://playground.tensorflow.org

- **Dataset:** Circle.
- **Hidden layers:** 3 layers, 8 neurons each. This is overkill for the circle problem — intentionally too large.
- **Noise:** set to `50` to add noise to the data.
- **Training to test ratio:** set to `10`. This means only 10% training data, a very small dataset for such a large network.
- **Train:** click ▶ and train for several hundred epochs.
!!! tip "Observe"
    Watch the test loss, the orange line on the graph. Does it decrease steadily, or does it decrease then start rising? The boundary in the right panel probably fits the training data very tightly - too tightly. It is wrapping itself around individual training points, including the noisy ones.
!!! tip "Question"
    Now reduce the network to 1 hidden layer with 4 neurons and reset. Does the test loss behave differently? The smaller network cannot memorize - it only has enough capacity to learn the rough circular pattern.
!!! tip "The central tension in all of machine learning"
    Every neural network sits somewhere on the spectrum from underfitting to overfitting. Too simple and it cannot learn. Too complex and it memorizes. Finding the right point is the core challenge of building ML models. The tools in this lesson - validation loss, dropout, early stopping - are how you navigate that tension in practice.

---

## From NumPy to Keras


---

So far you have built networks by hand in NumPy - every dot product, every activation, every line of the Adam update written out yourself. That was the point: you now know exactly what a neural network does. But nobody trains real networks that way. In practice you use a framework, and the most common one for beginners is **Keras**.

### What is Keras?
Keras is a high-level deep learning library. "High-level" means it lets you describe a network in terms of *what* you want - "a layer of 20 neurons with ReLU" - instead of *how* to compute it. It then handles the forward pass, the backpropagation, and the weight updates for you. The math you wrote by hand in the last few lessons is still happening; Keras just writes it for you.
Keras runs on top of **TensorFlow**, Google's machine learning engine. TensorFlow does the heavy numerical work (and can run it on a GPU); Keras is the friendly layer you actually type against. When you install one modern package you get both.

### Its philosophy
Keras was designed around one idea: **make the common things easy**. A network is described as a stack of layers, training is one function call, and sensible defaults are chosen for you. The goal is that the code you write looks like the picture of the network in your head - layers stacked top to bottom - with as little boilerplate as possible.
- **Stack layers** to define the architecture.
- **Compile** the model with a loss function and an optimizer.
- **Fit** it to your data - this runs the whole training loop you wrote by hand.
Those three steps - define, compile, fit - are the entire workflow, and you will see them in every Keras program for the rest of this course.

### Installing it
If you are using **Google Colab**, TensorFlow and Keras are already installed - you do not have to do anything. To run it on your own machine instead, install it once with pip:
Install TensorFlow (includes Keras)
```bash
pip install tensorflow
```
That single package gives you both TensorFlow and Keras. You then import Keras in Python like this:
Importing Keras
```python
import tensorflow as tf
from tensorflow import keras
```
!!! tip "Don't relearn the network - relearn the syntax"
    You already understand what every piece does. This lesson is not new theory; it is the same sine network you trained in NumPy, written the way professionals write it. Read each Keras line and ask: "which part of my hand-written code does this replace?"

### The same network, two ways
The math is identical to the sin(x) experiment. Only the syntax changes.
sin.ipynb NumPy version
```python
# Forward pass
z1 = np.dot(x, W1) + b1
a1 = relu(z1)
z2 = np.dot(a1, W2) + b2
a2 = relu(z2)
z3 = np.dot(a2, W3) + b3
a3 = linear(z3)

# Manual weight update (Adam)
m_W1 = beta1*m_W1 + (1-beta1)*dW1
...
```

Keras equivalent
```python
# Define the network
model = keras.Sequential([
    keras.layers.Dense(20, activation='relu', input_shape=(1,)),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(1)    # linear = no activation
])

# Compile (Adam is built in)
model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss='mse')

# Train (forward + backward + update all automatic)
model.fit(x, y, epochs=3000, verbose=0)
```
!!! info "What Keras hides"
    Keras does everything in `sin.ipynb`'s training loop automatically. The architecture is the same. The Adam optimizer is the same. The MSE loss is the same. Keras hides the implementation so you can focus on the design rather than the math.
    Now that you have implemented Adam by hand and watched the gradient flow backward through the layers, you know what Keras is doing under the hood. That understanding separates someone who uses Keras from someone who understands it.
[Open in Colab →](https://colab.research.google.com/github/purwar-lab/ml-for-robotics-/blob/main/notebooks/ch6-first-network.ipynb)
| Keras piece | Manual NumPy equivalent |
|---|---|
| `Dense(20, activation="relu")` | `W1`, `b1`, `np.dot`, and `relu` |
| `loss="mse"` | `mse_loss(y, a3)` |
| `Adam(0.001)` | The full Adam moment, bias-correction, and update block from lesson 6.7 |
| `model.fit(...)` | Forward pass, loss, backpropagation, update, repeat |
Cell 1: Generate sin(x) data
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(42)
x = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)
y = np.sin(x)

X_train, X_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=42
)

print(f"Train: {X_train.shape}  Val: {X_val.shape}")
```
Cell 2: Build the Keras sine network
```python
import tensorflow as tf
from tensorflow import keras

LEARNING_RATE = 0.001
EPOCHS = 3000
HIDDEN_DIM1 = 20
HIDDEN_DIM2 = 10

model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    keras.layers.Dense(HIDDEN_DIM1, activation="relu"),
    keras.layers.Dense(HIDDEN_DIM2, activation="relu"),
    keras.layers.Dense(1),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="mse",
    metrics=["mae"],
)

model.summary()
```
Cell 3: Train and plot generalization
```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=32,
    verbose=0,
)

x_plot = np.linspace(0, 4 * np.pi, 1000).reshape(-1, 1)
y_true_plot = np.sin(x_plot)
y_pred_plot = model.predict(x_plot, verbose=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

ax1.plot(history.history["loss"], label="Train loss")
ax1.plot(history.history["val_loss"], label="Val loss", linestyle="--")
ax1.set_title("MSE Loss")
ax1.set_xlabel("Epoch")
ax1.legend()

ax2.plot(x_plot, y_true_plot, label="Actual sin(x)")
ax2.plot(x_plot, y_pred_plot, label="Keras prediction", linestyle="--")
ax2.axvline(2 * np.pi, color="gray", linestyle=":", label="End of training range")
ax2.set_title("Generalization: 0 to 4pi")
ax2.set_xlabel("x")
ax2.set_ylabel("sin(x)")
ax2.legend()

plt.tight_layout()
plt.show()

print(f"Final val loss: {history.history['val_loss'][-1]:.6f}")
print(f"Final val MAE:  {history.history['val_mae'][-1]:.6f}")
```
!!! tip "Cell 4: Fix it progressively"
    Change the configuration in Cell 2 and rerun Cells 2 and 3. Try fewer neurons, more neurons, shorter training, longer training, and different learning rates. Compare the Keras behavior to the pure NumPy version.
Cell 5: Add dropout and early stopping
```python
LEARNING_RATE = 0.001
EPOCHS = 5000
HIDDEN_DIM1 = 64
HIDDEN_DIM2 = 32
DROPOUT_RATE = 0.1

model2 = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    keras.layers.Dense(HIDDEN_DIM1, activation="relu"),
    keras.layers.Dropout(DROPOUT_RATE),
    keras.layers.Dense(HIDDEN_DIM2, activation="relu"),
    keras.layers.Dropout(DROPOUT_RATE),
    keras.layers.Dense(1),
])

model2.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="mse",
    metrics=["mae"],
)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=150,
    restore_best_weights=True,
)

history2 = model2.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0,
)

stopped_at = len(history2.history["loss"])
print(f"Stopped at epoch {stopped_at} / {EPOCHS}")
print(f"Final val loss: {history2.history['val_loss'][-1]:.6f}")
print(f"Final val MAE:  {history2.history['val_mae'][-1]:.6f}")

x_plot = np.linspace(0, 4 * np.pi, 1000).reshape(-1, 1)
y_true_plot = np.sin(x_plot)
y_pred_plot = model2.predict(x_plot, verbose=0)

plt.figure(figsize=(10, 5))
plt.plot(x_plot, y_true_plot, label="Actual sin(x)")
plt.plot(x_plot, y_pred_plot, label="Dropout + early stopping prediction", linestyle="--")
plt.axvline(2 * np.pi, color="gray", linestyle=":", label="End of training range")
plt.title("Regularized Keras Model")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.legend()
plt.show()
```

---

## Reading Training Curves


---

### Reading Training Curves
Use this as a diagnostic card during later projects.

| Symptom | Diagnosis | Fix |
|---|---|---|
| Loss stays flat from epoch 1 | LR too small or bad initialization | Increase LR or use Adam |
| Loss spikes or goes to NaN | LR too large | Reduce LR by 10x |
| Train loss falls, val loss rises | Overfitting | Add dropout, early stopping, or reduce network size |
| Train and val loss plateau high | Underfitting | More epochs, larger network, or better architecture |
| Val loss lower than train loss | Normal with dropout | Expected because dropout is active only during training |
| Train accuracy 100%, val accuracy low | Severe memorization | Much more data or a much smaller network |
| Both losses are noisy | Batch size too small or LR too large | Increase batch size to 32 or 64, or lower LR |

---

## When to Use Neural Networks


---

### When to Use Neural Networks

**Do you have more than 10,000 training examples?** No: try Random Forest first. If you must use a neural network, keep it small.

**Images, audio, or raw text?** Yes: neural network. CNN for images; RNN or Transformer for sequence data.

**Tabular rows and columns?** Try Random Forest or XGBoost first. Move to NN only if they underperform.

**Need to explain predictions?** Consider Decision Tree or Logistic Regression. Neural networks are powerful but harder to interpret.
!!! tip "Robotics perspective"
    YOLO26n is a CNN you train. MediaPipe is a neural network Google trained. The lane follower deliberately uses no neural network. Knowing when not to use one is part of the skill.


