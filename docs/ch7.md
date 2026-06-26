# Chapter 7: Computer Vision for Robotics

---

**Computer Vision** — Techniques that let computers interpret images and video as useful sensor data.

---

## Why Vision Matters in Robotics


---

### Why Vision Matters in Robotics
Cameras are cheap, rich sensors. Robots use vision to read signs, detect obstacles, estimate pose, inspect parts, track lanes, and understand human workspaces.

---

## Tools: OpenCV


---

### Tools: OpenCV
OpenCV is the Open Source Computer Vision Library, the general-purpose toolkit for image processing. An image is a NumPy array: height x width x color channels.
OpenCV quick demo
```python
import cv2
import matplotlib.pyplot as plt

img_bgr = cv2.imread("sample_data/traffic_sign.jpg")
print(img_bgr.shape)             # height, width, channels

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Convert BGR to RGB for correct color display
small = cv2.resize(img_rgb, (32, 32)) # Resize to 32x32 for faster processing
edges = cv2.Canny(small, threshold1=50, threshold2=150) # Canny edge detection; less than 50 = black, more than 150 = white, in between = gray

# Display both the original image with rectangle and the edges
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(img_rgb)
axes[0].set_title('Original Image with Rectangle')
axes[0].axis('off')

axes[1].imshow(edges, cmap='gray')
axes[1].set_title('Detected Edges (Resized)')
axes[1].axis('off')

plt.show()
```
!!! warning "BGR vs RGB"
    OpenCV loads images as BGR. Matplotlib expects RGB. Convert before displaying or colors look wrong.

---

## Tools: TensorFlow/Keras


---

### Tools: TensorFlow/Keras
TensorFlow is Google's library for building and training neural networks. Keras is the friendly wrapper on top, like power steering on a car. You will use tensors, layers, models, epochs, batches, loss functions, and optimizers.
Input image → Layers stacked in order → Class probabilities

---

## How CNNs See


---

### How Convolutional Neural Networks See
A normal neural network sees a flat list of pixels. A CNN scans small filters across the image, so it can learn edges, corners, textures, shapes, and eventually objects.
Input 32x32 image → Conv2D + MaxPool → Dense → 43 signs

---

## Project: Traffic Sign Classifier


---

###  Project: Traffic Sign Classifier with TensorFlow + OpenCV
**Dataset:** [German Traffic Sign Recognition Benchmark (GTSRB)](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign). It has 43 classes and about 50,000 real-world sign images.
!!! tip "Download Dataset"
    Direct Kaggle URL: [gtsrb-german-traffic-sign](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).
[Open in Colab →](https://colab.research.google.com/github/purwar-lab/ml-for-robotics-/blob/main/notebooks/ch7-traffic-sign-classifier.ipynb)
Cell 1: Kaggle download of GTSRB
```python
!pip -q install kaggle
import json
import os
from getpass import getpass

os.makedirs("/root/.kaggle", exist_ok=True)
token = json.loads(getpass("Paste your Kaggle API token text: ").strip())
with open("/root/.kaggle/kaggle.json", "w") as f:
    json.dump(token, f)
os.chmod("/root/.kaggle/kaggle.json", 0o600)
!kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
!unzip -q -o gtsrb-german-traffic-sign.zip -d gtsrb
```
💡 What this does Downloads and extracts the traffic sign dataset into Colab. Expected output Folders containing sign images and labels.
Cell 2: Explore dataset
```python
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

train_csv = Path("gtsrb/Train.csv")
df = pd.read_csv(train_csv)
display(df.head())
df["ClassId"].value_counts().sort_index().plot(kind="bar", figsize=(14, 4))
plt.title("Images per traffic sign class")
plt.show()
```
💡 What this does Shows metadata and checks whether classes are balanced. Expected output A table and a 43-bar class count chart.
Cell 3: OpenCV preprocessing pipeline
```python
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

images, labels = [], []
for _, row in df.iterrows():
    path = Path("gtsrb") / row["Path"]
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32))
    img = img.astype("float32") / 255.0
    images.append(img)
    labels.append(row["ClassId"])

X = np.array(images)
y = np.array(labels)
print(X.shape, y.shape)
```
Cell 3b: Data augmentation setup
```python
augmenter = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.10,
    horizontal_flip=True
)

# Use with care: horizontal flips can change the meaning of some traffic signs.
```
💡 What this does Loads images, fixes color order, resizes to 32x32, normalizes pixels to 0-1, and defines simple augmentation for more varied training examples. Expected output (num_images, 32, 32, 3) and labels.
Cell 4: Split into train/validation/test
```python
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
```
💡 What this does Creates separate data for training, tuning during training, and final evaluation. Expected output No output; three splits are ready.
Cell 5: Build CNN architecture
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(43, activation="softmax")
])
model.summary()
```
🤖 What this does Conv layers learn visual features, pooling shrinks the image, dense layers make the final class decision. Expected output A model summary with trainable parameters.
Cell 6: Compile model
```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```
💡 What this does adam updates weights, loss measures wrongness, and accuracy reports percent correct. Expected output No output; model is ready to train.
Cell 7: Train for 15 epochs
```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=64
)
```
💡 What this does Each epoch passes through the training set once and reports train/validation accuracy. Expected output A live progress bar for each epoch.
Cell 8: Plot accuracy and loss curves
```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["accuracy"], label="train")
axes[0].plot(history.history["val_accuracy"], label="val")
axes[0].set_title("Accuracy")
axes[0].legend()

axes[1].plot(history.history["loss"], label="train")
axes[1].plot(history.history["val_loss"], label="val")
axes[1].set_title("Loss")
axes[1].legend()
plt.show()
```
⚠️ What this does If training accuracy rises but validation accuracy stalls, the model may be overfitting. Expected output Two learning curves.
Cell 9: Evaluate on test set
```python
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.3f}")
```
💡 What this does The test set is used once at the end to estimate real-world performance. Expected output A single test accuracy value.
Cell 10: Confusion matrix and top mistakes
```python
from sklearn.metrics import confusion_matrix

pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(9, 9))
plt.imshow(cm, cmap="Blues")
plt.title("43-class confusion matrix")
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.colorbar()
plt.show()

mistakes = []
for true_id in range(43):
    for pred_id in range(43):
        if true_id != pred_id and cm[true_id, pred_id] > 0:
            mistakes.append((cm[true_id, pred_id], true_id, pred_id))
print(sorted(mistakes, reverse=True)[:10])
```
💡 What this does The full matrix shows all class-to-class errors, and the sorted list highlights the 10 most common confusions. Expected output A 43x43 plot and top 10 confusion tuples.
Cell 11: Test one image
```python
idx = 0
probs = model.predict(X_test[idx:idx+1])[0]
pred_label = np.argmax(probs)
confidence = probs[pred_label]

plt.imshow(X_test[idx])
plt.title(f"Predicted {pred_label}, true {y_test[idx]}, confidence {confidence:.1%}")
plt.axis("off")
plt.show()
```
💡 What this does Shows a concrete example instead of only aggregate metrics. Expected output A sign image with predicted label, true label, and confidence.
Cell 12: OpenCV raw-image prediction overlay
```python
sample_path = Path("gtsrb") / df.iloc[0]["Path"]
raw_bgr = cv2.imread(str(sample_path))
raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

model_input = cv2.resize(raw_rgb, (32, 32)).astype("float32") / 255.0
probs = model.predict(model_input[None, ...])[0]
pred_label = np.argmax(probs)
confidence = probs[pred_label]

overlay = raw_rgb.copy()
cv2.putText(
    overlay,
    f"Class {pred_label}: {confidence:.1%}",
    (5, 25),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2
)

plt.imshow(overlay)
plt.axis("off")
plt.show()
```
🤖 What this does This loads a raw sign image with OpenCV, preprocesses it exactly like training images, feeds it to the CNN, and draws the prediction on top. Expected output A traffic sign image with prediction text overlaid.
Cell 13: Challenge - add a third Conv2D block
```python
# Add before Flatten:
# layers.Conv2D(128, (3, 3), activation="relu"),
# layers.MaxPooling2D((2, 2)),
#
# Retrain and compare validation accuracy and training time.
```
📝 What this does A deeper network may learn richer features, but it can also overfit or train slower. Expected output Your comparison notes.
