# Chapter 2: What is Machine Learning?

---

## What is Machine Learning?

*Beginner · 8 min*

---

**Machine Learning** — A way to build programs that learn patterns from examples instead of following only hand-written rules.

**Instead of programming rules, you feed the robot examples and let it figure out the rules itself.** Show it 10,000 throws. It watches, finds patterns, and builds its own internal model of how to catch. That's machine learning: computers learning from data, not from explicit instructions.
!!! tip "The Robotics Angle"
    Almost every modern robot uses ML in some form: walking over rough terrain, recognizing tissue types, detecting pedestrians, sorting packages, or spotting defects on a factory line.
Projects You Will Build Later
Once you understand the concepts in this chapter and the chapters that follow, you will turn them into applied robotics notebooks like these.


⚙️
Robot Failure Predictor
Predict whether a machine is likely to fail from temperature, torque, speed, and wear readings.
**Stack:** Pandas, scikit-learn, Logistic Regression, Random Forest


📈
Sensor Cluster Analyzer
Group vibration windows into operating modes without labels, then interpret what each group means.
**Stack:** NumPy, FFT features, K-Means, PCA, DBSCAN


🚦
Traffic Sign Classifier
Train a convolutional neural network that recognizes real traffic signs for mobile robots.
**Stack:** OpenCV, TensorFlow/Keras, CNNs, GTSRB

---

## 2.1 Traditional Programming vs ML

*Beginner · 6 min*

---

### 2.1 Traditional Programming vs ML
**Traditional Programming**
**Input:** data + rules
**Output:** answers
You write the logic. The computer follows it exactly.
**Machine Learning**
**Input:** data + answers
**Output:** rules, also called a model
The computer learns the logic from examples.

---

## 2.2 Three Flavors of ML

*Beginner · 8 min*

---

### 2.2 Three Flavors of ML
**Supervised Learning**
Learn from labeled examples. Example: sensor readings labeled as failed or healthy.
**Unsupervised Learning**
Find hidden structure without labels. Example: group vibration patterns into operating modes.
**Reinforcement Learning**
Learn through actions and rewards. Example: a robot learns a navigation policy by trial and error.

---

## 2.3 When to Use Which

*Beginner · 6 min*

---

### 2.3 When to Use Which
Question Use Robotics Example Do you have labeled answers? Supervised learning Predict failure from known historical failures. Do you have raw data but no labels? Unsupervised learning Cluster sensor behavior into normal and unusual modes. Does the system learn by acting? Reinforcement learning Teach a robot to navigate a maze. Is the input an image or video? Computer vision Classify signs, detect parts, follow lanes.
