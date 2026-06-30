# Chapter 0: Welcome & How to Use This Guide

---

## Welcome & How to Use This Guide

### Machine Learning for Robotics Engineers

**Mechanical Engineering x AI**

This page is a long-scroll textbook, lab manual, and project checklist. You will learn the programming basics, the core ML families, and how to run full notebooks in Google Colab without installing anything locally.

#### Learning Journey

| Step | Topic | Description |
|:--|:--|:--|
| 0 | **Setup** | Browser tools, Colab later, how to read this guide |
| 1 | **Python** | Variables, loops, functions, libraries |
| 2 | **ML Concepts** | Supervised, unsupervised, reinforcement |
| 3 | **Applied ML** | Training, evaluating, and interpreting models |
| 4 | **Vision** | OpenCV, CNNs, and image data |

---

## What You Will Build

This course builds the foundation you need to understand and use machine learning in robotics, even if you are starting with no programming background.

| Skill | Description |
|:--|:--|
| **Python Confidence** | Write and read basic Python, use notebooks, import libraries, and understand the code you run. |
| **Data Thinking** | Look at tables, sensor readings, labels, plots, and train/test splits the way an engineer would. |
| **ML Intuition** | Know when a problem is supervised, unsupervised, reinforcement learning, or computer vision. |
| **Robotics Context** | Connect models to mechanical systems, sensors, failures, vibration patterns, navigation, and images. |

Later chapters turn those skills into applied notebook projects. Chapter 0 is only here to orient you: what tools you will use, how the course is organized, and how to read it without getting lost.

---

## Tools You Need

### Early Chapters: Zero Installation

For the Python fundamentals chapters you can run code directly in the browser on this page. For the machine learning and computer vision projects in the middle chapters you will use Google Colab, which gives you a free Python environment with no installation required.

**Getting started with Colab:**

1. **Open Colab.** Go to [colab.research.google.com](https://colab.research.google.com/), sign in, and choose **New Notebook**.
2. **Create your first cell.** Click the gray code cell and type `print("Hello, Robot!")`.
3. **Run a cell.** Press `Shift+Enter`. The output appears directly underneath the cell.

```
Open Colab → Write code cell → Shift+Enter runs it
```

### Later Projects: Python & VS Code

The hardware projects — where you deploy code to a real robot — require Python installed on your machine and a code editor. We use **Visual Studio Code** for these stages.

!!! tip "Don't install anything yet"
    You do not need Python or VS Code right now. Each project chapter that requires them opens with a step-by-step installation tutorial. Follow that guide when you reach it.

1. **Python 3.** A local Python installation lets you run scripts directly on your computer and communicate with hardware over USB or serial. The project chapters walk you through installing it when the time comes.
2. **Visual Studio Code.** VS Code is a free code editor with excellent Python support. The project chapters include a full installation and setup tutorial — no prior experience needed.

Later in the course you will also create a free Kaggle account to access datasets — that setup is explained at the start of the relevant chapter.

---

## Reading Tips

- Read top to bottom the first time. The early Python ideas are used later without re-explaining them.
- Use the sidebar to jump back to specific topics when a notebook cell feels unfamiliar.
- Every Project section has a full Colab-style notebook on the page. Do not skip the projects.
