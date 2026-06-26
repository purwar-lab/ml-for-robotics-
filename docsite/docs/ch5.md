# Chapter 5: Reinforcement Learning

---

## Reinforcement Learning: Teach a Robot to Navigate a Maze

*Beginner · 6 min*

---

**Reinforcement Learning** — Learning by taking actions, receiving rewards, and improving future decisions.

---

## 5.1 Agent, Environment, Reward

*Beginner · 8 min*

---

### 5.1 Concept: Agent, Environment, Reward
Agent action → Environment reward + state → Agent updates policy
RL is different from supervised learning because there is no answer key for every state. The agent must explore, make mistakes, and learn from reward signals.
Real robotics uses RL in legged locomotion, drone control, robotic surgery research, dexterous manipulation, and simulation-to-real training.

---

## 5.2 Q-Learning Explained with a Grid

*Intermediate · 12 min*

---

### 5.2 Q-Learning Explained with a Grid
A Q-table stores the expected value of taking each action in each state. In FrozenLake, there are 16 grid states and 4 actions, so the table is 16 x 4.
Q-learning update equation
```
Q(s, a) <- Q(s, a) + alpha [ r + gamma * max Q(s', a') - Q(s, a) ]
                 old value    rate   reward discount best future value
```
- **s**: current state.
- **a**: action chosen.
- **alpha**: learning rate, or how aggressively we update.
- **r**: immediate reward.
- **gamma**: discount factor for future rewards.
**Exploration vs exploitation** is the central tradeoff. Early on, the agent flips a biased coin and explores random moves often. Later, it mostly exploits the best actions it has learned.

---

## 5.3 Project: Teaching a Robot to Navigate a Maze

*Intermediate · 30 min*

---

### 5.3 ★ Project: Teaching a Robot to Navigate a Maze
**Environment:** OpenAI Gym / Gymnasium `FrozenLake-v1`. The notebook logs rewards to CSV, uploads them as a Kaggle dataset, and loads them back for analysis.
!!! tip "Download Dataset"
    This project generates `frozenlake_training_log.csv`. Upload or retrieve logs through [kaggle.com/datasets](https://www.kaggle.com/datasets).
Cell 1: Install gym, import libraries
```python
!pip -q install gymnasium[toy-text]
      import gymnasium as gym
      import numpy as np
      import pandas as pd
      import matplotlib.pyplot as plt
```
💡 What this does Installs the grid-world environment and imports the tools for tables, arrays, and charts. Expected output Installation completes with no import errors.
Cell 2: Create FrozenLake environment
```python
env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="ansi")
      state, info = env.reset(seed=42)
      print(env.render())
```
💡 What this does Creates a 4x4 map. Slippery ice adds randomness, which makes learning harder and more realistic. Expected output An ASCII grid with S, F, H, and G.
Cell 3: Action and observation space
```python
print("States:", env.observation_space.n)
      print("Actions:", env.action_space.n)
      print("0=Left, 1=Down, 2=Right, 3=Up")
```
💡 What this does Confirms that the state is a grid square and the action is a movement direction. Expected output States: 16 , Actions: 4 .
Cell 4: Initialize Q-table
```python
q_table = np.zeros((env.observation_space.n, env.action_space.n))
      print(q_table.shape)
```
💡 What this does The agent starts knowing nothing, so every action value starts at zero. Expected output (16, 4) .
Cell 5: Train with Q-learning
```python
episodes = 1000
      alpha = 0.8
      gamma = 0.95
      epsilon = 1.0
      epsilon_decay = 0.995
      min_epsilon = 0.05
      logs = []
      
      for episode in range(episodes):
          state, info = env.reset()
          done = False
          total_reward = 0
      
          while not done:
              if np.random.random() < epsilon:
                  action = env.action_space.sample()
              else:
                  action = np.argmax(q_table[state])
      
              next_state, reward, terminated, truncated, info = env.step(action)
              done = terminated or truncated
      
              best_next = np.max(q_table[next_state])
              q_table[state, action] += alpha * (reward + gamma * best_next - q_table[state, action])
      
              state = next_state
              total_reward += reward
      
          epsilon = max(min_epsilon, epsilon * epsilon_decay)
          logs.append({"episode": episode, "reward": total_reward, "epsilon": epsilon})
```
🤖 What this does This is the learning loop: choose action, observe reward, update Q-table, repeat. Expected output No output until the next plotting cell.
Cell 6: Plot reward curve
```python
log_df = pd.DataFrame(logs)
      log_df["rolling_reward"] = log_df["reward"].rolling(50).mean()
      
      plt.plot(log_df["episode"], log_df["rolling_reward"])
      plt.xlabel("Episode")
      plt.ylabel("Rolling average reward")
      plt.title("Q-learning progress")
      plt.show()
```
💡 What this does Rolling averages smooth noisy rewards so learning trends are easier to see. Expected output A curve that should rise as the policy improves.
Cell 7: Visualize learned policy
```python
arrows = np.array(["<", "v", ">", "^"])
      policy = arrows[np.argmax(q_table, axis=1)].reshape(4, 4)
      print(policy)
```
💡 What this does Converts best actions into arrows so you can inspect the learned navigation plan. Expected output A 4x4 grid of arrows.
Cell 8: Test trained agent
```python
state, info = env.reset(seed=7)
      done = False
      while not done:
          action = np.argmax(q_table[state])
          state, reward, terminated, truncated, info = env.step(action)
          done = terminated or truncated
          print(env.render())
      print("Final reward:", reward)
```
💡 What this does Runs one greedy episode using the learned policy instead of random exploration. Expected output Several grid frames and a final reward of 0 or 1.
Cell 9: Kaggle tie-in - save and upload log
```python
log_df.to_csv("frozenlake_training_log.csv", index=False)
      print("Saved frozenlake_training_log.csv")
      
      # Optional: create a Kaggle dataset manually from this CSV,
      # then use the Kaggle API to download it in a future notebook.
```
💡 What this does The CSV turns your RL experiment into a dataset you can share and re-analyze. Expected output A saved CSV file in Colab's file browser.
Cell 10: Challenge - larger map or non-slippery ice
```python
# Try one change at a time:
      env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True, render_mode="ansi")
      # or
      env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
```
📝 What this does The 8x8 map is harder. Turning slipperiness off isolates planning from randomness. Expected output A changed environment; rerun training and compare reward curves.
