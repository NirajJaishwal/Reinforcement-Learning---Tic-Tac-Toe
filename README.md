# Reinforcement Learning-Based AI Player for Tic-Tac-Toe  

**Team Members:** Ardacan Yildiz, Niraj Jaishwal, Nuwan Janaranga  
**Course Project:** Human Computer Interaction  

---

## 📌 Introduction  

This project implements an **AI player for Tic-Tac-Toe** using **Reinforcement Learning (Q-Learning)**.  
The agent learns strategies through **self-play** rather than pre-programmed rules, improving its decision-making over time.  

The goal was to demonstrate how **reinforcement learning** enables autonomous strategy optimization in game environments. Tic-Tac-Toe, though simple, provides an ideal platform for testing RL concepts such as exploration, exploitation, and adversarial play.  

---

## ⚙️ Core Features  

- **Q-Learning Agent:** Learns optimal strategies via trial-and-error.  
- **Exploration vs Exploitation:** Implements ε-greedy strategy.  
- **Opponents Tested:**  
  - Random Player (baseline with no strategy)  
  - Minimax Player (optimal strategy benchmark)  
  - Self-Play (agent vs agent)  

---

## 📊 Results Summary  

- **Vs Random Player:**  
  - As First Player: **90.5% win rate**  
  - As Second Player: **86.3% win rate**  

- **Vs Minimax Player:**  
  - 0% win rate, **80.8% draw rate** (agent avoids losing most games)  

- **Vs Itself:**  
  - Balanced outcomes (≈29% win, 19% loss, 52% draw)  

---

## 🚀 Future Work  

- Extend to **Deep Q-Learning (DQN)** for handling larger state spaces.  
- Explore **Double Q-Learning** and **Dueling Networks** for more stable training.  
- Apply the framework to **more complex games** beyond Tic-Tac-Toe.  

---

## ▶️ Getting Started  

### Prerequisites  
- Python 3.8+  
- Libraries: `numpy`, `matplotlib`  

### Run Training & Play  
```bash
git clone https://github.com/your-username/rl-tictactoe.git
cd rl-tictactoe
python main.py   # Train the Q-learning agent
python minimax.py    # Play against the trained agent
