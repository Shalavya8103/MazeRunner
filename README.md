# MazeRunner: Maze Navigation Using Classical Search and Reinforcement Learning

A two-agent adversarial maze game comparing classical AI search algorithms against reinforcement learning strategies. Built as the final project for **CS 5100: Foundations of Artificial Intelligence** at Northeastern University.

**Authors:** Shalavya Agrawal & Param Jain

## Overview

MazeRunner pits a **Runner** agent against a **Chaser** agent on a randomly generated grid-based maze. The Runner must collect three checkpoints scattered across the maze and reach the exit before the Chaser catches it. Both agents move each turn simultaneously, creating a dynamic adversarial planning problem.


## Project Structure

```
MazeRunner/
├── maze_env.py              # Maze generation (Prim's algorithm), game state, entity placement
├── renderer.py              # Step-by-step gameplay visualization
├── agents/
│   ├── astar.py             # A* search + enhanced A* with chaser danger-zone avoidance
│   ├── minimax.py           # Minimax chaser with alpha-beta pruning (depth=2)
│   ├── q_agent.py           # Single-agent Q-learning runner (training + inference)
│   └── marl.py              # Multi-agent RL: independent Q-learning for both runner & chaser
├── main_minimax.py          # Run A* Runner vs Minimax Chaser gameplay
├── main_qlearning.py        # Run Q-Learning Runner vs Minimax Chaser gameplay
├── main_marl.py             # Run MARL gameplay
├── q_tables/
│   ├── q_table.pkl          # Pre-trained Q-table for single-agent runner
│   ├── runner_marl.pkl      # Pre-trained MARL runner Q-table
│   └── chaser_marl.pkl      # Pre-trained MARL chaser Q-table
└── tests/
    ├── test_minimax.py      # Benchmark: A* Runner vs Minimax Chaser (1000 episodes)
    ├── test_qlearning.py    # Benchmark: Q-Learning Runner vs Minimax Chaser (1000 episodes)
    ├── test_marl.py         # Benchmark: MARL evaluation (1000 episodes)
    ├── results_minimax.png  # Results chart for A* vs Minimax
    ├── results_q.png        # Results chart for Q-Learning vs Minimax
    └── results_marl.png     # Results chart for MARL
```

---

## Setup

### Requirements

- Python 3.10+
- NumPy
- Matplotlib (for benchmark visualization)


## Usage

### Play / Visualize Games

```bash
# A* Runner vs Minimax Chaser
python main_minimax.py

# Q-Learning Runner vs Minimax Chaser
python main_qlearning.py

# MARL: Q-Learning Runner vs Q-Learning Chaser
python main_marl.py
```

### Run Benchmarks (1,000 episodes each)

```bash
cd tests

# A* Runner vs Minimax Chaser
python test_minimax.py

# Q-Learning Runner vs Minimax Chaser
python test_qlearning.py

# MARL evaluation
python test_marl.py
```

Each test script outputs win rate, average steps, timeout count, and average checkpoints collected, and saves a 4-panel results chart to the `tests/` directory.

### Train Q-Learning Agents from Scratch

```bash
# Train single-agent Q-learning runner (100,000 episodes, ~30 min)
python agents/q_agent.py

# Train MARL runner + chaser (100,000 episodes, ~45 min)
python agents/marl.py
```

Pre-trained Q-tables are included in `q_tables/` so you can skip training and go straight to evaluation.

co-training episodes with separate Q-tables and epsilon schedules. The Chaser receives +500 for catching, −500 for Runner escape, and distance-based shaping rewards.

