# Simple Warehouse RL

A minimal reinforcement learning project using PPO to train an agent to navigate a grid.

## Quick Start

```bash
python start.py
```

## Project Structure

- **simple_warehouse_env.py** - The gym environment where the agent learns
- **start.py** - Training and testing script  
- **simple_warehouse_agent.zip** - Trained model (auto-created on first run)
- **requirements.txt** - Dependencies

## How It Works

1. **Agent** (red circle) learns to move to a **Goal** (green circle)
2. **Actions**: UP, DOWN, LEFT, RIGHT (4 discrete actions)
3. **Reward**: +100 for reaching goal, +1 for moving closer, -0.1 for moving away
4. **Training**: PPO algorithm, 50,000 timesteps

## To Run

```bash
# First time (trains for ~45 seconds)
python start.py

# Later runs (uses saved model)
python start.py
```

## Visual Output

- **Window shows**: 5x5 grid with agent and goal
- **Arrows**: Show direction of movement
- **Right panel**: Current step, action, reward, distance
- **Speed**: 2 FPS (slow to see each move)

## Files

Total: 4 files
- 1 environment file
- 1 training/testing script
- 1 model (auto-saved)
- 1 requirements file

Clean and minimal!
