# Warehouse Delivery RL

A reinforcement learning project where an agent learns to pick up items and deliver them to destinations.

## Quick Start

```bash
python start.py
```

## Project Structure

- **simple_warehouse_env.py** - The gym environment with pickup and delivery tasks
- **start.py** - Training and testing script  
- **warehouse_delivery_agent.zip** - Trained model (auto-created on first run)
- **requirements.txt** - Dependencies

## How It Works

**Agent Task:**
1. Navigate to PICKUP point (Blue "P")
2. Pick up the item
3. Navigate to DESTINATION (Green "D")
4. Deliver the item

**Actions:** 6 discrete actions
- 0: UP ↑
- 1: DOWN ↓
- 2: LEFT ←
- 3: RIGHT →
- 4: PICKUP (when at pickup point)
- 5: DROP (when at destination)

**Reward System:**
- +200 for successful delivery
- +25 for picking up item
- +0.5 for moving towards current target
- -0.01 per step (encourages efficiency)
- -5 for dropping in wrong location

**Training:** PPO algorithm, 100,000 timesteps

## To Run

```bash
# First time (trains for ~60 seconds)
python start.py

# Later runs (uses saved model)
python start.py
```

## Visual Output

- **Pygame Window:**
  - 5×5 grid
  - RED circle = Agent
  - BLUE "P" = Pickup point
  - GREEN "D" = Destination
  - YELLOW box = Agent carrying item
  - Colored arrows show movement direction

- **Right Panel Info:**
  - Current step number
  - Current action
  - Status (HOLDING/EMPTY)
  - Reward for last action

## Files

4 essential files:
- 1 environment with pickup/delivery
- 1 training/testing script
- 1 requirements file
- 1 model (auto-saved)

Clean and minimal!
