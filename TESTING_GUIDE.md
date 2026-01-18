# Testing Guide - Warehouse Delivery RL

## Quick Test (One Command)

```powershell
cd "c:\HA-Projects\reinforement learning project"
.\.venv\Scripts\python.exe start.py
```

This runs **3 test episodes** automatically and shows results.

---

## Test Methods

### Method 1: Simple Test (Recommended)

```powershell
python start.py
```

**What happens:**
- Loads trained model
- Runs 3 visual episodes
- Shows success/failure for each
- Displays summary statistics

**Expected Output:**
```
Episode 1/3: ✓ SUCCESS (12 steps)
Episode 2/3: ✓ SUCCESS (15 steps)
Episode 3/3: ✓ SUCCESS (10 steps)

Result: 3/3 Episodes Successful
```

---

### Method 2: Custom Test Script

Create a file called `test_custom.py`:

```python
#!/usr/bin/env python
import numpy as np
from stable_baselines3 import PPO
from simple_warehouse_env import SimpleWarehouseEnv

# Load trained model
model = PPO.load("warehouse_delivery_agent.zip")

# Test settings
num_episodes = 5
render = True  # Set to False for faster testing

# Run tests
successes = 0
total_steps = 0

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    steps = 0
    episode_reward = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(int(action))
        episode_reward += reward
        steps += 1
    
    if steps < 200:
        successes += 1
        print(f"Episode {episode+1}: ✓ SUCCESS ({steps} steps, {episode_reward:.1f} reward)")
    else:
        print(f"Episode {episode+1}: ✗ TIMEOUT ({steps} steps, {episode_reward:.1f} reward)")
    
    total_steps += steps

print(f"\nSuccess Rate: {successes}/{num_episodes}")
print(f"Average Steps: {total_steps/num_episodes:.1f}")
```

Run it:
```powershell
python test_custom.py
```

---

### Method 3: Test Without Visualization (Fast)

```powershell
.\.venv\Scripts\python.exe -c "
from stable_baselines3 import PPO
from simple_warehouse_env import SimpleWarehouseEnv

model = PPO.load('warehouse_delivery_agent.zip')
env = SimpleWarehouseEnv(grid_size=5, render_mode=None)

for i in range(10):
    obs = env.reset()
    done = False
    steps = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(int(action))
        steps += 1
    print(f'Episode {i+1}: {steps} steps')
"
```

**Advantages:**
- ✓ Runs 10 episodes in ~5 seconds
- ✓ No pygame window (faster)
- ✓ Good for quick validation

---

## What to Look For

### Visual Test (Pygame Window)

✅ **Good Test Results:**
- Agent navigates to Blue "P" (pickup)
- Picks up item (Yellow box appears)
- Navigates to Green "D" (destination)
- Delivers item successfully
- Status shows "HOLDING" then "EMPTY"
- Rewards mostly positive

❌ **Bad Test Results:**
- Agent wanders aimlessly
- Doesn't reach pickup point
- Can't find destination
- Frequent timeouts
- Negative rewards

### Terminal Output

✅ **Success Example:**
```
Episode 1: ✓ SUCCESS (12 steps)  ← Finished in 12 steps
Episode 2: ✓ SUCCESS (15 steps)  ← Good!
Episode 3: ✓ SUCCESS (10 steps)  ← Very efficient
Result: 3/3 Episodes Successful  ← 100% success rate
```

❌ **Failure Example:**
```
Episode 1: ✗ TIMEOUT (200 steps)  ← Hit step limit
Episode 2: ✗ TIMEOUT (200 steps)  ← Couldn't complete
Episode 3: ✓ SUCCESS (18 steps)   ← Sometimes works
Result: 1/3 Episodes Successful   ← Only 33% success
```

---

## Expected Performance

### After Training

**Typical Results:**
- ✓ Success Rate: 80-100%
- ✓ Average Steps: 10-20 per episode
- ✓ Delivery Reward: +300 (success)
- ✓ Episode Reward: +50 to +200

**What Agent Learns:**
```
Step 1-3:   Explores, finds Blue P
Step 4-5:   Picks up item
Step 6-8:   Navigates to Green D
Step 9-10:  Delivers item
Total:      10-15 steps average
Reward:     +300 + movement bonuses
```

---

## Test Checklist

Before claiming success:

- [ ] Run 5+ episodes
- [ ] Check success rate (should be 80%+)
- [ ] Verify agent picks up item (yellow box visible)
- [ ] Confirm agent reaches destination
- [ ] Check step count (should be 10-20 range)
- [ ] See positive rewards in terminal

---

## Troubleshooting

### Issue: Still Timing Out

```powershell
# Check if model exists
Get-ChildItem warehouse_delivery_agent.zip

# If not, retrain
Remove-Item warehouse_delivery_agent.zip
python start.py  # Retrains from scratch
```

### Issue: Low Success Rate (<50%)

Model needs more training:
```python
# In start.py, increase training:
model.learn(total_timesteps=300000)  # was 200000
```

### Issue: Pygame Window Doesn't Open

Run without visualization first:
```powershell
python test_custom.py  # with render=False
```

---

## Complete Test Workflow

```powershell
# 1. Train (if needed)
python start.py

# 2. Run more tests
python test_custom.py

# 3. Check git status
git status

# 4. Commit if satisfied
git add .
git commit -m "Testing passed: agent successfully learns pickup+delivery"
```

---

## Performance Metrics

After successful training, you should see:

| Metric | Value |
|--------|-------|
| Success Rate | 80-100% |
| Avg Steps/Episode | 10-20 |
| Min Steps | 6-8 (optimal) |
| Max Steps | 200 (timeout) |
| Pickup Success | 100% |
| Delivery Success | 80%+ |

---

## How to Verify Success

✅ **Agent Successfully Learned When:**
1. It reaches pickup point (Blue P)
2. It picks up the item (Yellow box)
3. It finds destination (Green D)
4. It delivers the item
5. 80%+ success rate
6. Completes in 10-20 steps

**Run this now:**
```powershell
python start.py
```

Watch the pygame window to see your trained agent work! 🎯
