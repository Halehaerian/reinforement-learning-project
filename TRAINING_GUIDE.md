# Training and Running Guide - Pickup & Delivery Task

## Quick Start (One Command)

```powershell
cd "c:\HA-Projects\reinforement learning project"
.\.venv\Scripts\python.exe start.py
```

This does EVERYTHING:
1. ✓ Trains model (first time only, ~60 seconds)
2. ✓ Loads trained model
3. ✓ Runs 3 visual episodes

---

## Step-by-Step Workflow

### Step 1: Train the Model (First Time Only)

```powershell
python start.py
```

**What happens:**
- Checks if `warehouse_delivery_agent.zip` exists
- If NOT → Trains for 100,000 timesteps (~60 seconds)
- Saves the model
- Then runs 3 test episodes

**Terminal Output:**
```
Training new model (first time, ~60 seconds)...

(training in progress...)

✓ Model trained!

Starting visual test (3 episodes)...
Watch the pygame window!
```

### Step 2: Watch the Agent Learn

**Visual Window Opens:**
- 5×5 grid appears
- RED circle = Agent
- BLUE "P" = Pickup point
- GREEN "D" = Destination

**What You See:**
```
Episode 1: ✓ SUCCESS (12 steps)
Episode 2: ✓ SUCCESS (15 steps)
Episode 3: ✓ SUCCESS (10 steps)

Result: 3/3 Episodes Successful
```

### Step 3: Run Again (Uses Trained Model)

```powershell
python start.py
```

**This time:**
- Model already exists
- Skips training (instant!)
- Goes straight to 3 test episodes
- Shows improved performance

---

## What Agent Learns to Do

### Episode Flow:

```
1. EXPLORE
   Agent wanders around the 5×5 grid
   Looking for the pickup point (Blue P)

2. PICKUP
   Agent reaches Blue P
   Action: PICKUP (action 4)
   Status changes to "HOLDING"
   Yellow box appears around agent
   Reward: +25

3. NAVIGATE TO DESTINATION
   Agent moves towards Green D
   Holds the item while moving
   Yellow box visible

4. DELIVER
   Agent reaches Green D
   Action: DROP (action 5)
   Item delivered!
   Status changes to "EMPTY"
   Reward: +200
   Episode SUCCESS!
```

---

## Visual Indicators

### Color Meanings:

- **RED circle** = Agent position
- **BLUE "P"** = Pick up point (go here first)
- **GREEN "D"** = Destination (deliver here)
- **YELLOW box** = Agent is holding item

### Right Panel Shows:

```
Step Info
─────────────────────
Step: 5
Action: RIGHT →
Status: HOLDING
Reward: +0.50
```

### Movement Arrows:

- **Blue arrow** = Moving UP
- **Orange arrow** = Moving DOWN
- **Pink arrow** = Moving LEFT
- **Cyan arrow** = Moving RIGHT

---

## Training Timeline

### First Run (Training + Testing):
```
Start: python start.py
  ↓
Train model: 60 seconds
  ↓
Load model: instant
  ↓
Run 3 episodes: 30 seconds (2 FPS)
  ↓
Show results
  ↓
Done! (~2 minutes total)
```

### Subsequent Runs (Testing Only):
```
Start: python start.py
  ↓
Load model: instant
  ↓
Run 3 episodes: 30 seconds
  ↓
Show results
  ↓
Done! (~30 seconds)
```

---

## Reward System

Agent gets rewards for:

```
+200  ← Successful delivery (main goal!)
 +25  ← Picking up item
+0.5  ← Moving towards current target
-0.01 ← Each step (encourages efficiency)
 -5   ← Dropping in wrong location
```

---

## Complete Training Command

```powershell
cd "c:\HA-Projects\reinforement learning project"
.\.venv\Scripts\python.exe start.py
```

**That's it!** 

The script handles:
- Model training
- Model loading
- Visual testing
- Results display

All automatic! 🎯

---

## To Train Again (Fresh Model)

```powershell
# Delete old model
Remove-Item warehouse_delivery_agent.zip

# Train new model
python start.py
```

---

## Expected Results

After training, expect:
- 80-100% success rate
- Average 10-15 steps per episode
- Quick pickup and delivery
- Agent learns to:
  1. Find pickup point
  2. Pick up item
  3. Find destination
  4. Deliver item
