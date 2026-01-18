"""
Test the environment manually to verify it works
"""
from simple_warehouse_env import SimpleWarehouseEnv
import numpy as np

env = SimpleWarehouseEnv(grid_size=5, render_mode='human')

# Test 1: Can we manually guide the agent to success?
print("Test 1: Manual guidance test")
print("="*50)

obs = env.reset()
print(f"Initial state: agent={obs[:2]}, pickup={obs[2:4]}, dest={obs[4:6]}, holding={obs[6]}")

# Move towards pickup
for i in range(20):
    agent_pos = obs[:2]
    pickup_pos = obs[2:4]
    
    # Simple pathfinding to pickup
    if agent_pos[0] < pickup_pos[0]:
        action = 3  # right
    elif agent_pos[0] > pickup_pos[0]:
        action = 2  # left
    elif agent_pos[1] < pickup_pos[1]:
        action = 1  # down
    elif agent_pos[1] > pickup_pos[1]:
        action = 0  # up
    else:
        action = 4  # pickup
    
    obs, reward, done, _ = env.step(action)
    dist_to_pickup = np.linalg.norm(obs[:2] - obs[2:4])
    print(f"Step {i+1}: action={action} dist_to_pickup={dist_to_pickup:.2f} holding={obs[6]} reward={reward:.1f}")
    
    if obs[6] > 0.5:  # holding item
        print(f"✓ PICKED UP!")
        break
    if i == 19:
        print(f"✗ Failed to pick up in 20 steps")

print()

# Now move towards destination
if obs[6] > 0.5:
    print("Moving to destination...")
    for i in range(30):
        agent_pos = obs[:2]
        dest_pos = obs[4:6]
        
        # Simple pathfinding to destination
        if agent_pos[0] < dest_pos[0]:
            action = 3  # right
        elif agent_pos[0] > dest_pos[0]:
            action = 2  # left
        elif agent_pos[1] < dest_pos[1]:
            action = 1  # down
        elif agent_pos[1] > dest_pos[1]:
            action = 0  # up
        else:
            action = 5  # drop
        
        obs, reward, done, _ = env.step(action)
        dist_to_dest = np.linalg.norm(obs[:2] - obs[4:6])
        print(f"Step {i+1}: action={action} dist_to_dest={dist_to_dest:.2f} holding={obs[6]} reward={reward:.1f}")
        
        if done:
            print(f"✓ DELIVERY COMPLETE! Total reward from drop: {reward}")
            break
        if i == 29:
            print(f"✗ Failed to deliver in 30 steps")

env.close()
