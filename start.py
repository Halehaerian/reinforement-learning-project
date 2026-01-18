#!/usr/bin/env python
"""
Warehouse Delivery RL - Agent learns to pick up and deliver items
Run: python start.py
"""

import os
import sys

print("\n" + "="*70)
print("  Warehouse Delivery RL Agent".center(70))
print("="*70 + "\n")

try:
    import numpy as np
    import pygame
    from stable_baselines3 import DQN
    from simple_warehouse_env import SimpleWarehouseEnv
    
    model_path = "warehouse_delivery_agent_dqn.zip"
    
    if not os.path.exists(model_path):
        print("Training DQN model for STRICT 2-Part Logic...")
        env = SimpleWarehouseEnv(grid_size=5, render_mode=None)
        # 50k steps is plenty for a 5x5 grid if exploration is efficient
        model = DQN('MlpPolicy', env, verbose=1, 
                    learning_rate=1e-3, 
                    exploration_fraction=0.5, # 25k steps of exploration
                    buffer_size=50000,
                    target_update_interval=500)
        model.learn(total_timesteps=50000)
        model.save(model_path)
        env.close()
        print("✓ Training complete.\n")
    
    model = DQN.load(model_path)
    env = SimpleWarehouseEnv(grid_size=5, render_mode='human')
    
    print("Testing One Full Mission...")
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = env.step(int(action))
        done = term or trunc
        pygame.time.wait(200)
    
    # Final pause to see the "PACKAGE DELIVERED" banner
    print("✓ Mission Complete. End of evaluation.")
    pygame.time.wait(3000)
    
    env.close()
    sys.exit(0)

except KeyboardInterrupt:
    print("\n\n⏹ Cancelled")
    sys.exit(0)

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
