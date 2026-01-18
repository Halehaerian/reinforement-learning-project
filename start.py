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
    import gym
    import pygame
    from stable_baselines3 import PPO
    from simple_warehouse_env import SimpleWarehouseEnv
    
    # Check if model exists
    model_path = "warehouse_delivery_agent.zip"
    
    if not os.path.exists(model_path):
        print("Training new model (first time, ~90 seconds)...\n")
        env = SimpleWarehouseEnv(grid_size=5, render_mode=None)
        model = PPO('MlpPolicy', env, verbose=0, learning_rate=0.001, 
                   n_steps=1024, batch_size=64, n_epochs=20, gamma=0.99,
                   ent_coef=0.01)
        model.learn(total_timesteps=200000)
        model.save(model_path)
        env.close()
        print("✓ Model trained!\n")
    
    # Load model
    model = PPO.load(model_path)
    
    # Run test episodes
    print("Starting visual test (3 episodes)...")
    print("Watch the pygame window!\n")
    print("="*70 + "\n")
    
    env = SimpleWarehouseEnv(grid_size=5, render_mode='human')
    
    successes = 0
    for episode in range(3):
        print(f"Episode {episode + 1}/3: ", end="", flush=True)
        
        obs = env.reset()
        done = False
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            obs, reward, done, _ = env.step(action)
            steps += 1
        
        if steps < 50:
            print(f"✓ SUCCESS ({steps} steps)")
            successes += 1
        else:
            print(f"⏱ Timeout ({steps} steps)")
    
    env.close()
    
    print("\n" + "="*70)
    print(f"Result: {successes}/3 Episodes Successful".center(70))
    print("="*70 + "\n")
    print("✓ Done!\n")

except KeyboardInterrupt:
    print("\n\n⏹ Cancelled")
    sys.exit(0)

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
