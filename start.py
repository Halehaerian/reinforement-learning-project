#!/usr/bin/env python
"""
Academic Reinforcement Learning Framework
Project: Warehouse Delivery Optimization
Algorithm: Deep Q-Network (DQN)
"""

import os
import sys
import time
import argparse
import numpy as np
import pygame
from stable_baselines3 import DQN
from simple_warehouse_env import AdvancedRobotWarehouseEnv, Actions


class RLManager:
    """
    Manages the lifecycle of an RL agent including environment creation,
    training, and evaluation.
    """
    def __init__(self, model_path="warehouse_robot_v2.zip"):
        self.model_path = model_path
        self.env = None

    def create_env(self, render_mode=None):
        """Standardized environment factory."""
        return AdvancedRobotWarehouseEnv(render_mode=render_mode)

    def train(self, timesteps=300000):
        """
        Executes the training phase.
        Uses MlpPolicy with optimized hyperparameters for the robotics-first environment.
        """
        print(f"\n[TRAINING] Initializing {timesteps} timesteps on Robot OS v2.0...")
        self.env = self.create_env(render_mode=None)
        
        model = DQN(
            policy="MlpPolicy",
            env=self.env,
            verbose=1,
            learning_rate=3e-4,  # More precise learning
            buffer_size=200000,
            learning_starts=10000,
            batch_size=256, # Larger batch for better stability
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.3, # Faster transition to exploitation
            exploration_final_eps=0.01 # Near pure exploitation
        )
        
        start_time = time.time()
        model.learn(total_timesteps=timesteps)
        duration = time.time() - start_time
        
        model.save(self.model_path)
        self.env.close()
        print(f"[SUCCESS] Training completed in {duration:.2f}s. Model saved to {self.model_path}")

    def run_random_agent(self, episodes=1):
        """Baseline performance: Pure random exploration."""
        print(f"\n[RANDOM AGENT] Running {episodes} evaluation episode(s)...")
        env = self.create_env(render_mode='human')
        
        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action = env.action_space.sample()  # Pure random
                obs, reward, done, _, _ = env.step(action)
                total_reward += reward
                pygame.time.wait(100)
            
            print(f"Random Agent Episode {ep+1} Finish. Total Reward: {total_reward:.2f}")
            pygame.time.wait(1000)
            
        env.close()

    def run_trained_agent(self, episodes=1):
        """Trained performance: Deterministic policy execution."""
        if not os.path.exists(self.model_path):
            print(f"[ERROR] No model found at {self.model_path}. Please train first.")
            return

        print(f"\n[TRAINED AGENT] Loading model and running {episodes} episode(s)...")
        model = DQN.load(self.model_path)
        env = self.create_env(render_mode='human')
        
        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = env.step(int(action))
                total_reward += reward
                pygame.time.wait(200)
            
            print(f"Trained Agent Episode {ep+1} Finish. Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000)
            
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Warehouse RL Management Script")
    parser.add_argument("--mode", type=str, choices=["train", "test", "random", "all"], default="test",
                        help="Mode: train, test (trained), random, or all (train+test)")
    parser.add_argument("--steps", type=int, default=500000, help="Training timesteps")
    
    args = parser.parse_args()
    
    manager = RLManager()
    
    if args.mode == "train" or args.mode == "all":
        # Force retrain by removing old model if it exists
        if os.path.exists(manager.model_path):
            os.remove(manager.model_path)
        manager.train(timesteps=args.steps)
        
    if args.mode == "random":
        manager.run_random_agent(episodes=1)
        
    if args.mode == "test" or args.mode == "all":
        manager.run_trained_agent(episodes=1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOPPED] User interrupted execution.")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()
