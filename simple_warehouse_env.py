import gym
from gym import spaces
import numpy as np
import pygame


class SimpleWarehouseEnv(gym.Env):
    """
    Simple warehouse environment.
    Agent learns to navigate to a random goal location.
    
    Actions: 0=up, 1=down, 2=left, 3=right
    Observation: [agent_x, agent_y, goal_x, goal_y]
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, grid_size=5, render_mode=None):
        super(SimpleWarehouseEnv, self).__init__()
        
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.cell_size = 60
        
        # Actions: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)
        self.action_names = ["UP ↑", "DOWN ↓", "LEFT ←", "RIGHT →"]
        
        # Observation: [agent_x, agent_y, goal_x, goal_y]
        self.observation_space = spaces.Box(
            low=0, high=grid_size,
            shape=(4,), dtype=np.float32
        )
        
        self.screen = None
        self.clock = None
        self.step_count = 0
        self.last_action = None
        self.last_reward = 0
        
        self.reset()
    
    def reset(self):
        """Reset environment with new random positions"""
        self.agent_pos = np.array([
            np.random.randint(0, self.grid_size),
            np.random.randint(0, self.grid_size)
        ], dtype=np.float32)
        
        while True:
            self.goal_pos = np.array([
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size)
            ], dtype=np.float32)
            if not np.array_equal(self.agent_pos, self.goal_pos):
                break
        
        self.step_count = 0
        return self._get_observation()
    
    def _get_observation(self):
        """Return current state"""
        return np.array([
            self.agent_pos[0],
            self.agent_pos[1],
            self.goal_pos[0],
            self.goal_pos[1]
        ], dtype=np.float32)
    
    def step(self, action):
        """Execute one action"""
        self.step_count += 1
        self.last_action = int(action)
        old_distance = np.linalg.norm(self.agent_pos - self.goal_pos)
        
        # Move agent
        if action == 0:  # up
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1:  # down
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
        elif action == 2:  # left
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3:  # right
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        
        # Check if reached goal
        distance = np.linalg.norm(self.agent_pos - self.goal_pos)
        done = distance < 0.5
        
        # Reward: +100 for reaching goal, +1 for getting closer, -0.1 otherwise
        if done:
            reward = 100
        elif distance < old_distance:
            reward = 1
        else:
            reward = -0.1
        
        reward -= 0.01  # Time penalty
        self.last_reward = reward
        
        # Timeout after 50 steps
        if self.step_count > 50:
            done = True
        
        if self.render_mode == 'human':
            self.render()
        
        return self._get_observation(), reward, done, {}
    
    def render(self):
        """Visualize environment"""
        if self.render_mode != 'human':
            return
        
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.grid_size * self.cell_size + 200, self.grid_size * self.cell_size + 50)
            )
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Simple Warehouse RL")
            self.font_large = pygame.font.Font(None, 32)
            self.font = pygame.font.Font(None, 24)
            self.font_small = pygame.font.Font(None, 20)
        
        # Clear
        self.screen.fill((240, 240, 240))
        
        # Draw grid
        grid_px = self.grid_size * self.cell_size
        pygame.draw.rect(self.screen, (255, 255, 255), (0, 0, grid_px, grid_px))
        
        for i in range(self.grid_size + 1):
            pygame.draw.line(self.screen, (200, 200, 200),
                           (i * self.cell_size, 0),
                           (i * self.cell_size, grid_px), 1)
            pygame.draw.line(self.screen, (200, 200, 200),
                           (0, i * self.cell_size),
                           (grid_px, i * self.cell_size), 1)
        
        # Draw goal
        goal_x = int(self.goal_pos[0] * self.cell_size + self.cell_size // 2)
        goal_y = int(self.goal_pos[1] * self.cell_size + self.cell_size // 2)
        pygame.draw.circle(self.screen, (50, 200, 50), (goal_x, goal_y), 20)
        pygame.draw.circle(self.screen, (100, 255, 100), (goal_x, goal_y), 15)
        self.screen.blit(self.font.render("G", True, (255, 255, 255)), (goal_x - 8, goal_y - 8))
        
        # Draw agent
        agent_x = int(self.agent_pos[0] * self.cell_size + self.cell_size // 2)
        agent_y = int(self.agent_pos[1] * self.cell_size + self.cell_size // 2)
        pygame.draw.circle(self.screen, (200, 50, 50), (agent_x, agent_y), 20)
        pygame.draw.circle(self.screen, (255, 100, 100), (agent_x, agent_y), 15)
        self.screen.blit(self.font.render("A", True, (255, 255, 255)), (agent_x - 8, agent_y - 8))
        
        # Draw movement arrow
        if self.last_action is not None:
            colors = {0: (0, 100, 255), 1: (255, 150, 0), 2: (255, 0, 150), 3: (0, 255, 150)}
            dx = [0, 0, -25, 25]
            dy = [-25, 25, 0, 0]
            
            arrow_x = agent_x + dx[self.last_action]
            arrow_y = agent_y + dy[self.last_action]
            pygame.draw.line(self.screen, colors[self.last_action], 
                           (agent_x, agent_y), (arrow_x, arrow_y), 4)
            pygame.draw.circle(self.screen, colors[self.last_action], (arrow_x, arrow_y), 8)
        
        # Draw info panel
        info_x = grid_px + 20
        info_y = 20
        
        self.screen.blit(self.font_large.render("Step Info", True, (50, 50, 50)), (info_x, info_y))
        self.screen.blit(self.font.render(f"Step: {self.step_count}", True, (50, 50, 50)), 
                        (info_x, info_y + 40))
        
        if self.last_action is not None:
            self.screen.blit(self.font.render(f"Action: {self.action_names[self.last_action]}", 
                                             True, (0, 0, 200)), (info_x, info_y + 70))
        
        reward_color = (0, 150, 0) if self.last_reward > 0 else (150, 0, 0)
        self.screen.blit(self.font.render(f"Reward: {self.last_reward:+.2f}", True, reward_color),
                        (info_x, info_y + 100))
        
        dist = np.linalg.norm(self.agent_pos - self.goal_pos)
        self.screen.blit(self.font_small.render(f"Distance: {dist:.1f}", True, (100, 100, 0)),
                        (info_x, info_y + 140))
        
        pygame.display.flip()
        self.clock.tick(2)
    
    def close(self):
        if self.screen is not None:
            pygame.quit()
