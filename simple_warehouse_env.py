import gym
from gym import spaces
import numpy as np
import pygame


class SimpleWarehouseEnv(gym.Env):
    """
    Warehouse environment with pickup and delivery.
    Agent: Pick up item at PICKUP point, deliver to DESTINATION.
    
    Actions: 0=up, 1=down, 2=left, 3=right, 4=pickup, 5=drop
    Observation: [agent_x, agent_y, pickup_x, pickup_y, dest_x, dest_y, holding_item]
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, grid_size=5, render_mode=None):
        super(SimpleWarehouseEnv, self).__init__()
        
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.cell_size = 60
        
        # Actions: 0=up, 1=down, 2=left, 3=right, 4=pickup, 5=drop
        self.action_space = spaces.Discrete(6)
        self.action_names = ["UP ↑", "DOWN ↓", "LEFT ←", "RIGHT →", "PICK", "DROP"]
        
        # Observation: [agent_x, agent_y, pickup_x, pickup_y, dest_x, dest_y, holding]
        self.observation_space = spaces.Box(
            low=0, high=grid_size,
            shape=(7,), dtype=np.float32
        )
        
        self.screen = None
        self.clock = None
        self.step_count = 0
        self.last_action = None
        self.last_reward = 0
        self.holding_item = False
        
        self.reset()
    
    def reset(self):
        """Reset environment with new random positions"""
        # Agent starts at random position
        self.agent_pos = np.array([
            np.random.randint(0, self.grid_size),
            np.random.randint(0, self.grid_size)
        ], dtype=np.float32)
        
        # Pickup point (random)
        self.pickup_pos = np.array([
            np.random.randint(0, self.grid_size),
            np.random.randint(0, self.grid_size)
        ], dtype=np.float32)
        
        # Destination (random, different from pickup)
        while True:
            self.dest_pos = np.array([
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size)
            ], dtype=np.float32)
            if not np.array_equal(self.dest_pos, self.pickup_pos):
                break
        
        self.holding_item = False
        self.step_count = 0
        return self._get_observation()
    
    def _get_observation(self):
        """Return current state"""
        return np.array([
            self.agent_pos[0],
            self.agent_pos[1],
            self.pickup_pos[0],
            self.pickup_pos[1],
            self.dest_pos[0],
            self.dest_pos[1],
            float(self.holding_item)
        ], dtype=np.float32)
    
    def step(self, action):
        """Execute one action"""
        self.step_count += 1
        self.last_action = int(action)
        reward = -0.01  # Small penalty per step
        
        # Movement actions (0-3)
        if action == 0:  # up
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1:  # down
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
        elif action == 2:  # left
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3:  # right
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        
        # Pickup action
        elif action == 4:
            if np.linalg.norm(self.agent_pos - self.pickup_pos) < 0.5 and not self.holding_item:
                self.holding_item = True
                reward += 25
        
        # Drop action
        elif action == 5:
            if self.holding_item and np.linalg.norm(self.agent_pos - self.dest_pos) < 0.5:
                self.holding_item = False
                reward += 200  # Big reward for successful delivery
                done = True
            elif self.holding_item:
                self.holding_item = False
                reward -= 5
            else:
                done = False
                return self._get_observation(), reward, done, {}
        
        # Reward for moving towards pickup (if not holding)
        if not self.holding_item:
            dist_to_pickup = np.linalg.norm(self.agent_pos - self.pickup_pos)
            if dist_to_pickup < 2:
                reward += 0.5
        
        # Reward for moving towards destination (if holding)
        else:
            dist_to_dest = np.linalg.norm(self.agent_pos - self.dest_pos)
            if dist_to_dest < 2:
                reward += 0.5
        
        self.last_reward = reward
        done = self.step_count > 100
        
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
            pygame.display.set_caption("Warehouse Delivery RL")
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
        
        # Draw pickup point (BLUE)
        pickup_x = int(self.pickup_pos[0] * self.cell_size + self.cell_size // 2)
        pickup_y = int(self.pickup_pos[1] * self.cell_size + self.cell_size // 2)
        pygame.draw.circle(self.screen, (50, 100, 200), (pickup_x, pickup_y), 18)
        pygame.draw.circle(self.screen, (100, 150, 255), (pickup_x, pickup_y), 14)
        self.screen.blit(self.font.render("P", True, (255, 255, 255)), (pickup_x - 8, pickup_y - 8))
        
        # Draw destination (GREEN)
        dest_x = int(self.dest_pos[0] * self.cell_size + self.cell_size // 2)
        dest_y = int(self.dest_pos[1] * self.cell_size + self.cell_size // 2)
        pygame.draw.circle(self.screen, (50, 200, 50), (dest_x, dest_y), 18)
        pygame.draw.circle(self.screen, (100, 255, 100), (dest_x, dest_y), 14)
        self.screen.blit(self.font.render("D", True, (255, 255, 255)), (dest_x - 8, dest_y - 8))
        
        # Draw agent (RED)
        agent_x = int(self.agent_pos[0] * self.cell_size + self.cell_size // 2)
        agent_y = int(self.agent_pos[1] * self.cell_size + self.cell_size // 2)
        pygame.draw.circle(self.screen, (200, 50, 50), (agent_x, agent_y), 20)
        pygame.draw.circle(self.screen, (255, 100, 100), (agent_x, agent_y), 15)
        
        # If holding item, draw a box around agent
        if self.holding_item:
            pygame.draw.rect(self.screen, (255, 200, 0), (agent_x - 22, agent_y - 22, 44, 44), 3)
        
        # Draw movement arrow
        if self.last_action is not None and self.last_action < 4:
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
        
        status = "HOLDING" if self.holding_item else "EMPTY"
        self.screen.blit(self.font_small.render(f"Status: {status}", True, (100, 100, 0)),
                        (info_x, info_y + 100))
        
        reward_color = (0, 150, 0) if self.last_reward > 0 else (150, 0, 0)
        self.screen.blit(self.font.render(f"Reward: {self.last_reward:+.2f}", True, reward_color),
                        (info_x, info_y + 130))
        
        pygame.display.flip()
        self.clock.tick(2)
    
    def close(self):
        if self.screen is not None:
            pygame.quit()
