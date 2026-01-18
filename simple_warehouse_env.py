import gymnasium
from gymnasium import spaces
import numpy as np
import pygame


class SimpleWarehouseEnv(gymnasium.Env):
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
        
        # Simplified Observation: [rel_target_x, rel_target_y, at_target, holding]
        # Adding 'at_target' helps the agent know when to press PICK/DROP
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(4,), dtype=np.float32
        )
        
        self.screen = None
        self.clock = None
        self.step_count = 0
        self.last_action = None
        self.last_reward = 0
        self.holding_item = False
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset environment with new random positions"""
        super().reset(seed=seed)
        
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
        obs = self._get_observation()
        return obs, {}
    
    def _get_observation(self):
        """Return relative distance ONLY to the current target (Pickup or Dest)"""
        # If not holding, target is pickup. If holding, target is destination.
        target = self.dest_pos if self.holding_item else self.pickup_pos
        
        rel_x = (target[0] - self.agent_pos[0]) / max(1, self.grid_size - 1)
        rel_y = (target[1] - self.agent_pos[1]) / max(1, self.grid_size - 1)
        
        dist = np.linalg.norm(self.agent_pos - target)
        at_target = 1.0 if dist < 0.1 else -1.0 # Significant contrast
        
        return np.array([rel_x, rel_y, at_target, float(self.holding_item)], dtype=np.float32)
    
    def step(self, action):
        """Execute one action - STRICT 2-Part Sequence"""
        self.step_count += 1
        self.last_action = int(action)
        reward = -1.0  # Increased time penalty for urgency
        done = False
        
        # Current status and distance
        target_pos = self.dest_pos if self.holding_item else self.pickup_pos
        dist_before = np.linalg.norm(self.agent_pos - target_pos)
        
        # Movement
        if action < 4:
            if action == 0: self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
            elif action == 1: self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
            elif action == 2: self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
            elif action == 3: self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
            
            dist_after = np.linalg.norm(self.agent_pos - target_pos)
            
            # Efficient Navigation: Small rewards to guide, but high enough to matter
            if dist_after < dist_before:
                reward += 5.0  # Reward moving closer
            elif dist_after > dist_before:
                reward -= 5.0  # Penalty for moving away
            else:
                reward -= 2.0  # Penalty for hitting a wall or staying still
            
        # Action logic
        elif action == 4:  # PICKUP
            if not self.holding_item and dist_before < 0.1:
                self.holding_item = True
                reward += 500.0  # Large reward for pickup
                print(">>> PACKAGE PICKED UP! Heading to Destination...")
            else:
                reward -= 20.0  # Penalty for invalid pickup
                
        elif action == 5:  # DROP
            if self.holding_item and dist_before < 0.1:
                self.holding_item = False  # Reset flag immediately (shows Red)
                reward += 1000.0 # Large reward for completion
                done = True
                print(">>> PACKAGE DELIVERED! MISSION COMPLETE.")
            else:
                reward -= 20.0  # Penalty for invalid drop
        
        # Immediate action requirement: If at target, MUST pick/drop
        if dist_before < 0.1:
            if (not self.holding_item and action != 4) or (self.holding_item and action != 5):
                reward -= 10.0 # Don't just sit there!
        
        self.last_reward = reward
        if self.step_count >= 100:
            done = True
        
        if self.render_mode == 'human':
            self.render(is_done=done)
            
        return self._get_observation(), reward, done, False, {}
    
    def render(self, is_done=False):
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
        
        # Handle Pygame events to keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

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
        
        # Draw agent (Red if empty, Yellow if holding)
        agent_color = (255, 200, 0) if self.holding_item else (200, 50, 50)
        agent_x = int(self.agent_pos[0] * self.cell_size + self.cell_size // 2)
        agent_y = int(self.agent_pos[1] * self.cell_size + self.cell_size // 2)
        
        # Bigger agent circle
        pygame.draw.circle(self.screen, agent_color, (agent_x, agent_y), 22)
        pygame.draw.circle(self.screen, (255, 255, 255), (agent_x, agent_y), 18, 2)
        
        # Success Overlay
        if is_done and not self.holding_item and np.linalg.norm(self.agent_pos - self.dest_pos) < 0.1:
            overlay = pygame.Surface((grid_px, 60))
            overlay.set_alpha(200)
            overlay.fill((50, 200, 50))
            self.screen.blit(overlay, (0, grid_px // 2 - 30))
            text = self.font_large.render("PACKAGE DELIVERED!", True, (255, 255, 255))
            self.screen.blit(text, (grid_px // 2 - text.get_width() // 2, grid_px // 2 - 10))
        
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
        
        status = "HOLDING ITEM" if self.holding_item else "SUCCESS!" if is_done else "LOOKING..."
        status_color = (200, 150, 0) if self.holding_item else (0, 150, 0) if is_done else (100, 100, 100)
        self.screen.blit(self.font.render(status, True, status_color),
                        (info_x, info_y + 100))
        
        # Draw target info
        target = "Done" if is_done else ("Destination (Green)" if self.holding_item else "Package (Blue)")
        self.screen.blit(self.font_small.render(f"Target: {target}", True, (50, 50, 50)),
                        (info_x, info_y + 160))
        
        reward_color = (0, 150, 0) if self.last_reward > 0 else (150, 0, 0)
        self.screen.blit(self.font.render(f"Reward: {self.last_reward:+.2f}", True, reward_color),
                        (info_x, info_y + 130))
        
        pygame.display.flip()
        self.clock.tick(10)  # Faster visualization
    
    def close(self):
        if self.screen is not None:
            pygame.quit()
