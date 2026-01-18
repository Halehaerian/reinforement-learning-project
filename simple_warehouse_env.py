import gymnasium
from gymnasium import spaces
import numpy as np
import pygame
from enum import IntEnum


class Actions(IntEnum):
    TURN_LEFT = 0
    TURN_RIGHT = 1
    FORWARD = 2
    PICKUP = 3
    DROP = 4


class Direction(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


class Colors:
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY_LIGHT = (240, 240, 240)
    GRAY_DARK = (80, 80, 80)
    WALL = (40, 40, 40)
    BLUE = (50, 100, 255)    # Pickup Point
    GREEN = (50, 200, 50)    # (Unused or Success)
    RED = (255, 50, 50)      # Destination Point (Agent is also Red/Yellow)
    YELLOW = (255, 220, 0)   # Carrying State


class AdvancedRobotWarehouseEnv(gymnasium.Env):
    """
    Robotics-First Warehouse Environment v3.1
    Fixed: Robot spinning in place. Optimized rewards for translation.
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, render_mode=None):
        super(AdvancedRobotWarehouseEnv, self).__init__()
        self.grid_size = 7
        self.render_mode = render_mode
        self.cell_size = 70
        
        self.action_space = spaces.Discrete(len(Actions))
        self.action_names = ["TURN L", "TURN R", "MOVE FWD", "PICK", "DROP"]
        
        # Obs: [rel_x, rel_y, dir_sin, dir_cos, holding, wall_F, wall_L, wall_R, at_target]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(9,), dtype=np.float32
        )
        
        # Map: 1 = Wall
        self.map = np.zeros((self.grid_size, self.grid_size))
        self.map[1, 1:4] = 1
        self.map[5, 3:6] = 1
        self.map[3, 3] = 1
        
        self.reset()
        self.screen = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self._get_free_pos()
        self.agent_dir = Direction.NORTH
        self.holding_item = False
        self.pickup_pos = self._get_free_pos()
        while np.array_equal(self.pickup_pos, self.agent_pos):
            self.pickup_pos = self._get_free_pos()
        self.dest_pos = self._get_free_pos()
        while np.array_equal(self.dest_pos, self.pickup_pos) or np.array_equal(self.dest_pos, self.agent_pos):
            self.dest_pos = self._get_free_pos()
            
        self.step_count = 0
        self.last_action = None
        self.last_reward = 0.0
        return self._get_observation(), {}

    def _get_free_pos(self):
        while True:
            pos = np.array([np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)], dtype=np.float32)
            if self.map[int(pos[1]), int(pos[0])] == 0: return pos

    def _get_observation(self):
        target = self.dest_pos if self.holding_item else self.pickup_pos
        rel_x = (target[0] - self.agent_pos[0]) / (self.grid_size - 1)
        rel_y = (target[1] - self.agent_pos[1]) / (self.grid_size - 1)
        angle = self.agent_dir * (np.pi / 2)
        
        # 3-Way Wall Radar: Front, Left, Right
        w_f = float(self._check_wall(self.agent_dir))
        w_l = float(self._check_wall((self.agent_dir - 1) % 4))
        w_r = float(self._check_wall((self.agent_dir + 1) % 4))
        
        dist = np.linalg.norm(self.agent_pos - target)
        at_target = 1.0 if dist < 0.9 else 0.0 
        
        return np.array([
            rel_x, rel_y, 
            np.sin(angle), np.cos(angle), 
            float(self.holding_item),
            w_f, w_l, w_r,
            at_target
        ], dtype=np.float32)

    def _check_wall(self, look_dir):
        """Helper: Checks for wall/boundary in a specific direction from current pos."""
        new_pos = self.agent_pos.copy()
        if look_dir == Direction.NORTH: new_pos[1] -= 1
        elif look_dir == Direction.SOUTH: new_pos[1] += 1
        elif look_dir == Direction.EAST:  new_pos[0] += 1
        elif look_dir == Direction.WEST:  new_pos[0] -= 1
        
        if not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
            return 1.0
        return 1.0 if self.map[int(new_pos[1]), int(new_pos[0])] == 1 else 0.0

    def step(self, action):
        self.step_count += 1
        self.last_action = int(action)
        
        # --- URGENCY: HIGH STEP PENALTY ---
        reward = -5.0  # Every single step costs money to force speed
        done = False
        
        active_target = self.dest_pos if self.holding_item else self.pickup_pos
        self.dist_before = np.linalg.norm(self.agent_pos - active_target)
        dist_before = self.dist_before
        
        # --- LURE REWARD: Encourage staying on the target cell to learn interaction ---
        if dist_before < 0.9:
            reward += 10.0 # Cancels step penalty and gives small bonus for reaching target
        
        # 1. Action Execution
        if action == Actions.TURN_LEFT or action == Actions.TURN_RIGHT:
            if action == Actions.TURN_LEFT: self.agent_dir = (self.agent_dir - 1) % 4
            else: self.agent_dir = (self.agent_dir + 1) % 4
            
        elif action == Actions.FORWARD:
            new_pos = self.agent_pos.copy()
            if self.agent_dir == Direction.NORTH: new_pos[1] -= 1
            elif self.agent_dir == Direction.SOUTH: new_pos[1] += 1
            elif self.agent_dir == Direction.EAST:  new_pos[0] += 1
            elif self.agent_dir == Direction.WEST:  new_pos[0] -= 1
            
            if (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size and 
                self.map[int(new_pos[1]), int(new_pos[0])] == 0):
                
                self.agent_pos = new_pos
                dist_after = np.linalg.norm(self.agent_pos - active_target)
                
                # --- AUTO-TRIGGER Logic ---
                # Check if we landed on the active target cell (center)
                if dist_after < 0.2:
                    if not self.holding_item:
                        self.holding_item = True
                        reward += 5000.0
                        print(">>> AUTO-PICKUP! (Heading to Destination...)")
                    else:
                        self.holding_item = False # Drop the parcel
                        reward += 10000.0
                        done = True
                        print(">>> MISSION SUCCESS! (Dropped at Destination)")
                
                if dist_after < dist_before:
                    reward += 60.0 # Increased reward for progress
                else:
                    reward -= 2.0  # LOWER penalty for moving away (allows maneuvering around walls)
            else:
                reward -= 100.0 # HEAVIER penalty for wall crash
            
        elif action == Actions.PICKUP:
            if not self.holding_item and dist_before < 0.9:
                self.holding_item = True
                reward += 3000.0 # Increased reward for clarity
                print(">>> STATUS: PICKED UP! (Robot Color -> Yellow)")
            else:
                reward -= 100.0 # Penalty for misuse
                
        elif action == Actions.DROP:
            if self.holding_item and dist_before < 0.9:
                self.holding_item = False # Color returns to Red instantly
                reward += 5000.0 
                done = True
                print(f">>> STATUS: MISSION COMPLETE! (Total Steps: {self.step_count})")
            else:
                reward -= 100.0
            
        if self.step_count >= 150: # Shorter timeout to discourage slow agents
            done = True
        self.last_reward = reward
        if self.render_mode == 'human': self.render(is_done=done)
        return self._get_observation(), reward, done, False, {}

    def render(self, is_done=False):
        if self.render_mode != 'human': return
        self._init_pygame()
        self.screen.fill(Colors.GRAY_LIGHT)
        # Draw Map
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                rect = (c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                color = Colors.WALL if self.map[r, c] == 1 else Colors.WHITE
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, Colors.GRAY_DARK, rect, 1)

        # Targets with Text Labels
        px, py = self._to_px(self.pickup_pos)
        pygame.draw.circle(self.screen, Colors.BLUE, (px, py), 15)
        self.screen.blit(self.font.render("PICK", True, Colors.WHITE), (px-16, py-6))
        
        dx, dy = self._to_px(self.dest_pos)
        # DESTINATION is RED as requested
        pygame.draw.circle(self.screen, Colors.RED, (dx, dy), 15)
        self.screen.blit(self.font.render("DEST", True, Colors.WHITE), (dx-16, dy-6))

        # Agent
        ax, ay = self._to_px(self.agent_pos)
        # Agent is RED while searching, YELLOW while holding
        # If search agent is Red, we draw a darker red or keep it bright
        color = Colors.YELLOW if self.holding_item else (220, 0, 0)
        pygame.draw.circle(self.screen, color, (ax, ay), 25)
        # Nose
        nose_dir = [(0, -25), (25, 0), (0, 25), (-25, 0)][self.agent_dir]
        pygame.draw.line(self.screen, Colors.BLACK, (ax, ay), (ax + nose_dir[0], ay + nose_dir[1]), 5)

        # Info Panel
        px_off = self.grid_size * self.cell_size + 20
        self.screen.blit(self.font.render(f"Step: {self.step_count}", True, Colors.BLACK), (px_off, 40))
        
        # STATUS TEXT
        status_text = "PICKED UP!" if self.holding_item else "SEARCHING..."
        if is_done and not self.holding_item and self.dist_before < 1.1: 
            status_text = "DROPPED SUCCESSFULLY!"
        
        status_color = Colors.YELLOW if self.holding_item else Colors.BLUE
        if is_done and not self.holding_item: status_color = Colors.GREEN
            
        self.screen.blit(self.font.render(f"STATUS:", True, Colors.BLACK), (px_off, 70))
        self.screen.blit(self.font.render(status_text, True, status_color), (px_off, 95))
        
        self.screen.blit(self.font.render(f"Action: {self.action_names[self.last_action] if self.last_action is not None else 'IDLE'}", True, Colors.BLACK), (px_off, 130))
        self.screen.blit(self.font.render(f"Reward: {self.last_reward:+.1f}", True, Colors.BLACK), (px_off, 160))
        
        if is_done:
            # MISSION COMPLETE BANNER
            banner = pygame.Surface((self.grid_size * self.cell_size, 80))
            banner.set_alpha(220)
            
            # Mission is successful if we finished without an item (meaning we dropped it)
            success = is_done and not self.holding_item and self.step_count < 150
            banner.fill((0, 150, 0) if success else (180, 0, 0))
            self.screen.blit(banner, (0, (self.grid_size * self.cell_size)//2 - 40))
            
            msg = "MISSION SUCCESS!" if success else "MISSION FAILED (TIMEOUT)"
            txt = self.font.render(msg, True, Colors.WHITE)
            self.screen.blit(txt, ((self.grid_size * self.cell_size)//2 - txt.get_width()//2, 
                                  (self.grid_size * self.cell_size)//2 - 10))
            
            self.screen.blit(self.font.render("MISSION END", True, (255, 255, 255)), (px_off, 200))
            
        pygame.display.flip()
        self.clock.tick(15)

    def _init_pygame(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.grid_size * self.cell_size + 250, self.grid_size * self.cell_size))
            pygame.display.set_caption("Robot Maze Solver")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 28)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); exit()

    def _to_px(self, pos):
        return (int(pos[0] * self.cell_size + self.cell_size // 2), int(pos[1] * self.cell_size + self.cell_size // 2))

    def close(self):
        if self.screen is not None: pygame.quit()
