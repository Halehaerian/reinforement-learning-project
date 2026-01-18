import gymnasium
from gymnasium import spaces
import numpy as np
import pygame
from enum import IntEnum


class Actions(IntEnum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    PICKUP = 4
    DROP = 5
    WAIT = 6


class Direction: # Keep for reference or remove if unused
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
    PURPLE = (160, 32, 240)  # Charger Point


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
        self.action_names = ["UP", "DOWN", "LEFT", "RIGHT", "PICK", "DROP"]
        
        # Obs: [rel_target(x,y), holding, wall_U,D,L,R, at_target, batt, rel_chg(x,y), wall_count]
        # Observation space: 16 features
        # [rel_x, rel_y, holding, w_u, w_d, w_l, w_r, at_target, battery, chg_x, chg_y, wall_count, tw_u, tw_d, tw_l, tw_r]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(16,), dtype=np.float32
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
        self.battery = 1.0
        self.needs_charging = False
        
        # Ensure all positions are unique and not on walls
        self.charger_pos = self._get_free_pos()
        while np.array_equal(self.charger_pos, self.agent_pos):
            self.charger_pos = self._get_free_pos()

        self.pickup_pos = self._get_free_pos()
        while (np.array_equal(self.pickup_pos, self.agent_pos) or 
               np.array_equal(self.pickup_pos, self.charger_pos)):
            self.pickup_pos = self._get_free_pos()
            
        self.dest_pos = self._get_free_pos()
        while (np.array_equal(self.dest_pos, self.pickup_pos) or 
               np.array_equal(self.dest_pos, self.agent_pos) or 
               np.array_equal(self.dest_pos, self.charger_pos)):
            self.dest_pos = self._get_free_pos()
            
        self.step_count = 0
        self.last_action = None
        self.last_reward = 0.0
        self.last_pos = self.agent_pos.copy()
        self.last_move_action = None
        self.pos_history = [tuple(self.agent_pos)]
        return self._get_observation(), {}

    def _get_free_pos(self):
        while True:
            pos = np.array([np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)], dtype=np.float32)
            if self.map[int(pos[1]), int(pos[0])] == 0: return pos

    def _get_observation(self):
        # Use existing target logic to ensure consistency
        if self.battery < 0.25 or self.needs_charging:
            target = self.charger_pos
        elif not self.holding_item:
            target = self.pickup_pos
        else:
            target = self.dest_pos
            
        rel_x = (target[0] - self.agent_pos[0]) / (self.grid_size - 1)
        rel_y = (target[1] - self.agent_pos[1]) / (self.grid_size - 1)
        
        rel_x_chg = (self.charger_pos[0] - self.agent_pos[0]) / (self.grid_size - 1)
        rel_y_chg = (self.charger_pos[1] - self.agent_pos[1]) / (self.grid_size - 1)
        
        # 4-Way Wall Radar: UP, DOWN, LEFT, RIGHT
        w_u = float(self._check_wall_at(self.agent_pos[0], self.agent_pos[1] - 1))
        w_d = float(self._check_wall_at(self.agent_pos[0], self.agent_pos[1] + 1))
        w_l = float(self._check_wall_at(self.agent_pos[0] - 1, self.agent_pos[1]))
        w_r = float(self._check_wall_at(self.agent_pos[0] + 1, self.agent_pos[1]))
        
        # Stuck awareness: How many walls are around the agent?
        wall_count = (w_u + w_d + w_l + w_r) / 4.0
        
        dist = np.linalg.norm(self.agent_pos - target)
        at_target = 1.0 if dist < 0.8 else 0.0 
        
        # Target's environment: Is the target near walls?
        tw_u = float(self._check_wall_at(target[0], target[1] - 1))
        tw_d = float(self._check_wall_at(target[0], target[1] + 1))
        tw_l = float(self._check_wall_at(target[0] - 1, target[1]))
        tw_r = float(self._check_wall_at(target[0] + 1, target[1]))
        
        return np.array([
            rel_x, rel_y, 
            float(self.holding_item),
            w_u, w_d, w_l, w_r,
            at_target,
            self.battery,
            rel_x_chg, rel_y_chg,
            wall_count,
            tw_u, tw_d, tw_l, tw_r
        ], dtype=np.float32)

    def _check_wall_at(self, x, y):
        """Helper: Checks for wall/boundary at specific coordinated."""
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return 1.0
        return 1.0 if self.map[int(y), int(x)] == 1 else 0.0

    def step(self, action):
        self.step_count += 1
        self.last_action = int(action)
        
        # --- BATTERY DRAIN ---
        self.battery -= 0.02 # 2% each step
        if self.battery <= 0:
            return self._get_observation(), -100.0, True, False, {} # Reduced death penalty
            
        reward = -1.0  # Standard time step penalty (encourage speed)
        done = False
        
        # Determine Current Target
        if self.battery < 0.25:
            self.needs_charging = True
            
        if self.needs_charging:
            active_target = self.charger_pos
            target_type = "CHARGER"
        elif not self.holding_item:
            target_type = "PICKUP"
            active_target = self.pickup_pos
        else:
            target_type = "DESTINATION"
            active_target = self.dest_pos

        self.target_type = target_type

        # Reset history on target switch
        if not hasattr(self, 'last_target_type') or self.last_target_type != target_type:
            self.pos_history = []
            self.last_target_type = target_type
            
        dist_before = np.linalg.norm(self.agent_pos - active_target)
        
        # 1. Action Execution
        if action < 4: # MOVE Actions
            dx, dy = 0, 0
            if action == Actions.MOVE_UP: dy = -1
            elif action == Actions.MOVE_DOWN: dy = 1
            elif action == Actions.MOVE_LEFT: dx = -1
            elif action == Actions.MOVE_RIGHT: dx = 1
            
            new_x = self.agent_pos[0] + dx
            new_y = self.agent_pos[1] + dy
            
            # Check for walls/borders
            if (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size and 
                self.map[int(new_y), int(new_x)] == 0):
                
                self.agent_pos[0] = new_x
                self.agent_pos[1] = new_y
                dist_after = np.linalg.norm(self.agent_pos - active_target)
                
                # --- PROGRESS REWARD ---
                # Positive if closer, Negative if further
                progress = dist_before - dist_after
                reward += progress * 10.0 
                
                # --- POSITION HISTORY (Anti-Loop) ---
                curr_pos_tuple = (int(self.agent_pos[0]), int(self.agent_pos[1]))
                if curr_pos_tuple in self.pos_history:
                    # Penalize revisiting spots, but not paralyzed
                    reward -= 5.0 
                
                self.pos_history.append(curr_pos_tuple)
                if len(self.pos_history) > 8:
                    self.pos_history.pop(0)

            else:
                # WALL COLLISION
                reward -= 10.0 # Penalty for hitting wall (wasted step)
                # No movement happened
            
        elif action == Actions.WAIT:
            # Only good if at charger and needs charging
            dist_to_chg = np.linalg.norm(self.agent_pos - self.charger_pos)
            if self.needs_charging and dist_to_chg < 1.0:
                reward += 1.0 # Slight bonus for waiting at charger
            else:
                reward -= 5.0 # Penalty for waiting unnecessarily
                
        elif action == Actions.PICKUP:
            if not self.holding_item and np.linalg.norm(self.agent_pos - self.pickup_pos) < 1.0:
                self.holding_item = True
                reward += 50.0
                print(">>> STATUS: PICKED UP!")
            else:
                reward -= 5.0 # Spammed pickup penalty
                
        elif action == Actions.DROP:
            if self.holding_item and np.linalg.norm(self.agent_pos - self.dest_pos) < 1.0:
                self.holding_item = False
                reward += 100.0 
                done = True
                print(">>> STATUS: MISSION COMPLETE!")
            else:
                reward -= 5.0

        # --- AUTO-TRIGGER / PROXIMITY LOGIC ---
        # Automatically trigger events if VERY close (convenience for RL)
        dist_to_active = np.linalg.norm(self.agent_pos - active_target)
        if dist_to_active < 0.5: 
            reward += 10.0 # Bonus for reaching target
            
            if target_type == "CHARGER":
                self.battery = min(1.0, self.battery + 0.3) 
                if self.battery >= 0.95 and self.needs_charging:
                    self.needs_charging = False
                    print(f">>> STATUS: CHARGED! Resuming Mission.")
            elif target_type == "PICKUP" and not self.holding_item:
                self.holding_item = True
                reward += 50.0
                print(">>> STATUS: AUTO-PICKUP!")
            elif target_type == "DESTINATION" and self.holding_item:
                self.holding_item = False
                reward += 100.0
                done = True
                print(">>> STATUS: AUTO-DROP! SUCCESS!")
            
        if self.step_count >= 200:
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
        
        # Charger
        cx, cy = self._to_px(self.charger_pos)
        pygame.draw.rect(self.screen, Colors.PURPLE, (cx-15, cy-15, 30, 30))
        self.screen.blit(self.font.render("CHG", True, Colors.WHITE), (cx-16, cy-6))

        # Agent
        ax, ay = self._to_px(self.agent_pos)
        # Agent is RED while searching, YELLOW while holding
        color = Colors.YELLOW if self.holding_item else (220, 0, 0)
        pygame.draw.circle(self.screen, color, (ax, ay), 25)

        # Info Panel
        px_off = self.grid_size * self.cell_size + 20
        self.screen.blit(self.font.render(f"Step: {self.step_count}", True, Colors.BLACK), (px_off, 40))
        
        # Battery display
        batt_color = Colors.GREEN if self.battery > 0.4 else (255, 100, 0)
        if self.battery < 0.2: batt_color = Colors.RED
        self.screen.blit(self.font.render(f"Battery: {self.battery*100:.0f}%", True, batt_color), (px_off, 70))
        
        # STATUS TEXT
        batt_low = self.battery < 0.2
        is_looping = tuple(self.agent_pos) in self.pos_history[:-1] if hasattr(self, 'pos_history') else False
        
        if batt_low:
            status_text = "LOW BATTERY! (Go to CHG)"
            status_color = Colors.RED
        elif is_looping:
            status_text = "LOOP DETECTED!"
            status_color = (255, 140, 0) # Orange
        elif not self.holding_item:
            status_text = "GO TO PICKUP"
            status_color = Colors.BLUE
        else:
            status_text = "GO TO DESTINATION"
            status_color = Colors.YELLOW
            
        self.screen.blit(self.font.render(f"STATUS:", True, Colors.BLACK), (px_off, 110))
        self.screen.blit(self.font.render(status_text, True, status_color), (px_off, 135))
        
        self.screen.blit(self.font.render(f"Action: {self.action_names[self.last_action] if self.last_action is not None else 'IDLE'}", True, Colors.BLACK), (px_off, 170))
        self.screen.blit(self.font.render(f"Reward: {self.last_reward:+.1f}", True, Colors.BLACK), (px_off, 200))
        
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
            
            # Draw MISSION END separately, below the main banner
            end_color = Colors.GREEN if success else Colors.RED
            self.screen.blit(self.font.render("MISSION END", True, end_color), (px_off, 230))
            
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
