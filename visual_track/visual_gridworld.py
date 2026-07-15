"""
visual_gridworld.py -- Phase 2 environment.

Same grid logic as gridworld.py, but the agent only receives rendered RGB
images. No symbolic state vector is exposed to the policy.

This is intentional: the organism must learn to extract spatial and object
information from pixels alone, as an infant's visual cortex does.

Cell types (matching gridworld.py):
    0 = empty (floor)
    1 = wall
    2 = novel_object_A
    3 = novel_object_B
"""
import numpy as np

EMPTY, WALL, OBJ_A, OBJ_B = 0, 1, 2, 3

ACTIONS = ["up", "down", "left", "right", "interact"]
ACTION_DELTAS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
    4: (0, 0),
}

COLOR_MAP = {
    EMPTY: np.array([220, 220, 210], dtype=np.uint8),
    WALL:  np.array([60,  60,  60],  dtype=np.uint8),
    OBJ_A: np.array([200, 60,  60],  dtype=np.uint8),
    OBJ_B: np.array([60,  60,  200], dtype=np.uint8),
}
AGENT_COLOR = np.array([60, 200, 60], dtype=np.uint8)
BORDER_COLOR = np.array([40, 40, 40], dtype=np.uint8)


class VisualGridWorld:
    def __init__(self, width=12, height=12, n_walls=8, n_obj_a=3, n_obj_b=3,
                 max_steps=200, render_size=64, seed=0):
        self.width, self.height = width, height
        self.max_steps = max_steps
        self.render_size = render_size
        self.rng = np.random.default_rng(seed)
        self.n_walls, self.n_obj_a, self.n_obj_b = n_walls, n_obj_a, n_obj_b
        self.action_dim = len(ACTIONS)
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.height, self.width), dtype=np.int32)
        self._scatter(WALL, self.n_walls)
        self._scatter(OBJ_A, self.n_obj_a)
        self._scatter(OBJ_B, self.n_obj_b)

        while True:
            y = int(self.rng.integers(0, self.height))
            x = int(self.rng.integers(0, self.width))
            if self.grid[y, x] == EMPTY:
                self.pos = (y, x)
                break

        self.step_count = 0
        self.done = False
        self.visited = {self.pos}
        return self.render()

    def _scatter(self, cell_type, n):
        placed = 0
        while placed < n:
            y = int(self.rng.integers(0, self.height))
            x = int(self.rng.integers(0, self.width))
            if self.grid[y, x] == EMPTY:
                self.grid[y, x] = cell_type
                placed += 1

    def render(self):
        """Return (H, W, 3) uint8 RGB image of the current state."""
        rs = self.render_size
        img = np.zeros((rs, rs, 3), dtype=np.uint8)

        cell_h = rs // self.height
        cell_w = rs // self.width
        offset_y = (rs - cell_h * self.height) // 2
        offset_x = (rs - cell_w * self.width) // 2

        for y in range(self.height):
            for x in range(self.width):
                color = COLOR_MAP.get(self.grid[y, x], np.array([0, 0, 0], dtype=np.uint8))
                y0 = offset_y + y * cell_h
                x0 = offset_x + x * cell_w
                img[y0:y0 + cell_h, x0:x0 + cell_w] = color

        ay, ax = self.pos
        y0 = offset_y + ay * cell_h
        x0 = offset_x + ax * cell_w
        margin_h = max(1, cell_h // 5)
        margin_w = max(1, cell_w // 5)
        img[y0 + margin_h:y0 + cell_h - margin_h,
            x0 + margin_w:x0 + cell_w - margin_w] = AGENT_COLOR

        return img

    def step(self, action: int):
        assert not self.done, "call reset() before stepping a finished episode"
        dy, dx = ACTION_DELTAS[action]
        y, x = self.pos
        ny, nx = y + dy, x + dx

        moved = False
        interacted_with = None
        if 0 <= ny < self.height and 0 <= nx < self.width and self.grid[ny, nx] != WALL:
            if action == 4:
                for ady, adx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ay, ax = y + ady, x + adx
                    if 0 <= ay < self.height and 0 <= ax < self.width:
                        if self.grid[ay, ax] in (OBJ_A, OBJ_B):
                            interacted_with = int(self.grid[ay, ax])
                            break
            else:
                self.pos = (ny, nx)
                moved = True
                self.visited.add(self.pos)

        self.step_count += 1
        self.done = self.step_count >= self.max_steps

        info = {"moved": moved, "interacted_with": interacted_with}
        return self.render(), 0.0, self.done, info

    def coverage(self):
        total_free = self.height * self.width - int((self.grid == WALL).sum())
        return len(self.visited) / total_free
