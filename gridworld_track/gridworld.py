"""
gridworld.py — Phase 1 environment.

A minimal grid the newborn agent explores. No task reward — the only
signal it receives is intrinsic (see rnd.py). Object types exist purely
to give the agent something to be curious about.

State representation: small fixed feature vector, NOT a raw patch.
    [x_norm, y_norm, nearest_obj_dx, nearest_obj_dy, nearest_obj_type_onehot(3), step_norm]
    -> state_dim = 2 + 2 + 3 + 1 = 8

Cell types:
    0 = empty
    1 = wall
    2 = novel_object_A
    3 = novel_object_B
"""
import numpy as np

EMPTY, WALL, OBJ_A, OBJ_B = 0, 1, 2, 3

ACTIONS = ["up", "down", "left", "right", "interact"]
ACTION_DELTAS = {
    0: (-1, 0),   # up
    1: (1, 0),    # down
    2: (0, -1),   # left
    3: (0, 1),    # right
    4: (0, 0),    # interact (no movement)
}


class GridWorld:
    def __init__(self, width=12, height=12, n_walls=8, n_obj_a=3, n_obj_b=3,
                 max_steps=200, seed=0):
        self.width, self.height = width, height
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.n_walls, self.n_obj_a, self.n_obj_b = n_walls, n_obj_a, n_obj_b
        self.state_dim = 8
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
        self.visited = {self.pos}  # coverage tracking: cells visited this episode
        return self._obs()

    def _scatter(self, cell_type, n):
        placed = 0
        while placed < n:
            y = int(self.rng.integers(0, self.height))
            x = int(self.rng.integers(0, self.width))
            if self.grid[y, x] == EMPTY:
                self.grid[y, x] = cell_type
                placed += 1

    def _nearest_object(self):
        """Returns (dy, dx, type_onehot) to the nearest non-empty, non-wall cell."""
        y0, x0 = self.pos
        best_dist = None
        best = (0, 0, np.zeros(3, np.float32))
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y, x]
                if cell in (OBJ_A, OBJ_B):
                    d = abs(y - y0) + abs(x - x0)
                    if best_dist is None or d < best_dist:
                        best_dist = d
                        onehot = np.zeros(3, np.float32)
                        onehot[cell - 1] = 1.0  # WALL=1->idx0, OBJ_A=2->idx1, OBJ_B=3->idx2
                        best = (y - y0, x - x0, onehot)
        return best

    def _obs(self):
        y, x = self.pos
        dy, dx, onehot = self._nearest_object()
        return np.array([
            y / self.height,
            x / self.width,
            np.clip(dy / self.height, -1, 1),
            np.clip(dx / self.width, -1, 1),
            *onehot,
            self.step_count / self.max_steps,
        ], dtype=np.float32)

    def step(self, action: int):
        assert not self.done, "call reset() before stepping a finished episode"
        dy, dx = ACTION_DELTAS[action]
        y, x = self.pos
        ny, nx = y + dy, x + dx

        moved = False
        interacted_with = None
        if 0 <= ny < self.height and 0 <= nx < self.width and self.grid[ny, nx] != WALL:
            if action == 4:  # interact: stay in place, check adjacent cells
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
        return self._obs(), 0.0, self.done, info  # extrinsic reward always 0 in Phase 1

    def coverage(self):
        """Fraction of non-wall cells visited so far this episode."""
        total_free = self.height * self.width - int((self.grid == WALL).sum())
        return len(self.visited) / total_free
