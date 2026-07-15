"""
visual_buffer.py -- Replay buffer for Phase 2's visual pipeline.

Stores raw images alongside latents, hiddens, and transition data so the
vision encoder can be trained via re-encoding during batch sampling.

Why store images and not just latents?
    The vision encoder's weights change during training. If we store only
    latents, they become stale as the encoder evolves. Re-encoding images
    during sampling gives the encoder fresh gradients from the world model
    loss, which is the primary representation-learning signal.

Memory: ~240 MB for 20 000 transitions at 64x64x3 uint8 (reasonable for
a research prototype).
"""
import numpy as np


class VisualReplayBuffer:
    def __init__(self, capacity, img_h, img_w, latent_dim, gru_dim, seed=0):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)

        self.images = np.zeros((capacity, img_h, img_w, 3), dtype=np.uint8)
        self.latents = np.zeros((capacity, latent_dim), np.float32)
        self.next_images = np.zeros((capacity, img_h, img_w, 3), dtype=np.uint8)
        self.next_latents = np.zeros((capacity, latent_dim), np.float32)
        self.hiddens = np.zeros((capacity, gru_dim), np.float32)
        self.next_hiddens = np.zeros((capacity, gru_dim), np.float32)
        self.actions = np.zeros(capacity, np.int32)
        self.rewards = np.zeros(capacity, np.float32)
        self.dones = np.zeros(capacity, np.float32)

        self._size = 0
        self._ptr = 0

    def add(self, image, latent, action, reward, next_image, next_latent,
            hidden, next_hidden, done):
        i = self._ptr
        self.images[i] = image
        self.latents[i] = latent
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_images[i] = next_image
        self.next_latents[i] = next_latent
        self.hiddens[i] = hidden
        self.next_hiddens[i] = next_hidden
        self.dones[i] = float(done)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def __len__(self):
        return self._size

    def sample(self, batch_size):
        if self._size < batch_size:
            return None
        idx = self.rng.integers(0, self._size, size=batch_size)
        return {
            "images": self.images[idx],
            "latents": self.latents[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_images": self.next_images[idx],
            "next_latents": self.next_latents[idx],
            "hiddens": self.hiddens[idx],
            "next_hiddens": self.next_hiddens[idx],
            "dones": self.dones[idx],
            "weights": np.ones(batch_size, np.float32),
        }

    def clear(self):
        self._size = 0
        self._ptr = 0
