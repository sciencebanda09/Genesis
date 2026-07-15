"""
augmentations.py — perception robustness transforms for Genesis.

Applies random augmentations to image observations before they enter the
encoder. These force the encoder to learn transformation-invariant features
rather than memorizing pixel-level patterns.

Lighting: random brightness/contrast shift
Rotation: small random rotation with nearest-neighbor fill
Noise: Gaussian noise injection
Occlusion: random rectangular patches dropped out

Each augmentation accepts and returns (C, H, W) float32 arrays normalized
in [0, 1], matching the encoder's input contract.
"""
import numpy as np


def lighting(img: np.ndarray, brightness=0.1, contrast=0.1, rng=None):
    """Random brightness shift and contrast scaling."""
    rng = rng or np.random.default_rng()
    img = np.asarray(img, np.float32).copy()
    b = rng.uniform(-brightness, brightness)
    c = 1.0 + rng.uniform(-contrast, contrast)
    img = img * c + b
    return np.clip(img, 0.0, 1.0)


def rotation(img: np.ndarray, max_deg=10, rng=None):
    """Small random rotation with zero-padding edges.

    Uses manual bilinear-ish mapping (no scipy dependency).
    """
    rng = rng or np.random.default_rng()
    angle = rng.uniform(-max_deg, max_deg)
    if abs(angle) < 0.5:
        return img
    theta = np.radians(angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    C, H, W = img.shape
    cy, cx = H / 2, W / 2

    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    y_off = (y - cy) * cos_t - (x - cx) * sin_t + cy
    x_off = (y - cy) * sin_t + (x - cx) * cos_t + cx

    y_off = np.clip(y_off, 0, H - 1).astype(np.int32)
    x_off = np.clip(x_off, 0, W - 1).astype(np.int32)
    out = img[:, y_off, x_off]
    return out


def gaussian_noise(img: np.ndarray, std=0.02, rng=None):
    """Add Gaussian noise. std relative to [0,1] range."""
    rng = rng or np.random.default_rng()
    return np.clip(img + rng.normal(0, std, img.shape), 0.0, 1.0)


def occlusion(img: np.ndarray, max_boxes=2, max_size=0.2, rng=None):
    """Drop random rectangular patches (set to 0)."""
    rng = rng or np.random.default_rng()
    img = img.copy()
    C, H, W = img.shape
    n = rng.integers(0, max_boxes + 1)
    for _ in range(n):
        bw = int(W * rng.uniform(0.05, max_size))
        bh = int(H * rng.uniform(0.05, max_size))
        x = rng.integers(0, W - bw)
        y = rng.integers(0, H - bh)
        img[:, y:y+bh, x:x+bw] = 0.0
    return img


def compose(img: np.ndarray, *, rng=None,
            p_lighting=0.8, p_rotation=0.5, p_noise=0.3, p_occlusion=0.3):
    """Apply a random subset of augmentations."""
    rng = rng or np.random.default_rng()
    if rng.uniform() < p_lighting:
        img = lighting(img, rng=rng)
    if rng.uniform() < p_rotation:
        img = rotation(img, rng=rng)
    if rng.uniform() < p_noise:
        img = gaussian_noise(img, rng=rng)
    if rng.uniform() < p_occlusion:
        img = occlusion(img, rng=rng)
    return img
