"""
logger.py — JSONL trajectory logger (Phase 2 extension).

Phase 1 schema preserved unchanged. Phase 2 adds optional fields:
    image_b64   : base64-encoded PNG of the rendered observation
    latent      : list of floats (the vision encoder output)

Both are optional — if omitted the record matches Phase 1 exactly,
maintaining backward compatibility with any Phase 3 reader.

One line per env step:
    {type="step", episode, step, global_step, obs, action, action_name,
     extrinsic_reward, intrinsic_reward, done, interacted_with, timestamp,
     image_b64?, latent?}

Plus one line per training update, tagged distinctly (type="update").
"""
import base64
import gzip
import json
import time
from pathlib import Path


class JsonlLogger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", buffering=1)

    @staticmethod
    def _image_to_b64(image):
        """Compress uint8 (H,W,3) image as gzip + base64."""
        raw = image.tobytes()
        compressed = gzip.compress(raw)
        return base64.b64encode(compressed).decode("ascii")

    def log_step(self, episode, step, global_step, obs, action, action_name,
                 extrinsic_reward, intrinsic_reward, done, interacted_with=None,
                 image=None, latent=None):
        record = {
            "type": "step",
            "episode": episode,
            "step": step,
            "global_step": global_step,
            "obs": [round(float(o), 5) for o in obs],
            "action": int(action),
            "action_name": action_name,
            "extrinsic_reward": float(extrinsic_reward),
            "intrinsic_reward": float(intrinsic_reward),
            "done": bool(done),
            "interacted_with": interacted_with,
            "timestamp": time.time(),
        }
        if image is not None:
            record["image_b64"] = self._image_to_b64(image)
        if latent is not None:
            record["latent"] = [round(float(v), 5) for v in latent]
        self._f.write(json.dumps(record) + "\n")

    def log_update(self, global_step, stats: dict):
        record = {"type": "update", "global_step": global_step, "timestamp": time.time(),
                   **stats}
        self._f.write(json.dumps(record) + "\n")

    def close(self):
        self._f.close()
