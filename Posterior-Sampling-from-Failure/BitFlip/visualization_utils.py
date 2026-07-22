import os
import json


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_episode_trace(out_dir: str, episode_idx: int, trace: dict):
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"episode_{episode_idx:04d}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(trace, f)
    os.replace(tmp, path)
