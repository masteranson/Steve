#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

try:
    from .PSRL_Agents import PSRLStandard, EarlyStopping, Recovery
except ImportError:
    from PSRL_Agents import PSRLStandard, EarlyStopping, Recovery


AGENTS = {
    "psrl": PSRLStandard,
    "early": EarlyStopping,
    "recovery": Recovery,
}


def empty_switches(n):
    arr = np.empty((int(n),), dtype=object)
    for i in range(int(n)):
        arr[i] = []
    return arr


def complete(seed_dir: Path, episodes: int) -> bool:
    required = ["returns.npy", "regret.npy", "switches.npy", "cfg.json"]
    if not all((seed_dir / x).exists() for x in required):
        return False
    try:
        returns = np.load(seed_dir / "returns.npy", allow_pickle=True)
        regret = np.load(seed_dir / "regret.npy", allow_pickle=True)
        cfg = json.loads((seed_dir / "cfg.json").read_text())
    except Exception:
        return False
    return len(returns) == episodes and len(regret) == episodes and int(cfg.get("num_episodes", -1)) == episodes


def save_atomic(path: Path, arr):
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, arr, allow_pickle=True)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reward_type", required=True, choices=["binary", "dense", "stochastic"])
    p.add_argument("--agent", required=True, choices=AGENTS.keys())
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n_bits", type=int, default=16)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--td_threshold", type=float, default=0.0)
    p.add_argument("--min_steps", type=int, default=0)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--detection", default="td", choices=["td", "advantage"])
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--out_root", default="visualizations_bitflip_core")
    args = p.parse_args()

    seed_dir = Path(args.out_root) / args.reward_type / args.agent / f"seed_{args.seed}"
    if complete(seed_dir, args.episodes):
        print(f"SKIP complete BitFlip reward={args.reward_type} agent={args.agent} seed={args.seed}")
        return

    if seed_dir.exists():
        shutil.rmtree(seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(
        n_bits=args.n_bits,
        gamma=args.gamma,
        reward_type=args.reward_type,
        num_episodes=args.episodes,
        max_steps_per_episode=args.steps,
        seed=args.seed,
        td_threshold=args.td_threshold,
        min_steps=args.min_steps,
        tau=args.tau,
        temperature=args.temperature,
        detection=args.detection,
        gae_lambda=args.gae_lambda,
        out_dir=None,
    )

    print(f"RUN BitFlip reward={args.reward_type} agent={args.agent} seed={args.seed}")
    agent = AGENTS[args.agent](**cfg)
    out = agent.run()
    if len(out) == 2:
        returns, regret = out
        switches = empty_switches(len(returns))
    else:
        returns, regret, switches = out

    save_atomic(seed_dir / "returns.npy", returns)
    save_atomic(seed_dir / "regret.npy", regret)
    save_atomic(seed_dir / "switches.npy", switches)
    cfg_to_save = dict(cfg)
    cfg_to_save.pop("out_dir", None)
    (seed_dir / "cfg.json").write_text(json.dumps(cfg_to_save, indent=2))

    if not complete(seed_dir, args.episodes):
        raise SystemExit(f"incomplete BitFlip output: {seed_dir}")
    print(f"DONE BitFlip reward={args.reward_type} agent={args.agent} seed={args.seed}")


if __name__ == "__main__":
    main()
