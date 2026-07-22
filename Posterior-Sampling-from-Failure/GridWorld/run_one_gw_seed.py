#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

import numpy as np

import Run_PSRL


def seed_complete(seed_dir: Path, seed: int, episodes: int) -> bool:
    cache = seed_dir / "cache.json"
    if not cache.exists():
        return False

    try:
        data = json.loads(cache.read_text())
    except Exception:
        return False

    for key in ["episodic_regrets", "episode_returns", "cumulative_rewards", "timesteps"]:
        if key not in data or len(data[key]) != episodes:
            return False

    meta = data.get("metadata", {})
    if meta.get("num_episodes") != episodes:
        return False

    pngs = list(seed_dir.glob(f"seed_{seed}_ep*.png"))
    if len(pngs) != episodes:
        return False

    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--agent", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--reward-type", required=True, choices=["binary", "euclidean", "stochastic"])
    p.add_argument("--num-episodes", type=int, required=True)
    p.add_argument("--max-steps", type=int, required=True)
    p.add_argument("--slip-prob", type=float, required=True)
    p.add_argument("--detection", required=True, choices=["td", "advantage"])
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--prior-alpha", type=float, default=0.1)
    p.add_argument("--prior-beta", type=float, default=0.1)
    args = p.parse_args()

    if args.env not in Run_PSRL.ENV_PRESETS:
        raise SystemExit(f"unknown env: {args.env}")
    if args.agent not in Run_PSRL.AGENT_CLASSES:
        raise SystemExit(f"unknown agent: {args.agent}")

    tag = args.reward_type
    if args.slip_prob > 0:
        tag += f"_slip{args.slip_prob}"
    if args.detection != "td":
        tag += f"_{args.detection}"

    seed_dir = Path("visualizations") / tag / args.env / args.agent / str(args.seed)

    if seed_complete(seed_dir, args.seed, args.num_episodes):
        print(f"SKIP complete agent={args.agent} seed={args.seed} dir={seed_dir}")
        return

    if seed_dir.exists():
        print(f"DELETE incomplete agent={args.agent} seed={args.seed} dir={seed_dir}")
        shutil.rmtree(seed_dir)

    env_cfg = Run_PSRL.ENV_PRESETS[args.env]
    agent_cls = Run_PSRL.AGENT_CLASSES[args.agent]

    np.random.seed(args.seed)

    agent = agent_cls(
        width=env_cfg["width"],
        height=env_cfg["height"],
        gamma=env_cfg["gamma"],
        candidate_goals=env_cfg["candidate_goals"],
        true_goal=env_cfg["true_goal"],
        walls=env_cfg["walls"],
        origin_state=env_cfg["origin_state"],
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        prior_alpha=args.prior_alpha,
        prior_beta=args.prior_beta,
        reward_type=args.reward_type,
        slip_prob=args.slip_prob,
        detection=args.detection,
        gae_lambda=args.gae_lambda,
    )

    agent.seed = args.seed
    agent.vis_cache_dir = str(seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)

    print(f"RUN agent={args.agent} seed={args.seed} reward={args.reward_type} slip={args.slip_prob} detection={args.detection}")
    res = agent.run()

    cache = {
        "metadata": {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "env": args.env,
            "agent": args.agent,
            "seed": args.seed,
            "reward_type": args.reward_type,
            "num_episodes": args.num_episodes,
            "max_steps": args.max_steps,
            "slip_prob": args.slip_prob,
            "detection": args.detection,
            "gae_lambda": args.gae_lambda,
        },
        "episodic_regrets": res["episodic_regrets"],
        "episode_returns": res["episode_returns"],
        "cumulative_rewards": res["cumulative_rewards"],
        "timesteps": res["timesteps"],
    }

    with open(seed_dir / "cache.json", "w") as f:
        json.dump(cache, f, indent=2)

    if not seed_complete(seed_dir, args.seed, args.num_episodes):
        raise SystemExit(f"incomplete after run: agent={args.agent} seed={args.seed} dir={seed_dir}")

    print(f"DONE agent={args.agent} seed={args.seed}")


if __name__ == "__main__":
    main()
