#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

import numpy as np
import Run_PSRL


def tag_for(reward_type, slip_prob, detection):
    tag = reward_type
    if float(slip_prob) > 0:
        tag += f"_slip{slip_prob}"
    if detection != "td":
        tag += f"_{detection}"
    return tag


def complete(seed_dir: Path, episodes: int) -> bool:
    cache = seed_dir / "cache.json"
    if not cache.exists():
        return False
    try:
        data = json.loads(cache.read_text())
    except Exception:
        return False
    for key in ("episodic_regrets", "episode_returns", "cumulative_rewards", "timesteps"):
        if key not in data or len(data[key]) != episodes:
            return False
    meta = data.get("metadata", {})
    return meta.get("num_episodes") == episodes


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

    tag = tag_for(args.reward_type, args.slip_prob, args.detection)
    seed_dir = Path("visualizations") / tag / args.env / args.agent / str(args.seed)

    if complete(seed_dir, args.num_episodes):
        print(f"SKIP complete {args.agent} seed={args.seed} {tag}")
        return

    if seed_dir.exists():
        shutil.rmtree(seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)

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
    agent.vis_cache_dir = None

    print(f"RUN GridWorld reward={args.reward_type} slip={args.slip_prob} detection={args.detection} agent={args.agent} seed={args.seed}")
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
    (seed_dir / "cache.json").write_text(json.dumps(cache, indent=2))

    if not complete(seed_dir, args.num_episodes):
        raise SystemExit(f"incomplete GridWorld output: {seed_dir}")
    print(f"DONE GridWorld {args.agent} seed={args.seed}")


if __name__ == "__main__":
    main()
