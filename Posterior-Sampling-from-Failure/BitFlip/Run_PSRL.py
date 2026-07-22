import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

try:
    from .PSRL_Agents import (
        PSRLStandard,
        EarlyStopping,
        Recovery,
        WeightedRecovery,
        WeightedGraphRecovery,
        WeightedDirectionalRecovery,
    )
except ImportError:
    from PSRL_Agents import (
        PSRLStandard,
        EarlyStopping,
        Recovery,
        WeightedRecovery,
        WeightedGraphRecovery,
        WeightedDirectionalRecovery,
    )


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def np_save_atomic(path: str, arr):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, arr, allow_pickle=True)
    os.replace(tmp, path)


def _empty_switches(n_episodes: int):
    arr = np.empty((int(n_episodes),), dtype=object)
    for i in range(int(n_episodes)):
        arr[i] = []
    return arr


def run_one(agent_cls, cfg, out_dir):
    ensure_dir(out_dir)
    agent = agent_cls(out_dir=out_dir, **cfg)

    out = agent.run()
    if len(out) == 2:
        returns, regret = out
        switches = _empty_switches(len(returns))
    else:
        returns, regret, switches = out

    np_save_atomic(os.path.join(out_dir, "returns.npy"), returns)
    np_save_atomic(os.path.join(out_dir, "regret.npy"), regret)
    np_save_atomic(os.path.join(out_dir, "switches.npy"), switches)

    with open(os.path.join(out_dir, "cfg.json"), "w") as f:
        json.dump(cfg, f)

    return returns, regret, switches


def load_existing_regrets(root_dir):
    regs = []
    if not os.path.isdir(root_dir):
        return None
    for name in sorted(os.listdir(root_dir)):
        if not name.startswith("seed_"):
            continue
        path = os.path.join(root_dir, name, "regret.npy")
        if os.path.isfile(path):
            regs.append(np.load(path, allow_pickle=True))
    if len(regs) == 0:
        return None
    return np.stack(regs, axis=0)


def plot_cumulative_regret(regrets_stack, label, out_path, stderr=False, title=None, logy=False):
    regs = np.asarray(regrets_stack, dtype=np.float64)
    cum = np.cumsum(regs, axis=1)
    mean = cum.mean(axis=0)

    plt.figure()
    plt.plot(mean, label=label)

    if stderr and cum.shape[0] > 1:
        se = cum.std(axis=0, ddof=1) / np.sqrt(cum.shape[0])
        x = np.arange(mean.shape[0])
        plt.fill_between(x, mean - se, mean + se, alpha=0.2)

    if logy:
        plt.yscale("log")

    plt.xlabel("Episode")
    plt.ylabel("Cumulative Regret")

    if title is not None:
        plt.title(title)

    plt.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=200)
    plt.close()


def agent_label(agent_key: str) -> str:
    return {
        "psrl": "PSRL",
        "early": "EarlyStopping",
        "recovery": "Recovery",
        "wrecovery": "WeightedRecovery",
        "wgraph": "WeightedGraphRecovery",
        "wdirectional": "WeightedDirectionalRecovery",
    }[agent_key]


def reward_label(reward_type: str) -> str:
    return {"binary": "Sparse", "dense": "Dense", "stochastic": "Stochastic"}[reward_type]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reward_type", type=str, default="binary", choices=["binary", "dense", "stochastic"])
    p.add_argument(
        "--agent",
        type=str,
        default="psrl",
        choices=["psrl", "early", "recovery", "wrecovery", "wgraph", "wdirectional"],
    )
    p.add_argument("--n_bits", type=int, default=16)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--steps", type=int, default=64)

    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--plot_only", action="store_true")
    p.add_argument("--plot-only", dest="plot_only", action="store_true")
    p.add_argument("--stderr", action="store_true")
    p.add_argument("--logy", action="store_true")

    p.add_argument("--td_threshold", type=float, default=0.0)
    p.add_argument("--min_steps", type=int, default=1)
    p.add_argument("--tau", type=float, default=1.0)

    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--detection", type=str, default="td", choices=["td", "advantage"])
    p.add_argument("--gae_lambda", type=float, default=0.95)

    p.add_argument("--out_root", type=str, default="visualizations_bitflip")
    args = p.parse_args()

    agent_map = {
        "psrl": PSRLStandard,
        "early": EarlyStopping,
        "recovery": Recovery,
        "wrecovery": WeightedRecovery,
        "wgraph": WeightedGraphRecovery,
        "wdirectional": WeightedDirectionalRecovery,
    }
    agent_cls = agent_map[args.agent]

    base_cfg = dict(
        n_bits=args.n_bits,
        gamma=args.gamma,
        reward_type=args.reward_type,
        num_episodes=args.episodes,
        max_steps_per_episode=args.steps,
        td_threshold=args.td_threshold,
        min_steps=args.min_steps,
        tau=args.tau,
        temperature=float(args.temperature),
        detection=args.detection,
        gae_lambda=args.gae_lambda,
    )

    root = os.path.join(args.out_root, args.reward_type, args.agent)
    ensure_dir(root)

    n_seeds = (args.seeds if args.seed is None else 1)
    title = (
        f"BitFlip-{args.n_bits} | {reward_label(args.reward_type)} | {agent_label(args.agent)}\n"
        f"seeds={n_seeds} | episodes={args.episodes} | steps={args.steps}"
    )
    plot_path = os.path.join(root, "regret.png")

    if args.plot_only:
        regs = load_existing_regrets(root)
        if regs is None:
            raise RuntimeError(f"No existing regret.npy files found under {root}")
        plot_cumulative_regret(regs, agent_label(args.agent), plot_path, stderr=args.stderr, title=title, logy=args.logy)
        return

    if args.seed is not None:
        cfg = dict(base_cfg)
        cfg["seed"] = int(args.seed)
        out_dir = os.path.join(root, f"seed_{int(args.seed)}")
        run_one(agent_cls, cfg, out_dir)
        return

    regs = []
    for seed in range(args.seeds):
        cfg = dict(base_cfg)
        cfg["seed"] = seed
        out_dir = os.path.join(root, f"seed_{seed}")
        _, reg, _ = run_one(agent_cls, cfg, out_dir)
        regs.append(reg)

    regs = np.stack(regs, axis=0)
    plot_cumulative_regret(regs, agent_label(args.agent), plot_path, stderr=args.stderr, title=title, logy=args.logy)


if __name__ == "__main__":
    main()
