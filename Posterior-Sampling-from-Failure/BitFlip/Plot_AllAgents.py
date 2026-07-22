import os
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


AGENT_KEYS = ["psrl", "early", "recovery"]

AGENT_LABELS = {
    "psrl": "PSRL",
    "early": "EarlyStopping",
    "recovery": "Recovery",
}

REWARD_LABELS = {"binary": "Binary", "dense": "Euclidean", "stochastic": "Stochastic"}


def _load_regrets(agent_dir: str):
    """
    agent_dir: <out_root>/<reward_type>/<agent>
    loads seed_*/regret.npy
    returns: (S, E) float64 or None
    """
    if not os.path.isdir(agent_dir):
        return None

    regs = []
    for name in sorted(os.listdir(agent_dir)):
        if not name.startswith("seed_"):
            continue
        path = os.path.join(agent_dir, name, "regret.npy")
        if os.path.isfile(path):
            regs.append(np.load(path, allow_pickle=True).astype(np.float64))

    if len(regs) == 0:
        return None
    return np.stack(regs, axis=0)


def _plot_overlay(series_by_agent, out_path, title, stderr: bool, logy: bool):
    plt.figure()

    for agent_key, regs in series_by_agent.items():
        # regs: (S, E)
        cum = np.cumsum(regs, axis=1)
        mean = np.maximum(cum.mean(axis=0), 1e-12)

        plt.plot(mean, label=AGENT_LABELS.get(agent_key, agent_key))

        if stderr and cum.shape[0] > 1:
            se = cum.std(axis=0, ddof=1) / np.sqrt(cum.shape[0])
            x = np.arange(mean.shape[0])
            lower = np.maximum(mean - se, 1e-12)
            upper = np.maximum(mean + se, 1e-12)
            plt.fill_between(x, lower, upper, alpha=0.2)

    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Regret")
    plt.suptitle(title)
    plt.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if logy:
        plt.yscale("log")
    max_eps = max(regs.shape[1] for regs in series_by_agent.values())
    plt.xlim(0, max_eps - 1)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reward_type", type=str, required=True, choices=["binary", "dense", "stochastic"])
    p.add_argument("--out_root", type=str, default="visualizations_bitflip")
    p.add_argument("--agents", nargs="*", default=AGENT_KEYS)
    p.add_argument("--stderr", action="store_true")
    p.add_argument("--n_bits", type=int, default=16)
    p.add_argument("--ylog", action="store_true")
    args = p.parse_args()

    reward_type = args.reward_type
    out_root = args.out_root

    series = {}
    for agent in args.agents:
        agent_dir = os.path.join(out_root, reward_type, agent)
        regs = _load_regrets(agent_dir)
        if regs is None:
            continue
        series[agent] = regs

    if len(series) == 0:
        raise RuntimeError(
            f"No regrets found under {os.path.join(out_root, reward_type)} "
            f"for agents={args.agents}. Did you run training first?"
        )

    out_dir = os.path.join(out_root, reward_type, "_ALL_AGENTS")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "regret.png")

    n_seeds = max(regs.shape[0] for regs in series.values())
    title = f"Cumulative Regret Comparison ({n_seeds}-Seed Average) - {REWARD_LABELS[reward_type]} Reward"
    _plot_overlay(series, out_path, title, stderr=args.stderr, logy=args.ylog)


if __name__ == "__main__":
    main()
