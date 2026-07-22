import argparse
import datetime as _dt
import json
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from .PSRL_Agents import (
        PSRLStandard,
        PSRLEarlyStopping,
        PSRLWithRecovery,
        PSRLWeightedRecovery,
        PSRLWeightedGraphRecovery,
        PSRLWeightedDirectionalRecovery,
        PSRLWithLEAST,
        PSRLWithSEE,
    )
except ImportError:
    from PSRL_Agents import (
        PSRLStandard,
        PSRLEarlyStopping,
        PSRLWithRecovery,
        PSRLWeightedRecovery,
        PSRLWeightedGraphRecovery,
        PSRLWeightedDirectionalRecovery,
        PSRLWithLEAST,
        PSRLWithSEE,
    )

# --------------------------------------------------------------------------- #
#  Environment presets                                                        #
# --------------------------------------------------------------------------- #
ENV_PRESETS = {
    '2_candidates': {
        'width': 13,
        'height': 13,
        'gamma': 0.9,
        'walls': [
            (0,6),(1,6),(4,6),(5,6),(8,6),(11,6),(12,6),
            (6,0),(6,1),(6,4),(6,5),(6,6),
            (7,6),(7,7),(7,8),(7,9),(7,10),(7,11),(7,12)
        ],
        'origin_state': (10,2),
        'candidate_goals': [(0,12), (0,0), (6,12)],
        'true_goal': (0,12)
    },
    '5_candidates': {
        'width': 13,
        'height': 13,
        'gamma': 0.9,
        'walls': [
            (0,6),(1,6),(4,6),(5,6),(8,6),(11,6),(12,6),
            (6,0),(6,1),(6,4),(6,5),(6,6),
            (7,6),(7,7),(7,8),(7,9),(7,10),(7,11),(7,12)
        ],
        'origin_state': (10,2),
        'candidate_goals': [(0,12), (0,1), (1,0), (6,12), (12,11), (11,12)],
        'true_goal': (0,12)
    },
    '10_candidates': {
        'width': 13,
        'height': 13,
        'gamma': 0.9,
        'walls': [
            (0,6),(1,6),(4,6),(5,6),(8,6),(11,6),(12,6),
            (6,0),(6,1),(6,4),(6,5),(6,6),
            (7,6),(7,7),(7,8),(7,9),(7,10),(7,11),(7,12)
        ],
        'origin_state': (10,2),
        'candidate_goals': [(0,12), (0,1), (1,0), (0,4), (2,2), (4,0), (12,11), (11,12), (12,8), (10,10), (8,12)],
        'true_goal': (0,12)
    },
    '15_candidates': {
        'width': 13,
        'height': 13,
        'gamma': 0.9,
        'walls': [
            (0,6),(1,6),(4,6),(5,6),(8,6),(11,6),(12,6),
            (6,0),(6,1),(6,4),(6,5),(6,6),
            (7,6),(7,7),(7,8),(7,9),(7,10),(7,11),(7,12)
        ],
        'origin_state': (10,2),
        'candidate_goals': [(0,12), (0,1), (1,0), (0,4), (2,2), (4,0), (12,11), (11,12), (12,8), (10,10), (8,12), (11,7), (8,10), (6,10), (5,4), (4,5)],
        'true_goal': (0,12)
    },
    'room1': {
        'width': 13,
        'height': 13,
        'gamma': 0.9,
        'walls': [
            (0,6),(1,6),(4,6),(5,6),(8,6),(11,6),(12,6),
            (6,0),(6,1),(6,4),(6,5),(6,6),
            (7,6),(7,7),(7,8),(7,9),(7,10),(7,11),(7,12)
        ],
        'origin_state': (10,2),
        'candidate_goals': [(0,12), (0,1), (1,0), (0,4), (2,2), (4,0), (5,4), (4,5)],
        'true_goal': (0,12)
    },
    'room4': {
        'width': 13,
        'height': 13,
        'gamma': 0.9,
        'walls': [
            (0,6),(1,6),(4,6),(5,6),(8,6),(11,6),(12,6),
            (6,0),(6,1),(6,4),(6,5),(6,6),
            (7,6),(7,7),(7,8),(7,9),(7,10),(7,11),(7,12)
        ],
        'origin_state': (10,2),
        'candidate_goals': [(0,12), (12,11), (11,12), (12,8), (10,10), (8,12), (11,7), (8,10), (6,10)],
        'true_goal': (0,12)
    }
}

# --------------------------------------------------------------------------- #
#  Agent mapping                                                              #
# --------------------------------------------------------------------------- #
AGENT_CLASSES = {
    "standard": PSRLStandard,
    "early_stopping": PSRLEarlyStopping,
    "recovery": PSRLWithRecovery,
    "weighted_recovery": PSRLWeightedRecovery,
    "weighted_graph": PSRLWeightedGraphRecovery,
    "weighted_directional": PSRLWeightedDirectionalRecovery,
    "least": PSRLWithLEAST,
    "see_oracle": PSRLWithSEE,
    "see": PSRLWithSEE,  # backward-compatible alias for see_oracle
}


# --------------------------------------------------------------------------- #
#  Argument parsing                                                           #
# --------------------------------------------------------------------------- #


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PSRL variants or plot cached results"
    )

    # Common filters -------------------------------------------------------- #
    parser.add_argument("--env", choices=ENV_PRESETS.keys(), help="Environment preset")
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=AGENT_CLASSES.keys(),
        help="Subset of agents to run / plot (default: all for --plot-only)",
    )
    parser.add_argument(
        "--reward-type",
        choices=["binary", "euclidean", "stochastic"],
        default="binary",
        help="Reward function (default: binary)",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=5,
        help="Number of seeds to *run* or *plot* (see README)",
    )

    # Run-only hyper-parameters --------------------------------------------- #
    parser.add_argument(
        "--num-episodes", type=int, default=20, help="Episodes per agent (run-mode)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=100, help="Max steps per episode (run-mode)"
    )
    parser.add_argument(
        "--prior-alpha", type=float, default=0.1, help="Beta prior alpha (run-mode)"
    )
    parser.add_argument(
        "--prior-beta", type=float, default=0.1, help="Beta prior beta (run-mode)"
    )

    # Stochastic dynamics & detection -------------------------------------- #
    parser.add_argument(
        "--slip-prob",
        type=float,
        default=0.0,
        help="R&N slippage probability (0.0 = deterministic, 0.2 = standard)",
    )
    parser.add_argument(
        "--detection",
        choices=["td", "advantage"],
        default="td",
        help="Failure-detection signal: advantage is realized GAE-style TD; td is oracle expected TD",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda for advantage detection (ignored when --detection=td)",
    )

    # Plot controls --------------------------------------------------------- #
    parser.add_argument(
        "--plot-regret",
        choices=["episodes", "timesteps", "var_episodes", "var_timesteps"],
        help="Choose regret x-axis / stderr style (both modes)",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip simulation; plot cached results instead",
    )

    # Housekeeping ---------------------------------------------------------- #
    parser.add_argument(
        "--clear-vis",
        action="store_true",
        help="Wipe *all* visualizations before running (run-mode only)",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------- #
    #  Validation                                                         #
    # ------------------------------------------------------------------- #
    if args.plot_only:
        if args.clear_vis:
            parser.error("--clear-vis cannot be used with --plot-only.")

        if args.env is None:
            parser.error("--env is required when using --plot-only.")

        # Block obviously run-specific overrides.
        forbidden_if_plot = [
            "num_episodes", "max_steps", "prior_alpha", "prior_beta",
        ]
        for name in forbidden_if_plot:
            if getattr(args, name) != parser.get_default(name):
                parser.error(f"--{name.replace('_', '-')} cannot be used with --plot-only.")

    else:  # run-mode
        if (args.env is None) != (args.agents is None):
            parser.error("Specify both --env and --agents, or neither.")
        if args.env is None and args.plot_regret:
            parser.error("--plot-regret alone does nothing; use --plot-only.")
        if args.env is None:
            parser.error("Run-mode requires --env and --agents (or use --plot-only).")

    # Accept default plot style
    if args.plot_regret is None:
        args.plot_regret = "episodes"

    return args


# --------------------------------------------------------------------------- #
#  Helpers:  paths, cache I/O                                                #
# --------------------------------------------------------------------------- #
def _vis_root(args) -> Path:
    """
    Build a unique output directory that won't collide with other conditions.

    Layout:
        visualizations/{tag}/{env}/...
    where tag encodes reward_type + optional slip + optional detection mode:
        binary                        (original deterministic, td detection)
        binary_slip0.2                (stochastic dynamics, td detection)
        binary_slip0.2_advantage      (stochastic dynamics, advantage detection)
    """
    tag = args.reward_type
    if args.slip_prob > 0:
        tag += f"_slip{args.slip_prob}"
    if args.detection != "td":
        tag += f"_{args.detection}"
    return Path("visualizations") / tag / args.env


def _agent_dir(args, agent: str) -> Path:
    return _vis_root(args) / agent


def _seed_dir(args, agent: str, seed: int) -> Path:
    return _agent_dir(args, agent) / str(seed)


def _save_seed_cache(path: Path, data: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "cache.json", "w") as f:
        json.dump(data, f, indent=2)


def _load_seed_cache(path: Path) -> dict:
    with open(path / "cache.json", "r") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
#  RUN MODE                                                                  #
# --------------------------------------------------------------------------- #
def _run_experiments(args):
    # Housekeeping ---------------------------------------------------------- #
    if args.clear_vis:
        shutil.rmtree("visualizations", ignore_errors=True)

    vis_root = _vis_root(args)
    vis_root.mkdir(parents=True, exist_ok=True)

    # Seeds list
    seeds = list(range(args.num_seeds))

    # Store aggregate results for optional plotting
    agg_results = {}

    env_cfg = ENV_PRESETS[args.env]

    for ag in args.agents:
        regrets, returns, cum_rewards, timesteps = [], [], [], []

        for seed in seeds:
            # Resume: skip seeds that already have cached results
            cached = _seed_dir(args, ag, seed) / "cache.json"
            if cached.exists():
                dat = _load_seed_cache(_seed_dir(args, ag, seed))
                regrets.append(dat["episodic_regrets"])
                returns.append(dat["episode_returns"])
                cum_rewards.append(dat["cumulative_rewards"])
                timesteps.append(dat["timesteps"])
                print(f"  {ag} seed {seed}: loaded from cache")
                continue

            np.random.seed(seed)
            agent_cls = AGENT_CLASSES[ag]
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

            agent.seed = seed
            agent.vis_cache_dir = str(_seed_dir(args, ag, seed))
            Path(agent.vis_cache_dir).mkdir(parents=True, exist_ok=True)

            # Suppress noisy prints except first seed
            if seed != 0:
                import builtins as _bi

                _orig_print = _bi.print
                _bi.print = lambda *a, **k: None

            res = agent.run()
            if seed != 0:
                _bi.print = _orig_print

            regrets.append(res["episodic_regrets"])
            returns.append(res["episode_returns"])
            cum_rewards.append(res["cumulative_rewards"])
            timesteps.append(res["timesteps"])

            # Per-seed cache ------------------------------------------------ #
            seed_cache = {
                "metadata": {
                    "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
                    "env": args.env,
                    "agent": ag,
                    "seed": seed,
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
            _save_seed_cache(_seed_dir(args, ag, seed), seed_cache)

        # Aggregate single-run stats (mean/std) ----------------------------- #
        r_arr = np.array(regrets)
        ret_arr = np.array(returns)
        crew_arr = np.array(cum_rewards)
        t_arr = np.array(timesteps, dtype=float)
        cum_regret = np.cumsum(r_arr, axis=1)
        ddof = 1 if r_arr.shape[0] > 1 else 0

        agg_results[ag] = {
            "mean_regret": r_arr.mean(axis=0),
            "mean_cum_regret": cum_regret.mean(axis=0),
            "se_cum_regret": cum_regret.std(axis=0, ddof=ddof) / np.sqrt(r_arr.shape[0]),
            "mean_return": ret_arr.mean(axis=0),
            "mean_cumulative_rewards": crew_arr.mean(axis=0),
            "mean_timesteps": t_arr.mean(axis=0),
        }

    # Optional plot -------------------------------------------------------- #
    if args.plot_regret:
        _plot_regret_from_aggregate(
            agg_results,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            seeds=len(seeds),
            mode=args.plot_regret,
            reward_type=args.reward_type,
            slip_prob=args.slip_prob,
            detection=args.detection,
        )


# --------------------------------------------------------------------------- #
#  PLOT-ONLY MODE                                                            #
# --------------------------------------------------------------------------- #
def _plot_only(args):
    vis_root = _vis_root(args)
    if not vis_root.exists():
        raise SystemExit(f"No data: {vis_root} does not exist.")

    # Determine agent list ------------------------------------------------- #
    agents = args.agents or sorted(
        [d.name for d in vis_root.iterdir() if d.is_dir()]
    )
    if not agents:
        raise SystemExit("No agent folders present to plot.")

    # Collect seeds per agent --------------------------------------------- #
    first_n = args.num_seeds
    seed_sets = {}
    for ag in agents:
        seeds = sorted(
            int(d.name)
            for d in _agent_dir(args, ag).iterdir()
            if d.is_dir() and d.name.isdigit()
        )
        seed_sets[ag] = seeds
    # Minimum common prefix length
    min_common = min(len(s) for s in seed_sets.values())
    if min_common == 0:
        raise SystemExit("At least one agent has no seed folders.")
    seeds_used = list(range(min(min_common, first_n)))

    # Load & aggregate ----------------------------------------------------- #
    agg_results = {}
    for ag in agents:
        timesteps, regrets = [], []
        for seed in seeds_used:
            cache_p = _seed_dir(args, ag, seed)
            if not cache_p.exists():
                raise SystemExit(f"Missing seed {seed} for agent {ag}.")
            dat = _load_seed_cache(cache_p)
            regrets.append(dat["episodic_regrets"])
            timesteps.append(dat["timesteps"])
            num_episodes = dat["metadata"]["num_episodes"]
            max_steps = dat["metadata"]["max_steps"]

        r_arr = np.array(regrets, dtype=float)
        t_arr = np.array(timesteps, dtype=float)
        cum_regret = np.cumsum(r_arr, axis=1)
        ddof = 1 if r_arr.shape[0] > 1 else 0
        agg_results[ag] = {
            "mean_regret": r_arr.mean(axis=0),
            "mean_cum_regret": cum_regret.mean(axis=0),
            "se_cum_regret": cum_regret.std(axis=0, ddof=ddof) / np.sqrt(r_arr.shape[0]),
            "mean_timesteps": t_arr.mean(axis=0),
        }

    # Truncate all agents to the shortest episode count
    min_eps = min(len(d["mean_regret"]) for d in agg_results.values())
    for ag in agg_results:
        for key in ("mean_regret", "mean_cum_regret", "se_cum_regret", "mean_timesteps"):
            agg_results[ag][key] = agg_results[ag][key][:min_eps]
    num_episodes = min_eps

    # Read slip/detection from first cached seed's metadata
    first_ag = agents[0]
    first_dat = _load_seed_cache(_seed_dir(args, first_ag, seeds_used[0]))
    slip_prob = first_dat["metadata"].get("slip_prob", 0.0)
    detection = first_dat["metadata"].get("detection", "td")

    # Plot ----------------------------------------------------------------- #
    _plot_regret_from_aggregate(
        agg_results,
        num_episodes=num_episodes,
        max_steps=max_steps,
        seeds=len(seeds_used),
        mode=args.plot_regret,
        reward_type=args.reward_type,
        slip_prob=slip_prob,
        detection=detection,
    )


# --------------------------------------------------------------------------- #
#  Plot helper (shared)                                                      #
# --------------------------------------------------------------------------- #
def _plot_regret_from_aggregate(
    agg_results: dict,
    *,
    num_episodes: int,
    max_steps: int,
    seeds: int,
    mode: str,
    reward_type: str,
    slip_prob: float = 0.0,
    detection: str = "td",
) -> None:
    use_var = mode.startswith("var_")
    x_mode = mode.replace("var_", "")

    plt.figure(figsize=(8, 6))

    # ---------- FIRST PASS: bands ----------
    if use_var:
        for ag, data in agg_results.items():
            y = data["mean_cum_regret"]
            x = np.arange(len(y)) if x_mode == "episodes" else data["mean_timesteps"]
            stderr = data["se_cum_regret"]
            plt.fill_between(
                x,
                y - stderr,
                y + stderr,
                alpha=0.25,
                edgecolor="none",
                zorder=1,          # all bands sit at the bottom layer
            )

    # ---------- SECOND PASS: lines ----------
    for ag, data in agg_results.items():
        y = data["mean_cum_regret"]
        x = np.arange(len(y)) if x_mode == "episodes" else data["mean_timesteps"]
        plt.plot(
            x,
            y,
            label=("see" if ag == "see_oracle" else ag),
            linewidth=2.2,
            zorder=3
        )

    # ---------- Title & labels ----------
    title = f"Cumulative Regret ({seeds}-Seed Average) - {reward_type.capitalize()} Reward"
    if slip_prob > 0:
        title += f" | Slip={slip_prob}"
    if detection != "td":
        title += f" | {detection.capitalize()} Detection"

    plt.xlabel("Episodes" if x_mode == "episodes" else "Timesteps")
    plt.ylabel("Cumulative Regret")
    plt.title(title)
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.yscale("log")
    savename = f"regret_{reward_type}"
    if slip_prob > 0:
        savename += f"_slip{slip_prob}"
    if detection != "td":
        savename += f"_{detection}"
    plt.savefig(f"{savename}.png", dpi=150, bbox_inches="tight")
    print(f"Saved {savename}.png")
    plt.close()

    # ---------- Exact Statistics (separate window) ----------
    fig, ax = plt.subplots(figsize=(6, len(agg_results)*0.6 + 1.5))
    ax.axis('off')

    # Build statistics table
    stats = []
    for ag, data in agg_results.items():
        y   = np.asarray(data["mean_cum_regret"], dtype=float)
        inc = np.asarray(data["mean_regret"], dtype=float)
        w   = 3
        inc_s = np.array([inc[max(0, i - w + 1): i + 1].mean() for i in range(len(inc))])
        peak = inc_s.max()
        conv_ep = None
        if peak <= 0:
            conv_ep = 0
        else:
            bar = 0.10 * peak          # converged: per-episode regret <=10% of its peak
            for i in range(len(inc_s)):
                if np.all(inc_s[i:] <= bar):
                    conv_ep = i
                    break
        conv_str = str(conv_ep) if conv_ep is not None else "--"
        stats.append([ag.replace('see_oracle', 'see').replace('_', ' ').title(), f"{y[-1]:.1f}", conv_str])

    cols = ["Agent", "Final Regret", "Episodes to Converge"]
    table = ax.table(
        cellText=stats,
        colLabels=cols,
        cellLoc="center",
        colLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)  # increase row height

    # Style header row
    for j in range(len(cols)):
        cell = table[(0, j)]
        cell.set_facecolor('#cccccc')
        cell.set_text_props(weight='bold', color='black')

    # Alternate row shading and add cell borders
    n_rows = len(stats)
    for i in range(1, n_rows+1):
        for j in range(len(cols)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f2f2f2')
            else:
                cell.set_facecolor('white')
            cell.set_edgecolor('black')

    # Draw table
    plt.tight_layout()
    plt.savefig(f"{savename}_stats.png", dpi=150, bbox_inches="tight")
    print(f"Saved {savename}_stats.png")
    plt.close()

# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #
def main():
    args = _parse_args()

    if args.plot_only:
        _plot_only(args)
    else:
        _run_experiments(args)


if __name__ == "__main__":
    main()
