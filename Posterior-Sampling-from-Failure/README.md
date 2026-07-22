# Posterior Sampling from Failure

This repository contains the tabular research code for experiments on posterior
sampling from failure signals. It includes two domains:

- `GridWorld/`: four-room GridWorld experiments with sparse, Euclidean, and
  stochastic information-bit rewards.
- `BitFlip/`: optimized BitFlip-16 experiments with reward semantics matched to
  the GridWorld variants.

The code is intentionally lightweight. It is meant to reproduce and inspect the
paper experiments, not to serve as a packaged library.

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows
python -m pip install -r requirements.txt
```

`pygame` is used only for GridWorld visualizations. The core numerical runs use
`numpy` and `matplotlib`.

## Repository Layout

- `GridWorld/GridWorld.py`: GridWorld environment and reward dynamics.
- `GridWorld/PSRL_Agents.py`: GridWorld PSRL variants and baselines.
- `GridWorld/Run_PSRL.py`: GridWorld experiment runner and plotting entrypoint.
- `BitFlip/BitFlip.py`: BitFlip-16 environment and bit-index helpers.
- `BitFlip/PSRL_Agents.py`: optimized BitFlip PSRL variants.
- `BitFlip/Run_PSRL.py`: BitFlip experiment runner and plotting entrypoint.
- `tests/`: fast characterization tests that protect current behavior.

## Reproducing the Paper

Deterministic GridWorld, all eight agents (paper Figure 1). Run once per reward type:

```bash
python -m GridWorld.Run_PSRL --env 15_candidates \
  --agents standard early_stopping recovery weighted_recovery \
           weighted_graph weighted_directional least see \
  --reward-type binary --num-seeds 50 --num-episodes 50 --max-steps 100
```

Repeat with `--reward-type euclidean` and `--reward-type stochastic`.

Stochastic transitions with the realizable GAE detector (paper Figure 2):

```bash
python -m GridWorld.Run_PSRL --env 15_candidates \
  --agents standard early_stopping recovery \
  --reward-type binary --slip-prob 0.1 \
  --detection advantage --gae-lambda 0.95 \
  --num-seeds 50 --num-episodes 50 --max-steps 100
```

Repeat for the other reward types, and for the Appendix B sweep use
`--slip-prob 0.05 0.10 0.15 0.20`. The Appendix B oracle expected-TD runs use
`--detection td --slip-prob 0.2`.

BitFlip-16 (paper Figure 3):

```bash
python -m BitFlip.Run_PSRL --reward_type binary --agent recovery \
  --seeds 50 --episodes 50 --steps 64
```

Repeat with `--reward_type dense` / `stochastic` and `--agent psrl` / `early`.

**Flag-to-paper mapping.** `--detection advantage` is the realizable GAE
accumulator; `--detection td` is the oracle expected-TD signal. BitFlip's
`--reward_type dense` is the dense Hamming-distance regime.

## Outputs

GridWorld runs write per-seed JSON caches under:

```text
visualizations/<condition>/<env>/<agent>/<seed>/cache.json
```

BitFlip runs write NumPy arrays and configs under:

```text
visualizations_bitflip/<reward_type>/<agent>/seed_<seed>/
```

Generated visualization folders, cache directories, and PNGs are ignored by Git.

## Verification

Run the fast behavior checks before and after any cleanup:

```bash
python -m compileall -q BitFlip GridWorld tests
python -m pytest -q
```

The tests characterize the current transition, reward, value-iteration, and
seeded GridWorld-agent behavior. Full BitFlip agent construction is deliberately
kept out of the default test path because it performs value iteration over the
full `2 * 2^16` state space.
