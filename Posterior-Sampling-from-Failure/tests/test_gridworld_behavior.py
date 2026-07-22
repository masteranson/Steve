import contextlib
import io
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GRIDWORLD_DIR = ROOT / "GridWorld"
if str(GRIDWORLD_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDWORLD_DIR))

from GridWorld import GridWorld
from PSRL_Agents import PSRLStandard, StochasticGridWorld, expected_td
from ValueIteration import ValueIteration


def test_gridworld_rewards_walls_and_terminal_absorption():
    env = GridWorld(
        width=3,
        height=3,
        gamma=0.9,
        walls=[(1, 1)],
        origin_state=(2, 0),
        terminal_states=[(0, 2)],
        reward_type="euclidean",
    )

    assert len(env.states) == 8
    assert env.origin_state == (2, 0, 1)
    assert env.get_next_state_and_reward((2, 0, 1), "up") == (
        (1, 0, 1),
        -0.7905694150420948,
    )
    assert env.get_next_state_and_reward((2, 0, 1), "right") == (
        (2, 1, 1),
        -0.7905694150420948,
    )
    assert env.get_next_state_and_reward((2, 0, 1), "down") == ((2, 0, 1), -1.0)
    assert env.get_next_state_and_reward((2, 0, 1), "left") == ((2, 0, 1), -1.0)
    assert env.get_next_state_and_reward((0, 2, 1), "left") == ((0, 2, 1), 0)


def test_slip_transition_distribution_merges_wall_outcomes():
    env = GridWorld(
        width=3,
        height=3,
        gamma=0.9,
        walls=[(1, 1)],
        origin_state=(2, 0),
        terminal_states=[(0, 2)],
        reward_type="binary",
        slip_prob=0.2,
    )

    assert env.transition_probs((2, 0, 0), "up") == [
        (0.8, (1, 0, 0), 0.0),
        (0.1, (2, 0, 0), 0.0),
        (0.1, (2, 1, 0), 0.0),
    ]


def test_value_iteration_policy_and_expected_td_are_stable():
    env = GridWorld(
        width=3,
        height=3,
        gamma=0.9,
        walls=[(1, 1)],
        origin_state=(2, 0),
        terminal_states=[(0, 2)],
        reward_type="euclidean",
    )

    values, policy = ValueIteration(env).run()

    assert math.isclose(values[env.origin_state], -0.984343764491, abs_tol=1e-12)
    assert policy.get_action_probabilities(env.origin_state) == [0.5, 0.5, 0.0, 0.0]
    assert math.isclose(expected_td(env, values, env.origin_state, "up"), 0.0, abs_tol=1e-12)


def test_stochastic_gridworld_info_probability_shape():
    env = StochasticGridWorld(
        width=3,
        height=3,
        gamma=0.9,
        walls=[(1, 1)],
        origin_state=(2, 0),
        candidate_goals=[(0, 2), (2, 2)],
        true_goal=(0, 2),
        slip_prob=0.0,
    )

    assert len(env.states) == 16
    assert math.isclose(env.info_prob1(2, 0), 0.453261848015, abs_tol=1e-12)


def test_seeded_gridworld_agent_output_schema_and_values():
    np.random.seed(3)
    agent = PSRLStandard(
        width=3,
        height=3,
        gamma=0.9,
        candidate_goals=[(0, 2), (2, 2)],
        true_goal=(0, 2),
        walls=[(1, 1)],
        origin_state=(2, 0),
        num_episodes=2,
        max_steps_per_episode=3,
        reward_type="binary",
    )

    with contextlib.redirect_stdout(io.StringIO()):
        result = agent.run()

    assert result["episode_returns"] == [0.0, 0.0]
    assert result["cumulative_rewards"] == [0.0, 0.0]
    assert result["episodic_regrets"] == [0.0, 0.08999999999999997]
    assert result["timesteps"] == [3, 6]
    assert result["final_beta_params"] == {
        (0, 2): [1.0, 1.0],
        (2, 2): [1.0, 3.0],
    }
