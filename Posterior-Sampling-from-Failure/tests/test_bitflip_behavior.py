import math

from BitFlip.BitFlip import (
    BitFlipEnv,
    StochasticBitFlipEnv,
    bits_to_idx,
    idx_to_bits,
    pack_state_idx,
    unpack_state_idx,
)


def test_bit_index_roundtrip_and_state_packing():
    assert bits_to_idx((1, 0, 1, 0), 4) == 5
    assert idx_to_bits(5, 4) == (1, 0, 1, 0)
    assert pack_state_idx(5, 1, 4) == 21
    assert unpack_state_idx(21, 4) == (5, 1)


def test_dense_reward_uses_current_info_bit_and_next_bits():
    goal = (1,) + (0,) * 15
    origin = (0,) * 16 + (1,)
    env = BitFlipEnv(
        n_bits=16,
        gamma=0.95,
        origin_state=origin,
        terminal_goal_bits=goal,
        reward_type="dense",
        seed=123,
    )

    next_state, reward = env.get_next_state_and_reward(env.origin_state, 0)

    assert next_state == goal + (1,)
    assert reward == 1.0


def test_terminal_state_is_absorbing_with_zero_reward():
    goal = (1,) + (0,) * 15
    env = BitFlipEnv(
        n_bits=16,
        origin_state=(0,) * 17,
        terminal_goal_bits=goal,
        reward_type="binary",
        seed=123,
    )
    terminal_state = env.index_to_state(env.term1_idx)

    next_state, reward = env.get_next_state_and_reward(terminal_state, 3)

    assert next_state == terminal_state
    assert reward == 0.0


def test_stochastic_bit_probability_matches_current_formula():
    candidate_goals = [
        (1,) + (0,) * 15,
        (0, 1) + (0,) * 14,
        (0, 0, 1) + (0,) * 13,
        (0, 0, 0, 1) + (0,) * 12,
    ]
    env = StochasticBitFlipEnv(
        n_bits=16,
        origin_state=(0,) * 17,
        candidate_goals=candidate_goals,
        true_goal_bits=candidate_goals[0],
        temperature=1.0,
        seed=123,
    )

    assert math.isclose(env.info_prob1((0,) * 16), 0.25, abs_tol=1e-12)

