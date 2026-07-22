import numpy as np


def bits_to_idx(bits, n_bits=16) -> int:
    idx = 0
    for i in range(n_bits):
        idx |= (int(bits[i]) & 1) << i
    return int(idx)


def idx_to_bits(idx: int, n_bits=16):
    return tuple((int(idx) >> i) & 1 for i in range(n_bits))


def popcount_u16(x: int) -> int:
    return int(int(x & 0xFFFF).bit_count())


def hamming_idx(a_bits_idx: int, b_bits_idx: int) -> int:
    return popcount_u16(int(a_bits_idx) ^ int(b_bits_idx))


def hamming_bits(bits_a, bits_b, n_bits=16) -> int:
    return hamming_idx(bits_to_idx(bits_a, n_bits), bits_to_idx(bits_b, n_bits))


def flip_bits_idx(bits_idx: int, action: int) -> int:
    return int(bits_idx) ^ (1 << int(action))


def pack_state_idx(bits_idx: int, info_bit: int, n_bits=16) -> int:
    return int(bits_idx) + (int(info_bit) << int(n_bits))


def unpack_state_idx(state_idx: int, n_bits=16):
    state_idx = int(state_idx)
    bits_mask = (1 << int(n_bits)) - 1
    bits_idx = state_idx & bits_mask
    info_bit = (state_idx >> int(n_bits)) & 1
    return int(bits_idx), int(info_bit)


class BitFlipEnv:
    """
    GridWorld-parity reward semantics.

    State: (b0..b_{n-1}, info_bit)
    Action: flip one bit index
    Terminal: goal bits with either info_bit
    Terminal transition: stay put, reward 0

    Reward uses CURRENT info bit and NEXT bits:
      - bit==0: r = 1 if next_bits == goal else 0
      - bit==1: r = 1{next_bits==goal} - dist(next_bits, goal)/n_bits
    """

    def __init__(
        self,
        *,
        n_bits=16,
        gamma=0.95,
        origin_state=None,
        terminal_goal_bits=None,
        reward_type="binary",
        seed=0,
        task_reward_only=False,
    ):
        assert int(n_bits) == 16, "This optimized implementation assumes n_bits=16."
        assert reward_type in ("binary", "dense")
        self.n_bits = int(n_bits)
        self.gamma = float(gamma)
        self.reward_type = reward_type
        self.task_reward_only = bool(task_reward_only)
        self.rng = np.random.default_rng(int(seed))

        if terminal_goal_bits is None:
            terminal_goal_bits = tuple(self.rng.integers(0, 2, size=(self.n_bits,), dtype=np.int8).tolist())
        self.terminal_goal_bits = tuple(int(x) for x in terminal_goal_bits)
        self.goal_bits_idx = bits_to_idx(self.terminal_goal_bits, self.n_bits)

        self.actions = list(range(self.n_bits))

        if origin_state is None:
            origin_bits = tuple(self.rng.integers(0, 2, size=(self.n_bits,), dtype=np.int8).tolist())
            info_bit = 1 if reward_type == "dense" else 0
            origin_state = origin_bits + (info_bit,)
        else:
            origin_state = tuple(int(x) for x in origin_state)
            if len(origin_state) == self.n_bits:
                info_bit = 1 if reward_type == "dense" else 0
                origin_state = origin_state + (info_bit,)
            assert len(origin_state) == self.n_bits + 1

        self.origin_state = origin_state
        self.state = self.origin_state

        self.term0_idx = pack_state_idx(self.goal_bits_idx, 0, self.n_bits)
        self.term1_idx = pack_state_idx(self.goal_bits_idx, 1, self.n_bits)

    def reset(self):
        self.state = self.origin_state
        return self.state

    def is_terminal(self, state):
        if isinstance(state, int):
            bits_idx, _ = unpack_state_idx(state, self.n_bits)
            return bits_idx == self.goal_bits_idx
        return tuple(state)[: self.n_bits] == self.terminal_goal_bits

    def _flip(self, state, action):
        bits = list(state[: self.n_bits])
        bits[int(action)] ^= 1
        return tuple(bits) + (int(state[self.n_bits]),)

    def get_next_state_and_reward(self, state, action):
        if self.is_terminal(state):
            return state, 0.0

        base_next = self._flip(state, action)
        next_bits = base_next[: self.n_bits]

        curr_bit = int(state[self.n_bits])
        goal_bonus = 1.0 if next_bits == self.terminal_goal_bits else 0.0
        if curr_bit == 1 and not self.task_reward_only:
            dist = hamming_bits(next_bits, self.terminal_goal_bits, self.n_bits)
            reward = goal_bonus - float(dist) / float(self.n_bits)
        else:
            reward = goal_bonus

        return base_next, float(reward)

    def transition_probs(self, state, action):
        ns, r = self.get_next_state_and_reward(state, action)
        return [(1.0, ns, float(r))]

    def state_to_index(self, state):
        bits_idx = bits_to_idx(state[: self.n_bits], self.n_bits)
        b = int(state[self.n_bits]) & 1
        return pack_state_idx(bits_idx, b, self.n_bits)

    def index_to_state(self, idx):
        bits_idx, b = unpack_state_idx(idx, self.n_bits)
        bits = idx_to_bits(bits_idx, self.n_bits)
        return tuple(bits) + (int(b),)


class PSRLStandardBitFlipEnv(BitFlipEnv):
    """
    Planning env analogue:
      - goal is the candidate goal
      - if bit==0: reward is 1 upon next_bits == candidate_goal
      - if bit==1: reward is dense formula relative to candidate_goal
    """

    def __init__(
        self,
        *,
        n_bits=16,
        gamma=0.95,
        origin_state=None,
        candidate_goal_bits=None,
        reward_type="binary",
        seed=0,
        task_reward_only=False,
    ):
        assert candidate_goal_bits is not None
        super().__init__(
            n_bits=n_bits,
            gamma=gamma,
            origin_state=origin_state,
            terminal_goal_bits=candidate_goal_bits,
            reward_type=reward_type,
            seed=seed,
            task_reward_only=task_reward_only,
        )

    def get_next_state_and_reward(self, state, action):
        if self.is_terminal(state):
            return state, 0.0

        base_next = self._flip(state, action)
        next_bits = base_next[: self.n_bits]

        curr_bit = int(state[self.n_bits])
        goal_bonus = 1.0 if next_bits == self.terminal_goal_bits else 0.0
        if curr_bit == 1 and not self.task_reward_only:
            dist = hamming_bits(next_bits, self.terminal_goal_bits, self.n_bits)
            reward = goal_bonus - float(dist) / float(self.n_bits)
            return base_next, float(reward)

        return base_next, goal_bonus


class StochasticBitFlipEnv(BitFlipEnv):
    """
    GridWorld stochastic analogue:

      1) take action -> base_next bits
      2) reward uses current bit and base_next bits
      3) sample next bit with weights 1/d^2 softmaxed over candidate_goals
      4) next state is (base_next_bits, b_new)

    transition_probs enumerates b_new with identical reward.
    """

    def __init__(
        self,
        *,
        n_bits=16,
        gamma=0.95,
        origin_state=None,
        candidate_goals=None,
        true_goal_bits=None,
        temperature=1.0,
        seed=0,
        task_reward_only=False,
    ):
        assert candidate_goals is not None
        assert true_goal_bits is not None

        self.candidate_goals = [tuple(int(x) for x in g) for g in candidate_goals]
        self.true_goal_bits = tuple(int(x) for x in true_goal_bits)
        self.temperature = float(temperature)

        super().__init__(
            n_bits=n_bits,
            gamma=gamma,
            origin_state=origin_state,
            terminal_goal_bits=self.true_goal_bits,
            reward_type="binary",
            seed=seed,
            task_reward_only=task_reward_only,
        )
        self.reward_type = "stochastic"

        try:
            self._true_goal_cand_idx = self.candidate_goals.index(self.true_goal_bits)
        except ValueError:
            raise ValueError("true_goal_bits must appear in candidate_goals")

    def info_prob1(self, bits):
        w = []
        for g in self.candidate_goals:
            d = max(hamming_bits(bits, g, self.n_bits), 1)
            w.append(1.0 / (d * d))
        w = np.asarray(w, dtype=float)

        temp = max(self.temperature, 1e-8)
        logits = w / temp
        logits = logits - logits.max()
        w_exp = np.exp(logits)
        soft = w_exp / w_exp.sum()
        return float(soft[self._true_goal_cand_idx])

    def get_next_state_and_reward(self, state, action):
        if self.is_terminal(state):
            return state, 0.0

        base_next = self._flip(state, action)
        next_bits = base_next[: self.n_bits]

        curr_bit = int(state[self.n_bits])
        goal_bonus = 1.0 if next_bits == self.true_goal_bits else 0.0
        if curr_bit == 1 and not self.task_reward_only:
            dist = hamming_bits(next_bits, self.true_goal_bits, self.n_bits)
            reward = goal_bonus - float(dist) / float(self.n_bits)
        else:
            reward = goal_bonus

        p1 = self.info_prob1(next_bits)
        b_new = int(self.rng.binomial(1, p1))
        next_state = tuple(next_bits) + (b_new,)
        return next_state, float(reward)

    def transition_probs(self, state, action):
        if self.is_terminal(state):
            return [(1.0, state, 0.0)]

        base_next = self._flip(state, action)
        next_bits = base_next[: self.n_bits]

        curr_bit = int(state[self.n_bits])
        goal_bonus = 1.0 if next_bits == self.true_goal_bits else 0.0
        if curr_bit == 1 and not self.task_reward_only:
            dist = hamming_bits(next_bits, self.true_goal_bits, self.n_bits)
            reward = goal_bonus - float(dist) / float(self.n_bits)
        else:
            reward = goal_bonus

        p1 = self.info_prob1(next_bits)
        return [
            (1.0 - p1, tuple(next_bits) + (0,), float(reward)),
            (p1, tuple(next_bits) + (1,), float(reward)),
        ]
