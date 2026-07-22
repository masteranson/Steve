import numpy as np


class Policy:
    """
    Runtime-optimized policy:
      - stores best_action_mask[state_idx] as a uint16 bitmask
      - can still return probability vectors for compatibility
      - adds fast uniform sampling among set bits
    """

    def __init__(self, env, best_action_mask, num_actions: int):
        self.env = env
        self.best_action_mask = best_action_mask  # (num_states,) uint16
        self.num_actions = int(num_actions)

    def get_action_probabilities(self, state):
        s_idx = int(self.env.state_to_index(state))
        mask = int(self.best_action_mask[s_idx])

        if mask == 0:
            return np.full((self.num_actions,), 1.0 / self.num_actions, dtype=np.float64)

        probs = np.zeros((self.num_actions,), dtype=np.float64)
        cnt = 0
        for a in range(self.num_actions):
            if (mask >> a) & 1:
                cnt += 1
        if cnt == 0:
            probs[:] = 1.0 / self.num_actions
            return probs

        p = 1.0 / float(cnt)
        for a in range(self.num_actions):
            if (mask >> a) & 1:
                probs[a] = p
        return probs

    def sample_action_idx(self, state_idx: int, rng: np.random.Generator) -> int:
        """
        Uniformly sample among best actions encoded in the mask.
        Falls back to uniform over all actions if mask is empty.
        """
        mask = int(self.best_action_mask[int(state_idx)])
        if mask == 0:
            return int(rng.integers(0, self.num_actions))

        cnt = int(mask.bit_count())
        if cnt <= 0:
            return int(rng.integers(0, self.num_actions))

        k = int(rng.integers(0, cnt))
        m = mask
        while m:
            lsb = m & -m
            a = int(lsb.bit_length() - 1)
            if k == 0:
                return a
            k -= 1
            m ^= lsb

        return int(rng.integers(0, self.num_actions))
