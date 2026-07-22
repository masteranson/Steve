import numpy as np

try:
    from .Policy import Policy
    from .BitFlip import bits_to_idx, popcount_u16
except ImportError:
    from Policy import Policy
    from BitFlip import bits_to_idx, popcount_u16


class ValueIteration:
    """
    GridWorld-parity VI:
      - terminal states: V=0.0, no backups
      - uses env.transition_probs when present
      - policy is uniform among ties

    Vectorized for BitFlip-16 with 2 * 2^16 states.
    """

    def __init__(self, env, theta=1e-10, max_iterations=5000):
        self.env = env
        self.theta = float(theta)
        self.max_iterations = int(max_iterations)

        self.n_bits = int(env.n_bits)
        assert self.n_bits == 16

        self.num_bits_states = 1 << self.n_bits
        self.num_states = self.num_bits_states * 2
        self.num_actions = self.n_bits
        self.gamma = float(env.gamma)

        self.V = np.zeros((self.num_states,), dtype=np.float64)
        self.best_action_mask = np.zeros((self.num_states,), dtype=np.uint16)

        self._bits = np.arange(self.num_bits_states, dtype=np.uint16)
        self._action_masks = (1 << np.arange(self.num_actions, dtype=np.uint16)).astype(np.uint16)
        self._next_bits = (self._bits[None, :] ^ self._action_masks[:, None]).astype(np.uint16)

        goal_bits = getattr(env, "terminal_goal_bits", None)
        if goal_bits is None:
            raise ValueError("env must define terminal_goal_bits")

        # Keep this as a Python int; uint16 would overflow when adding 2^16.
        self._goal_bits_idx = int(bits_to_idx(goal_bits, self.n_bits))
        self._goal_bits_u16 = np.uint16(self._goal_bits_idx)

        self._term0 = int(self._goal_bits_idx)
        self._term1 = int(self._goal_bits_idx + (1 << self.n_bits))

        self._reward_type = getattr(env, "reward_type", "binary")
        self._task_reward_only = bool(getattr(env, "task_reward_only", False))

        if self._reward_type == "stochastic":
            self._temperature = float(getattr(env, "temperature", 1.0))

            cand_bits = [bits_to_idx(g, self.n_bits) for g in env.candidate_goals]
            self._cand_bits = np.asarray(cand_bits, dtype=np.uint16)
            self._true_bits = np.uint16(bits_to_idx(env.true_goal_bits, self.n_bits))
            self._true_idx = int(env._true_goal_cand_idx)

            dist_k = np.empty((len(self._cand_bits), self.num_bits_states), dtype=np.float64)
            for i, g in enumerate(self._cand_bits):
                dist_k[i, :] = np.vectorize(popcount_u16)(self._bits ^ np.uint16(g)).astype(np.float64)

            d = np.maximum(dist_k, 1.0)
            w = 1.0 / (d * d)

            temp = max(self._temperature, 1e-8)
            logits = w / temp
            logits = logits - logits.max(axis=0, keepdims=True)
            w_exp = np.exp(logits)
            soft = w_exp / w_exp.sum(axis=0, keepdims=True)
            self._p1_table = soft[self._true_idx].astype(np.float64)

            self._dist_true = np.vectorize(popcount_u16)(self._bits ^ self._true_bits).astype(np.float64)
        else:
            self._dist_goal = np.vectorize(popcount_u16)(self._bits ^ self._goal_bits_u16).astype(np.float64)

    def run(self):
        V = self.V
        gamma = self.gamma

        V[self._term0] = 0.0
        V[self._term1] = 0.0

        for _ in range(self.max_iterations):
            V_old = V.copy()

            V0 = V_old[: self.num_bits_states]
            V1 = V_old[self.num_bits_states :]

            if self._reward_type == "stochastic":
                Q0 = np.empty((self.num_actions, self.num_bits_states), dtype=np.float64)
                Q1 = np.empty((self.num_actions, self.num_bits_states), dtype=np.float64)

                true_bits_int = int(self._true_bits)

                for a in range(self.num_actions):
                    nb = self._next_bits[a, :].astype(np.int64)
                    p1 = self._p1_table[nb]
                    expV = (1.0 - p1) * V0[nb] + p1 * V1[nb]

                    r0 = (nb == true_bits_int).astype(np.float64)
                    if self._task_reward_only:
                        r1 = r0
                    else:
                        dist = self._dist_true[nb]
                        r1 = r0 - dist / float(self.n_bits)

                    Q0[a, :] = r0 + gamma * expV
                    Q1[a, :] = r1 + gamma * expV

                best0 = Q0.max(axis=0)
                best1 = Q1.max(axis=0)

                V[: self.num_bits_states] = best0
                V[self.num_bits_states :] = best1

                V[self._term0] = 0.0
                V[self._term1] = 0.0

                eq0 = (Q0 == best0[None, :])
                eq1 = (Q1 == best1[None, :])

                mask0 = np.zeros((self.num_bits_states,), dtype=np.uint16)
                mask1 = np.zeros((self.num_bits_states,), dtype=np.uint16)
                for a in range(self.num_actions):
                    mask0 |= (eq0[a, :].astype(np.uint16) << np.uint16(a))
                    mask1 |= (eq1[a, :].astype(np.uint16) << np.uint16(a))

                self.best_action_mask[: self.num_bits_states] = mask0
                self.best_action_mask[self.num_bits_states :] = mask1

            else:
                Q0 = np.empty((self.num_actions, self.num_bits_states), dtype=np.float64)
                Q1 = np.empty((self.num_actions, self.num_bits_states), dtype=np.float64)

                goal_int = int(self._goal_bits_idx)

                for a in range(self.num_actions):
                    nb = self._next_bits[a, :].astype(np.int64)

                    r0 = (nb == goal_int).astype(np.float64)
                    if self._task_reward_only:
                        r1 = (nb == goal_int).astype(np.float64)
                    else:
                        dist = self._dist_goal[nb]
                        r1 = (nb == goal_int).astype(np.float64) - dist / float(self.n_bits)

                    Q0[a, :] = r0 + gamma * V0[nb]
                    Q1[a, :] = r1 + gamma * V1[nb]

                best0 = Q0.max(axis=0)
                best1 = Q1.max(axis=0)

                V[: self.num_bits_states] = best0
                V[self.num_bits_states :] = best1

                V[self._term0] = 0.0
                V[self._term1] = 0.0

                eq0 = (Q0 == best0[None, :])
                eq1 = (Q1 == best1[None, :])

                mask0 = np.zeros((self.num_bits_states,), dtype=np.uint16)
                mask1 = np.zeros((self.num_bits_states,), dtype=np.uint16)
                for a in range(self.num_actions):
                    mask0 |= (eq0[a, :].astype(np.uint16) << np.uint16(a))
                    mask1 |= (eq1[a, :].astype(np.uint16) << np.uint16(a))

                self.best_action_mask[: self.num_bits_states] = mask0
                self.best_action_mask[self.num_bits_states :] = mask1

            if float(np.max(np.abs(V - V_old))) < self.theta:
                break

        self.best_action_mask[self._term0] = np.uint16(0)
        self.best_action_mask[self._term1] = np.uint16(0)

        pi = Policy(self.env, self.best_action_mask, self.num_actions)
        return V, pi
