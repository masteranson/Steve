import os
import numpy as np
from abc import ABC, abstractmethod

try:
    from .BitFlip import (
        BitFlipEnv,
        PSRLStandardBitFlipEnv,
        StochasticBitFlipEnv,
        bits_to_idx,
        idx_to_bits,
        popcount_u16,
        flip_bits_idx,
        pack_state_idx,
        unpack_state_idx,
    )
    from .ValueIteration import ValueIteration
    from .visualization_utils import save_episode_trace
except ImportError:
    from BitFlip import (
        BitFlipEnv,
        PSRLStandardBitFlipEnv,
        StochasticBitFlipEnv,
        bits_to_idx,
        idx_to_bits,
        popcount_u16,
        flip_bits_idx,
        pack_state_idx,
        unpack_state_idx,
    )
    from ValueIteration import ValueIteration
    from visualization_utils import save_episode_trace


def _rand_bits(rng, n_bits: int):
    return tuple(int(x) for x in rng.integers(0, 2, size=(int(n_bits),), dtype=np.int8).tolist())


def _switches_to_obj_array(switches_all, n_episodes: int):
    arr = np.empty((int(n_episodes),), dtype=object)
    for i in range(int(n_episodes)):
        arr[i] = switches_all[i]
    return arr


def _build_dist_table_to_goal(goal_bits_idx: int, n_bits: int = 16):
    bits = np.arange(1 << int(n_bits), dtype=np.uint16)
    goal = np.uint16(int(goal_bits_idx))
    return np.vectorize(popcount_u16)(bits ^ goal).astype(np.float64)


def _build_p1_table(candidate_goal_bits_idxs, true_goal_idx_in_list: int, n_bits: int = 16, temperature: float = 1.0):
    cand = [np.uint16(int(x)) for x in candidate_goal_bits_idxs]
    bits = np.arange(1 << int(n_bits), dtype=np.uint16)

    dist = np.empty((len(cand), bits.shape[0]), dtype=np.float64)
    for i, g in enumerate(cand):
        dist[i, :] = np.vectorize(popcount_u16)(bits ^ g).astype(np.float64)

    d = np.maximum(dist, 1.0)
    w = 1.0 / (d * d)

    temp = max(float(temperature), 1e-8)
    logits = w / temp
    logits = logits - logits.max(axis=0, keepdims=True)
    w_exp = np.exp(logits)
    soft = w_exp / w_exp.sum(axis=0, keepdims=True)
    return soft[int(true_goal_idx_in_list)].astype(np.float64)


class PSRLAgentBase(ABC):
    """
    BitFlip implementations are intended to match GridWorld functionality exactly,
    with the ONLY difference being the distance metric (Hamming for BitFlip-16).
    """

    def __init__(
        self,
        *,
        n_bits=16,
        gamma=0.95,
        reward_type="binary",
        num_episodes=50,
        max_steps_per_episode=64,
        prior_alpha=1.0,
        prior_beta=1.0,
        seed=0,
        out_dir=None,
        td_threshold=0.0,
        min_steps=1,
        tau=1.0,
        temperature=1.0,
        candidate_goals=None,
        true_goal=None,
        detection="td",
        gae_lambda=0.95,
    ):
        assert reward_type in ("binary", "dense", "stochastic")
        self.n_bits = int(n_bits)
        assert self.n_bits == 16, "This optimized implementation assumes n_bits=16."
        self.gamma = float(gamma)
        self.reward_type = str(reward_type)
        self.num_episodes = int(num_episodes)
        self.max_steps = int(max_steps_per_episode)

        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.td_threshold = float(td_threshold)
        self.min_steps = int(min_steps)
        self.tau = float(tau)
        self.temperature = float(temperature)
        self.detection = str(detection)
        self.gae_lambda = float(gae_lambda)
        if self.detection not in ("td", "advantage"):
            raise ValueError("detection must be 'td' or 'advantage'")

        if true_goal is None:
            true_goal = _rand_bits(self.rng, self.n_bits)
        self.true_goal = tuple(int(x) for x in true_goal)
        self.true_goal_bits_idx = int(bits_to_idx(self.true_goal, self.n_bits))

        if candidate_goals is None:
            cands = [self.true_goal]
            while len(cands) < 4:
                g = _rand_bits(self.rng, self.n_bits)
                if g not in cands:
                    cands.append(g)
            candidate_goals = cands

        self.candidate_goals = [tuple(int(x) for x in g) for g in candidate_goals]
        if self.true_goal not in self.candidate_goals:
            self.candidate_goals = [self.true_goal] + self.candidate_goals

        self.candidate_goal_bits_idxs = [int(bits_to_idx(g, self.n_bits)) for g in self.candidate_goals]

        origin_bits = _rand_bits(self.rng, self.n_bits)
        origin_bit = 1 if reward_type == "dense" else 0
        self.origin_bits_idx = int(bits_to_idx(origin_bits, self.n_bits))
        self.origin_bit = int(origin_bit)
        self.origin_state = origin_bits + (int(origin_bit),)

        self.beta_params = {
            g: [float(prior_alpha), float(prior_beta)]
            for g in self.candidate_goals
        }

        if reward_type == "stochastic":
            self.true_env = StochasticBitFlipEnv(
                n_bits=self.n_bits,
                gamma=self.gamma,
                origin_state=origin_bits + (0,),
                candidate_goals=self.candidate_goals,
                true_goal_bits=self.true_goal,
                temperature=self.temperature,
                seed=self.seed,
            )
            self._true_is_stochastic = True
            self._true_p1_table = _build_p1_table(
                self.candidate_goal_bits_idxs,
                self.candidate_goals.index(self.true_goal),
                self.n_bits,
                temperature=self.temperature,
            )
        else:
            env_reward_type = "dense" if reward_type == "dense" else "binary"
            self.true_env = BitFlipEnv(
                n_bits=self.n_bits,
                gamma=self.gamma,
                origin_state=self.origin_state,
                terminal_goal_bits=self.true_goal,
                reward_type=env_reward_type,
                seed=self.seed,
            )
            self._true_is_stochastic = False
            self._true_p1_table = None

        self._true_dist_goal = _build_dist_table_to_goal(self.true_goal_bits_idx, self.n_bits)

        self.num_bits_states = 1 << self.n_bits
        self.num_states = self.num_bits_states * 2
        self.num_actions = self.n_bits

        # Match the original experimental metric: regret is computed under
        # the same reward regime used by the environment.
        self.eval_env = self.true_env

        self.V_star, _ = ValueIteration(self.eval_env).run()
        self.Q_star = np.zeros((self.num_states, self.num_actions), dtype=np.float64)
        self._build_true_Q_star()

        self.out_dir = out_dir
        self._vi_cache = {}

    def sample_candidate(self):
        samples = np.array([
            self.rng.beta(self.beta_params[g][0], self.beta_params[g][1])
            for g in self.candidate_goals
        ], dtype=np.float64)
        idx = int(np.argmax(samples))
        return self.candidate_goals[idx], float(samples[idx])

    def _observed_false_candidate(self, bits_idx):
        bits_idx = int(bits_idx)
        if bits_idx == self.true_goal_bits_idx:
            return None

        for candidate, candidate_bits_idx in zip(self.candidate_goals, self.candidate_goal_bits_idxs):
            if bits_idx == int(candidate_bits_idx):
                return candidate

        return None

    def update_posterior(self, candidate_goal, success: bool, bit=None):
        candidate_goal = tuple(candidate_goal)
        if candidate_goal not in self.beta_params:
            raise ValueError(f"unknown candidate goal: {candidate_goal}")

        if success:
            self.beta_params[candidate_goal][0] += 1.0
        else:
            self.beta_params[candidate_goal][1] += 1.0

    def plan_policy(self, candidate_goal):
        candidate_goal = tuple(int(x) for x in candidate_goal)
        key = (self.reward_type, candidate_goal)
        cached = self._vi_cache.get(key)
        if cached is not None:
            return cached

        if self.reward_type == "stochastic":
            env = StochasticBitFlipEnv(
                n_bits=self.n_bits,
                gamma=self.gamma,
                origin_state=idx_to_bits(self.origin_bits_idx, self.n_bits) + (0,),
                candidate_goals=self.candidate_goals,
                true_goal_bits=candidate_goal,
                temperature=self.temperature,
                seed=self.seed,
            )
        else:
            env = PSRLStandardBitFlipEnv(
                n_bits=self.n_bits,
                gamma=self.gamma,
                origin_state=idx_to_bits(self.origin_bits_idx, self.n_bits) + (self.origin_bit,),
                candidate_goal_bits=candidate_goal,
                reward_type=("dense" if self.reward_type == "dense" else "binary"),
                seed=self.seed,
            )

        V, pi = ValueIteration(env).run()
        self._vi_cache[key] = (env, V, pi)
        return self._vi_cache[key]

    def _true_reward_for_next_bits(self, next_bits_idx: int, curr_info_bit: int):
        goal = int(self.true_goal_bits_idx)
        if int(curr_info_bit) == 1:
            return (1.0 if int(next_bits_idx) == goal else 0.0) - float(self._true_dist_goal[int(next_bits_idx)]) / float(self.n_bits)
        return 1.0 if int(next_bits_idx) == goal else 0.0

    def _true_step_idx(self, state_idx: int, action: int):
        bits_idx, info_bit = unpack_state_idx(int(state_idx), self.n_bits)
        if int(bits_idx) == int(self.true_goal_bits_idx):
            return int(state_idx), 0.0

        next_bits_idx = flip_bits_idx(bits_idx, int(action))
        r = self._true_reward_for_next_bits(next_bits_idx, info_bit)

        if self._true_is_stochastic:
            p1 = float(self._true_p1_table[int(next_bits_idx)])
            next_info = int(self.rng.binomial(1, p1))
        else:
            next_info = int(info_bit)

        ns_idx = pack_state_idx(next_bits_idx, next_info, self.n_bits)
        return int(ns_idx), float(r)

    def _expected_td_true(self, V: np.ndarray, state_idx: int, action: int):
        bits_idx, info_bit = unpack_state_idx(int(state_idx), self.n_bits)
        if int(bits_idx) == int(self.true_goal_bits_idx):
            return 0.0

        next_bits_idx = flip_bits_idx(bits_idx, int(action))
        r = self._true_reward_for_next_bits(next_bits_idx, info_bit)

        if self._true_is_stochastic:
            p1 = float(self._true_p1_table[int(next_bits_idx)])
            v0 = float(V[pack_state_idx(next_bits_idx, 0, self.n_bits)])
            v1 = float(V[pack_state_idx(next_bits_idx, 1, self.n_bits)])
            expV = (1.0 - p1) * v0 + p1 * v1
            return float(r) + self.gamma * expV - float(V[int(state_idx)])

        ns_idx = pack_state_idx(next_bits_idx, info_bit, self.n_bits)
        return float(r) + self.gamma * float(V[int(ns_idx)]) - float(V[int(state_idx)])

    def _detection_signal(self, V: np.ndarray, state_idx: int, action: int,
                          ns_idx: int, reward: float, A_prev: float):
        if self.detection == "td":
            return float(self._expected_td_true(V, state_idx, action)), A_prev

        delta = float(reward) + self.gamma * float(V[int(ns_idx)]) - float(V[int(state_idx)])
        A_new = delta + self.gamma * self.gae_lambda * float(A_prev)
        return float(A_new), float(A_new)

    def _build_true_Q_star(self):
        for state_idx in range(self.num_states):
            state = self.eval_env.index_to_state(state_idx)
            for a in range(self.num_actions):
                q = 0.0
                for p, ns, r in self.eval_env.transition_probs(state, a):
                    ns_idx = self.eval_env.state_to_index(ns)
                    q += float(p) * (float(r) + self.gamma * float(self.V_star[ns_idx]))
                self.Q_star[state_idx, a] = q

    @abstractmethod
    def run(self):
        pass


class PSRLStandard(PSRLAgentBase):
    def run(self):
        returns = np.zeros((self.num_episodes,), dtype=np.float64)
        regrets = np.zeros((self.num_episodes,), dtype=np.float64)
        switches_all = [[] for _ in range(self.num_episodes)]

        goal_bits_idx = int(self.true_goal_bits_idx)
        n_bits = self.n_bits
        max_steps = self.max_steps
        for ep in range(self.num_episodes):
            A_acc = 0.0
            cand, samp = self.sample_candidate()
            _, V, pi = self.plan_policy(cand)

            bits_idx = int(self.origin_bits_idx)
            info_bit = int(self.origin_bit if self.reward_type != "stochastic" else 0)
            state_idx = pack_state_idx(bits_idx, info_bit, n_bits)

            total = 0.0
            reg = 0.0
            observed_false_candidates = []
            observed_false_set = set()

            trace = {
                "states": [],
                "actions": [],
                "rewards": [],
                "td": [],
                "switches": [],
                "success": False,
                "goal": list(self.true_goal),
            }

            for _ in range(max_steps):
                if bits_idx == goal_bits_idx:
                    break

                a = int(pi.sample_action_idx(state_idx, self.rng))
                ns_idx, r_true = self._true_step_idx(state_idx, a)
                td, A_acc = self._detection_signal(V, state_idx, a, ns_idx, r_true, A_acc)
                total += float(r_true)

                reg += float(np.max(self.Q_star[int(state_idx)])) - float(self.Q_star[int(state_idx), a])

                if self.out_dir is not None:
                    bits_i, bit_i = unpack_state_idx(int(state_idx), n_bits)
                    s_tuple = idx_to_bits(bits_i, n_bits) + (int(bit_i),)
                    trace["states"].append(list(s_tuple))
                    trace["actions"].append(int(a))
                    trace["rewards"].append(float(r_true))
                    trace["td"].append(float(td))

                state_idx = int(ns_idx)
                bits_idx, info_bit = unpack_state_idx(state_idx, n_bits)

                false_cand = self._observed_false_candidate(bits_idx)
                if false_cand is not None and false_cand not in observed_false_set:
                    observed_false_candidates.append((false_cand, int(info_bit)))
                    observed_false_set.add(false_cand)

            success = (bits_idx == goal_bits_idx)
            trace["success"] = bool(success)
            if success:
                self.update_posterior(self.true_goal, True, bit=int(info_bit))
            else:
                for false_cand, bit in observed_false_candidates:
                    self.update_posterior(false_cand, False, bit=bit)

            returns[ep] = total
            regrets[ep] = reg

            if self.out_dir is not None:
                save_episode_trace(os.path.join(self.out_dir, "traces"), ep, trace)

        return returns, regrets, _switches_to_obj_array(switches_all, self.num_episodes)


class EarlyStopping(PSRLAgentBase):
    def run(self):
        returns = np.zeros((self.num_episodes,), dtype=np.float64)
        regrets = np.zeros((self.num_episodes,), dtype=np.float64)
        switches_all = [[] for _ in range(self.num_episodes)]

        goal_bits_idx = int(self.true_goal_bits_idx)
        n_bits = self.n_bits
        max_steps = self.max_steps
        for ep in range(self.num_episodes):
            A_acc = 0.0
            cand, samp = self.sample_candidate()
            _, V, pi = self.plan_policy(cand)

            bits_idx = int(self.origin_bits_idx)
            info_bit = int(self.origin_bit if self.reward_type != "stochastic" else 0)
            state_idx = pack_state_idx(bits_idx, info_bit, n_bits)

            total = 0.0
            reg = 0.0
            observed_false_candidates = []
            observed_false_set = set()

            trace = {
                "states": [],
                "actions": [],
                "rewards": [],
                "td": [],
                "switches": [],
                "early_stop_idx": None,
                "success": False,
                "goal": list(self.true_goal),
            }

            reached = False
            failure_recorded = False
            for t in range(max_steps):
                if bits_idx == goal_bits_idx:
                    reached = True
                    break

                a = int(pi.sample_action_idx(state_idx, self.rng))
                ns_idx, r_true = self._true_step_idx(state_idx, a)
                td, A_acc = self._detection_signal(V, state_idx, a, ns_idx, r_true, A_acc)
                total += float(r_true)

                reg += float(np.max(self.Q_star[int(state_idx)])) - float(self.Q_star[int(state_idx), a])

                if self.out_dir is not None:
                    bits_i, bit_i = unpack_state_idx(int(state_idx), n_bits)
                    s_tuple = idx_to_bits(bits_i, n_bits) + (int(bit_i),)
                    trace["states"].append(list(s_tuple))
                    trace["actions"].append(int(a))
                    trace["rewards"].append(float(r_true))
                    trace["td"].append(float(td))

                state_idx = int(ns_idx)
                bits_idx, info_bit = unpack_state_idx(state_idx, n_bits)

                false_cand = self._observed_false_candidate(bits_idx)
                if false_cand is not None and false_cand not in observed_false_set:
                    observed_false_candidates.append((false_cand, int(info_bit)))
                    observed_false_set.add(false_cand)

                if bits_idx == goal_bits_idx:
                    reached = True
                    break

                if (t + 1) >= self.min_steps and td < self.td_threshold:
                    trace["early_stop_idx"] = int(t)
                    self.update_posterior(cand, False, bit=int(info_bit))
                    failure_recorded = True
                    break
            
            if reached:
                self.update_posterior(self.true_goal, True, bit=int(info_bit))
            elif not failure_recorded:
                for false_cand, bit in observed_false_candidates:
                    self.update_posterior(false_cand, False, bit=bit)

            trace["success"] = bool(reached)

            returns[ep] = total
            regrets[ep] = reg

            if self.out_dir is not None:
                save_episode_trace(os.path.join(self.out_dir, "traces"), ep, trace)

        return returns, regrets, _switches_to_obj_array(switches_all, self.num_episodes)


class Recovery(PSRLAgentBase):
    def run(self):
        returns = np.zeros((self.num_episodes,), dtype=np.float64)
        regrets = np.zeros((self.num_episodes,), dtype=np.float64)
        switches_all = [[] for _ in range(self.num_episodes)]

        goal_bits_idx = int(self.true_goal_bits_idx)
        n_bits = self.n_bits
        max_steps = self.max_steps
        for ep in range(self.num_episodes):
            A_acc = 0.0
            curr_cand, samp = self.sample_candidate()
            _, V, pi = self.plan_policy(curr_cand)

            bits_idx = int(self.origin_bits_idx)
            info_bit = int(self.origin_bit if self.reward_type != "stochastic" else 0)
            state_idx = pack_state_idx(bits_idx, info_bit, n_bits)

            total = 0.0
            reg = 0.0
            switches_ep = []

            trace = {
                "states": [],
                "actions": [],
                "rewards": [],
                "td": [],
                "switches": [],
                "success": False,
                "goal": list(self.true_goal),
            }

            reached = False
            for t in range(max_steps):
                if bits_idx == goal_bits_idx:
                    reached = True
                    break

                active_cand = curr_cand
                curr_bit = int(info_bit)

                a = int(pi.sample_action_idx(state_idx, self.rng))
                ns_idx, r_true = self._true_step_idx(state_idx, a)
                td, A_acc = self._detection_signal(V, state_idx, a, ns_idx, r_true, A_acc)
                total += float(r_true)
                reg += float(np.max(self.Q_star[int(state_idx)])) - float(self.Q_star[int(state_idx), a])

                if self.out_dir is not None:
                    bits_i, bit_i = unpack_state_idx(int(state_idx), n_bits)
                    s_tuple = idx_to_bits(bits_i, n_bits) + (int(bit_i),)
                    trace["states"].append(list(s_tuple))
                    trace["actions"].append(int(a))
                    trace["rewards"].append(float(r_true))
                    trace["td"].append(float(td))

                state_idx = int(ns_idx)
                bits_idx, info_bit = unpack_state_idx(state_idx, n_bits)

                if bits_idx == goal_bits_idx:
                    reached = True
                    break

                false_cand = self._observed_false_candidate(bits_idx)
                if false_cand is not None:
                    self.update_posterior(false_cand, False, bit=int(info_bit))
                    if false_cand == active_cand:
                        if t + 1 >= max_steps:
                            break
                        switches_ep.append(int(t))
                        trace["switches"].append(int(t))
                        A_acc = 0.0
                        curr_cand, samp = self.sample_candidate()
                        _, V, pi = self.plan_policy(curr_cand)
                        continue

                if (t + 1) >= self.min_steps and td < self.td_threshold:
                    self.update_posterior(active_cand, False, bit=curr_bit)

                    if t + 1 >= max_steps:
                        break
                    switches_ep.append(int(t))
                    trace["switches"].append(int(t))
                    A_acc = 0.0
                    curr_cand, samp = self.sample_candidate()
                    _, V, pi = self.plan_policy(curr_cand)
                    continue
            
            if reached:
                self.update_posterior(self.true_goal, True, bit=int(info_bit))

            trace["success"] = bool(reached)

            returns[ep] = total
            regrets[ep] = reg
            switches_all[ep] = switches_ep

            if self.out_dir is not None:
                save_episode_trace(os.path.join(self.out_dir, "traces"), ep, trace)

        return returns, regrets, _switches_to_obj_array(switches_all, self.num_episodes)


class WeightedRecovery(Recovery):
    def _weighted_draw(self, false_candidate_goal_bits):
        false_idx = int(bits_to_idx(false_candidate_goal_bits, self.n_bits))
        cand_idxs = np.asarray(self.candidate_goal_bits_idxs, dtype=np.int64)

        raw = np.vectorize(popcount_u16)((cand_idxs ^ false_idx).astype(np.int64)).astype(np.float64)
        dists = raw / raw.max() if raw.max() > 0 else raw
        bias = np.exp(self.tau * dists)
        beta_samples = np.array([
            self.rng.beta(self.beta_params[g][0], self.beta_params[g][1])
            for g in self.candidate_goals
        ], dtype=np.float64)
        scores = beta_samples * bias
        idx = int(np.argmax(scores))
        return self.candidate_goals[idx], float(beta_samples[idx])

    def run(self):
        returns = np.zeros((self.num_episodes,), dtype=np.float64)
        regrets = np.zeros((self.num_episodes,), dtype=np.float64)
        switches_all = [[] for _ in range(self.num_episodes)]

        goal_bits_idx = int(self.true_goal_bits_idx)
        n_bits = self.n_bits
        max_steps = self.max_steps
        for ep in range(self.num_episodes):
            A_acc = 0.0
            curr_cand, samp = self.sample_candidate()
            _, V, pi = self.plan_policy(curr_cand)

            bits_idx = int(self.origin_bits_idx)
            info_bit = int(self.origin_bit if self.reward_type != "stochastic" else 0)
            state_idx = pack_state_idx(bits_idx, info_bit, n_bits)

            total = 0.0
            reg = 0.0
            switches_ep = []

            trace = {
                "states": [],
                "actions": [],
                "rewards": [],
                "td": [],
                "switches": [],
                "success": False,
                "goal": list(self.true_goal),
            }

            reached = False
            for t in range(max_steps):
                if bits_idx == goal_bits_idx:
                    reached = True
                    break

                active_cand = curr_cand
                curr_bit = int(info_bit)

                a = int(pi.sample_action_idx(state_idx, self.rng))
                ns_idx, r_true = self._true_step_idx(state_idx, a)
                td, A_acc = self._detection_signal(V, state_idx, a, ns_idx, r_true, A_acc)
                total += float(r_true)
                reg += float(np.max(self.Q_star[int(state_idx)])) - float(self.Q_star[int(state_idx), a])

                if self.out_dir is not None:
                    bits_i, bit_i = unpack_state_idx(int(state_idx), n_bits)
                    s_tuple = idx_to_bits(bits_i, n_bits) + (int(bit_i),)
                    trace["states"].append(list(s_tuple))
                    trace["actions"].append(int(a))
                    trace["rewards"].append(float(r_true))
                    trace["td"].append(float(td))

                state_idx = int(ns_idx)
                bits_idx, info_bit = unpack_state_idx(state_idx, n_bits)

                if bits_idx == goal_bits_idx:
                    reached = True
                    break

                false_cand = self._observed_false_candidate(bits_idx)
                if false_cand is not None:
                    self.update_posterior(false_cand, False, bit=int(info_bit))
                    if false_cand == active_cand:
                        if t + 1 >= max_steps:
                            break
                        switches_ep.append(int(t))
                        trace["switches"].append(int(t))
                        A_acc = 0.0
                        curr_cand, samp = self._weighted_draw(active_cand)
                        _, V, pi = self.plan_policy(curr_cand)
                        continue

                if (t + 1) >= self.min_steps and td < self.td_threshold:
                    self.update_posterior(active_cand, False, bit=curr_bit)

                    if t + 1 >= max_steps:
                        break
                    switches_ep.append(int(t))
                    trace["switches"].append(int(t))
                    A_acc = 0.0
                    curr_cand, samp = self._weighted_draw(active_cand)
                    _, V, pi = self.plan_policy(curr_cand)
                    continue
            
            if reached:
                self.update_posterior(self.true_goal, True, bit=int(info_bit))

            trace["success"] = bool(reached)

            returns[ep] = total
            regrets[ep] = reg
            switches_all[ep] = switches_ep

            if self.out_dir is not None:
                save_episode_trace(os.path.join(self.out_dir, "traces"), ep, trace)

        return returns, regrets, _switches_to_obj_array(switches_all, self.num_episodes)


class WeightedGraphRecovery(WeightedRecovery):
    pass


class WeightedDirectionalRecovery(Recovery):
    @staticmethod
    def _angle_weight(move_vec, goal_vec):
        nm = float(np.linalg.norm(move_vec))
        ng = float(np.linalg.norm(goal_vec))
        if nm == 0.0 or ng == 0.0:
            return 0.0
        cos = float(np.clip(np.dot(move_vec, goal_vec) / (nm * ng), -1.0, 1.0))
        theta = float(np.arccos(cos))
        return theta / np.pi

    def _weighted_draw(self, prev_bits_idx: int, next_bits_idx: int):
        prev_bits = np.asarray(idx_to_bits(int(prev_bits_idx), self.n_bits), dtype=np.float64)
        next_bits = np.asarray(idx_to_bits(int(next_bits_idx), self.n_bits), dtype=np.float64)
        move_vec = next_bits - prev_bits

        bias = []
        for g in self.candidate_goals:
            goal_vec = np.asarray(g, dtype=np.float64) - prev_bits
            w = float(self._angle_weight(move_vec, goal_vec))
            bias.append(float(np.exp(self.tau * w)))
        bias = np.asarray(bias, dtype=np.float64)
        beta_samples = np.array([
            self.rng.beta(self.beta_params[g][0], self.beta_params[g][1])
            for g in self.candidate_goals
        ], dtype=np.float64)
        scores = beta_samples * bias
        idx = int(np.argmax(scores))
        return self.candidate_goals[idx], float(beta_samples[idx])

    def run(self):
        returns = np.zeros((self.num_episodes,), dtype=np.float64)
        regrets = np.zeros((self.num_episodes,), dtype=np.float64)
        switches_all = [[] for _ in range(self.num_episodes)]

        goal_bits_idx = int(self.true_goal_bits_idx)
        n_bits = self.n_bits
        max_steps = self.max_steps
        for ep in range(self.num_episodes):
            A_acc = 0.0
            curr_cand, samp = self.sample_candidate()
            _, V, pi = self.plan_policy(curr_cand)

            bits_idx = int(self.origin_bits_idx)
            info_bit = int(self.origin_bit if self.reward_type != "stochastic" else 0)
            state_idx = pack_state_idx(bits_idx, info_bit, n_bits)

            total = 0.0
            reg = 0.0
            switches_ep = []

            trace = {
                "states": [],
                "actions": [],
                "rewards": [],
                "td": [],
                "switches": [],
                "success": False,
                "goal": list(self.true_goal),
            }

            reached = False
            for t in range(max_steps):
                if bits_idx == goal_bits_idx:
                    reached = True
                    break

                active_cand = curr_cand
                curr_bit = int(info_bit)

                a = int(pi.sample_action_idx(state_idx, self.rng))
                ns_idx, r_true = self._true_step_idx(state_idx, a)
                td, A_acc = self._detection_signal(V, state_idx, a, ns_idx, r_true, A_acc)
                prev_bits_idx = int(bits_idx)
                next_bits_idx, _ = unpack_state_idx(int(ns_idx), n_bits)
                total += float(r_true)
                reg += float(np.max(self.Q_star[int(state_idx)])) - float(self.Q_star[int(state_idx), a])

                if self.out_dir is not None:
                    bits_i, bit_i = unpack_state_idx(int(state_idx), n_bits)
                    s_tuple = idx_to_bits(bits_i, n_bits) + (int(bit_i),)
                    trace["states"].append(list(s_tuple))
                    trace["actions"].append(int(a))
                    trace["rewards"].append(float(r_true))
                    trace["td"].append(float(td))

                state_idx = int(ns_idx)
                bits_idx, info_bit = unpack_state_idx(state_idx, n_bits)

                if bits_idx == goal_bits_idx:
                    reached = True
                    break

                false_cand = self._observed_false_candidate(bits_idx)
                if false_cand is not None:
                    self.update_posterior(false_cand, False, bit=int(info_bit))
                    if false_cand == active_cand:
                        if t + 1 >= max_steps:
                            break
                        switches_ep.append(int(t))
                        trace["switches"].append(int(t))
                        A_acc = 0.0
                        curr_cand, samp = self._weighted_draw(prev_bits_idx, next_bits_idx)
                        _, V, pi = self.plan_policy(curr_cand)
                        continue

                if (t + 1) >= self.min_steps and td < self.td_threshold:
                    self.update_posterior(active_cand, False, bit=curr_bit)

                    if t + 1 >= max_steps:
                        break
                    switches_ep.append(int(t))
                    trace["switches"].append(int(t))
                    A_acc = 0.0
                    curr_cand, samp = self._weighted_draw(prev_bits_idx, next_bits_idx)
                    _, V, pi = self.plan_policy(curr_cand)
                    continue
            
            if reached:
                self.update_posterior(self.true_goal, True, bit=int(info_bit))

            trace["success"] = bool(reached)

            returns[ep] = total
            regrets[ep] = reg
            switches_all[ep] = switches_ep

            if self.out_dir is not None:
                save_episode_trace(os.path.join(self.out_dir, "traces"), ep, trace)

        return returns, regrets, _switches_to_obj_array(switches_all, self.num_episodes)
