import os
import numpy as np
from abc import ABC, abstractmethod
from collections import deque

try:
    from .GridWorld import GridWorld
    from .ValueIteration import ValueIteration
except ImportError:
    from GridWorld import GridWorld
    from ValueIteration import ValueIteration

try:
    from .visualization_utils import save_episode_visualization
except ImportError as e:
    if getattr(e, "name", None) == "pygame":
        def save_episode_visualization(*args, **kwargs):
            return None
    else:
        try:
            from visualization_utils import save_episode_visualization
        except ModuleNotFoundError as fallback_error:
            if fallback_error.name != "pygame":
                raise

            def save_episode_visualization(*args, **kwargs):
                return None


if "save_episode_visualization" not in globals():
    def save_episode_visualization(*args, **kwargs):
        return None

# --- GridWorld subclasses for each PSRL variant ---
class PSRLStandardGridWorld(GridWorld):
    def __init__(self, width, height, gamma, candidate_goal, reward_type='binary',
                 special_states=None, walls=None, origin_state=None, true_goal=None,
                 slip_prob=0.0):
        terminal_states = [
            (candidate_goal[0], candidate_goal[1], 0),
            (candidate_goal[0], candidate_goal[1], 1),
        ]
        super().__init__(width, height, gamma, special_states, walls, origin_state, terminal_states,
                         reward_type=reward_type, slip_prob=slip_prob)
        self.candidate_goal = candidate_goal
        self.true_goal = true_goal

    def _deterministic_step(self, state, action):
        """Candidate-reward variant of the deterministic step."""
        if self.is_terminal(state):
            return (state, 0)
        if state in self.special_dict:
            return self.special_dict[state]
        i, j, b = state
        new_i, new_j = i, j
        if action == 'up':    new_i = max(i - 1, 0)
        elif action == 'right': new_j = min(j + 1, self.width - 1)
        elif action == 'down':  new_i = min(i + 1, self.height - 1)
        elif action == 'left':  new_j = max(j - 1, 0)
        next_state = (new_i, new_j, b)
        if b == 1:
            # delegate full euclidean handling to parent
            return super()._deterministic_step(state, action)

        if (new_i, new_j) in self.walls or next_state == state:
            return (state, 0)
        if next_state in self.terminal_states:
            return (next_state, 1.0)
        return (next_state, 0)

# ------------------------------------------------------------------ #
#  GridWorld wrapper that implements the "stochastic" information bit
# ------------------------------------------------------------------ #
class StochasticGridWorld(GridWorld):
    # global cache: {(layout_key, cand_tuple): {goal: dist-array}}
    _GLOBAL_DIST_CACHE = {}
    def __init__(self, *, candidate_goals, true_goal, task_reward_only=False, **kw):
        terminal_states = [
            (true_goal[0], true_goal[1], 0),
            (true_goal[0], true_goal[1], 1),
        ]
        super().__init__(reward_type="stochastic",
                         terminal_states=terminal_states,
                         task_reward_only=task_reward_only,
                         **kw)

        self.candidate_goals = list(candidate_goals)
        self.true_goal       = tuple(true_goal)

        # --- build / fetch cached distance maps -----------------------
        self._walls2d = {(w[0], w[1]) for w in self.walls}
        H, W          = self.height, self.width
        layout_key    = (H, W, frozenset(self._walls2d))
        cand_key      = tuple(sorted(self.candidate_goals))
        cache_key     = (layout_key, cand_key)

        if cache_key not in StochasticGridWorld._GLOBAL_DIST_CACHE:
            dist_bank = {}
            for (gx, gy) in self.candidate_goals:
                dist = np.full((H, W), np.inf, dtype=np.float32)
                dist[gx, gy] = 0.0
                q = deque([(gx, gy)])
                while q:
                    x, y = q.popleft()
                    step = dist[x, y] + 1.0
                    for dx, dy in ((-1,0),(1,0),(0,-1),(0,1)):
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < H and 0 <= ny < W and
                            (nx, ny) not in self._walls2d and
                            dist[nx, ny] == np.inf):
                            dist[nx, ny] = step
                            q.append((nx, ny))
                dist_bank[(gx, gy)] = dist
            StochasticGridWorld._GLOBAL_DIST_CACHE[cache_key] = dist_bank

        # point to the shared (possibly newly-created) dict
        self._dist_map = StochasticGridWorld._GLOBAL_DIST_CACHE[cache_key]
        self._max_dist = np.linalg.norm([H - 1, W - 1])

    def info_prob1(self, x, y):
        """Return p_1 = P(bit = 1) at grid cell (x,y)."""
        # 1) compute weights  w_g = 1 / d_g^2
        w = []
        for g in self.candidate_goals:
            d = max(self._dist_map[g][x, y], 1)  # avoid 0
            w.append(1.0 / (d * d))
        w = np.asarray(w, dtype=float)

        # 2) soft-max and pick the true-goal component
        w_exp = np.exp(w - w.max())
        soft  = w_exp / w_exp.sum()
        idx   = self.candidate_goals.index(self.true_goal)
        return float(soft[idx])

    # ---------- transition distribution -------------------------------- #
    def transition_probs(self, state, action):
        """Enumerate (p, next_state, reward) including info-bit stochasticity
        and optional movement slippage."""
        if self.is_terminal(state):
            return [(1.0, state, 0)]

        # Get movement outcomes from base GridWorld (handles slippage)
        base_outcomes = super().transition_probs(state, action)

        # Split each movement outcome by info bit
        result = []
        for p_move, base_ns, reward in base_outcomes:
            x, y, _ = base_ns
            p1 = self.info_prob1(x, y)
            result.append((p_move * (1.0 - p1), (x, y, 0), reward))
            result.append((p_move * p1,          (x, y, 1), reward))
        return result

    # ---------- main transition function ------------------------------ #
    def get_next_state_and_reward(self, state, action):
        if self.is_terminal(state):
            return state, 0.0

        if self.slip_prob == 0.0:
            # Original path: deterministic movement + stochastic info bit
            # (preserves random-number sequence for backward compatibility)
            base_next, reward = super()._deterministic_step(state, action)
            x, y, _ = base_next

            p_one = self.info_prob1(x, y)
            b_new = np.random.binomial(1, p_one)
            return (x, y, b_new), reward
        else:
            # With slippage: sample from full transition distribution
            outcomes = self.transition_probs(state, action)
            probs = [o[0] for o in outcomes]
            idx = np.random.choice(len(outcomes), p=probs)
            return outcomes[idx][1], outcomes[idx][2]


# ------------------------------------------------------------
#  helpers
# ------------------------------------------------------------
def expected_td(env, V, state, action):
    """
    Return E[ TD ]  =  Sigma_{s'} P(s'|s,a)[ r(s,a,s') + gamma V(s') ] - V(s).
    Works for any env that implements transition_probs().
    """
    if not hasattr(env, "transition_probs"):
        # deterministic fallback -- uses the single successor
        ns, r = env.get_next_state_and_reward(state, action)
        return r + env.gamma * V[ns] - V[state]

    q_exp = 0.0
    for p, sp, r in env.transition_probs(state, action):
        q_exp += p * (r + env.gamma * V[sp])
    return q_exp - V[state]

# --- Base class for PSRL agents ---
class PSRLAgentBase(ABC):
    def __init__(self, width, height, gamma,
                 candidate_goals, true_goal,
                 walls=None, special_states=None, origin_state=None, reward_type='binary',
                 num_episodes=50, max_steps_per_episode=100,
                 prior_alpha=1.0, prior_beta=1.0,
                 slip_prob=0.0, detection='td', gae_lambda=0.95):
        self.width = width
        self.height = height
        self.gamma = gamma
        self.candidate_goals = candidate_goals
        self.true_goal = true_goal
        self.walls = walls or []
        self.special_states = special_states or []

        # Convert a 2-tuple (x, y) start position into the full (x, y, b) triple
        # so that it matches the state keys stored in value-iteration policies.
        if origin_state is not None and len(origin_state) == 2:
            info_bit   = 1 if reward_type == "euclidean" else 0
            origin_state = (*origin_state, info_bit)
        self.origin_state = origin_state

        self.num_episodes = num_episodes
        self.max_steps = max_steps_per_episode
        self.reward_type = reward_type

        # Stochastic dynamics & detection
        self.slip_prob = float(slip_prob)
        self.detection = detection        # 'td' (oracle) or 'advantage' (GAE-style)
        self.gae_lambda = float(gae_lambda)

        # Independent Beta priors over candidate-goal success scores.
        self.candidate_goals = [tuple(c) for c in candidate_goals]
        self.beta_params = {
            c: [float(prior_alpha), float(prior_beta)]
            for c in self.candidate_goals
        }
        if reward_type == "stochastic":
            self.true_env = StochasticGridWorld(
                width=self.width,
                height=self.height,
                gamma=self.gamma,
                special_states=self.special_states,
                walls=self.walls,
                origin_state=self.origin_state,
                candidate_goals=self.candidate_goals,
                true_goal=self.true_goal,
                slip_prob=self.slip_prob,
            )
        else:
            self.true_env = GridWorld(
                width=self.width,
                height=self.height,
                gamma=self.gamma,
                special_states=self.special_states,
                walls=self.walls,
                origin_state=self.origin_state,
                terminal_states=[self.true_goal],
                reward_type=reward_type,
                slip_prob=self.slip_prob,
            )

        # Match the original experimental metric: regret is computed under
        # the same reward regime used by the environment.
        self.eval_env = self.true_env

        self.seed = None
        self.vis_cache_dir = None
        self._vi_cache = {}

    def sample_candidate(self):
        samples = {
            c: np.random.beta(a, b)
            for c, (a, b) in self.beta_params.items()
        }
        selected = max(samples, key=samples.get)
        return selected, float(samples[selected])

    def _observed_false_candidate(self, state):
        candidate = tuple(state[:2])
        if candidate in self.beta_params and candidate != self.true_goal:
            return candidate
        return None

    def _observed_false_candidates_in_td_log(self, td_log):
        false_candidates = []
        seen = set()
        for state, _ in td_log:
            false_cand = self._observed_false_candidate(state)
            if false_cand is not None and false_cand not in seen:
                false_candidates.append((false_cand, state[2]))
                seen.add(false_cand)
        return false_candidates

    # ---------- Value Iteration on sampled goal ----------
    
    def _stoch_vi(self, candidate_goal):
        """
        Exact value-iteration in the *true* stochastic model whose
        state already contains the information bit and where the bit
        flips according to `info_prob1`.
        """
        key = ("stoch", candidate_goal)
        if key not in self._vi_cache:
            env = StochasticGridWorld(
                width=self.width,
                height=self.height,
                gamma=self.gamma,
                special_states=self.special_states,
                walls=self.walls,
                origin_state=self.origin_state,
                candidate_goals=self.candidate_goals,
                true_goal=candidate_goal,   # hypothesized goal
                slip_prob=self.slip_prob,
            )
            V, pi = ValueIteration(env).run()
            self._vi_cache[key] = (env, V, pi)
        return self._vi_cache[key]

    def update_posterior(self, candidate, success, bit=None):
        candidate = tuple(candidate)
        if candidate not in self.beta_params:
            raise ValueError(f"unknown candidate goal: {candidate}")

        if success:
            self.beta_params[candidate][0] += 1.0
        else:
            self.beta_params[candidate][1] += 1.0

    def _build_Q_star(self, V):
        """Compute extrinsic-task Q*(s,a) = E[r_task + gamma V*(s') | s,a]."""
        Q = {}
        for s in self.eval_env.states:
            Q[s] = {}
            for a in self.eval_env.actions:
                q = 0.0
                for p, ns, r in self.eval_env.transition_probs(s, a):
                    q += p * (r + self.gamma * V[ns])
                Q[s][a] = q
        return Q

    def _detection_signal(self, V, state, action, ns, r, A_prev):
        """
        Compute the failure-detection signal.

        In 'td' mode (oracle):
            E_{P_true}[r + gamma V_plan(s')] - V_plan(s)
            (marginalizes over transition stochasticity -- requires oracle
            access to the true reward function)

        In 'advantage' mode (GAE-style accumulated advantage):
            A_t = delta_t + gamma * lambda * A_{t-1}
            where delta_t = r_realized + gamma V_plan(s'_realized) - V_plan(s)
            Uses only observed (realized) transitions -- no oracle access.

        Returns (signal_value, updated_A_accumulator).
        """
        if self.detection == 'td':
            sig = expected_td(self.true_env, V, state, action)
            return sig, A_prev        # A_prev unchanged, not used in td mode

        # Advantage mode: realized TD error + GAE accumulation
        delta = r + self.gamma * V[ns] - V[state]
        A_new = delta + self.gamma * self.gae_lambda * A_prev
        return A_new, A_new

    @abstractmethod
    def run(self):
        pass

# --- Variant A: Standard PSRL ---
class PSRLStandard(PSRLAgentBase):
    def __init__(self, *args, prior_alpha=1.0, prior_beta=1.0, **kwargs):
        super().__init__(*args, prior_alpha=prior_alpha, prior_beta=prior_beta, **kwargs)
        # Compute true optimal V and Q* for regret
        vi = ValueIteration(self.eval_env)
        self.true_V, self.true_policy = vi.run()
        self.Q_star = self._build_Q_star(self.true_V)

    def plan_policy(self, candidate):
        if self.reward_type == "stochastic":
            return self._stoch_vi(candidate)

        env = PSRLStandardGridWorld(
            width=self.width,
            height=self.height,
            gamma=self.gamma,
            candidate_goal=candidate,
            special_states=self.special_states,
            walls=self.walls,
            origin_state=self.origin_state,
            true_goal=self.true_goal,
            reward_type=self.reward_type,
            slip_prob=self.slip_prob,
        )
        V, pi = ValueIteration(env).run()
        return env, V, pi

    def run_episode(self, candidate_V, sampled_candidate, episode):
        state = self.origin_state
        env_model, V, pi = candidate_V
        total_reward = 0
        reached = False
        steps = 0
        episode_regret = 0.0
        td_log = []
        A_acc = 0.0       # advantage accumulator (used in 'advantage' mode)

        while not self.true_env.is_terminal(state) and steps < self.max_steps:
            probs  = pi.get_action_probabilities(state)
            idx    = np.random.choice(len(self.true_env.actions), p=probs)
            action = self.true_env.actions[idx]
            ns, r  = self.true_env.get_next_state_and_reward(state, action)
            total_reward += r
            
            td, A_acc = self._detection_signal(V, state, action, ns, r, A_acc)
            # Regret via Q*
            opt = max(self.Q_star[state].values())
            actual = self.Q_star[state][action]
            episode_regret += opt - actual
            state = ns
            td_log.append((state, td))
            steps += 1
            if self.true_env.is_terminal(state):
                td_log.append((state, 0.0))
                reached = True
                self.update_posterior(self.true_goal, True, bit=state[2])
                break
        policy_cands = [sampled_candidate]
        return total_reward, reached, episode_regret, steps, td_log, policy_cands

    def run(self):
        results = {
            'episode_returns': [],
            'cumulative_rewards': [],
            'episodic_regrets': [],
            'timesteps': []
        }
        cum_reward = 0
        total_steps = 0
        for ep in range(self.num_episodes):
            print(f"PSRL Standard - Episode {ep+1}:")
            cand, samp = self.sample_candidate()
            print(f"  Sampled {cand}")
            candidate_V = self.plan_policy(cand)
            ret, reached, reg, steps, td_log, policy_cands = self.run_episode(candidate_V, cand, ep+1)
            if self.vis_cache_dir:
                save_episode_visualization(
                    grid_mdp=self.true_env,
                    trajectory=td_log,
                    switches=None,
                    early_stop_idx=None,
                    candidate_goals=self.candidate_goals,
                    policy_candidates=policy_cands,
                    save_path=os.path.join(self.vis_cache_dir, f"seed_{self.seed}_ep{ep}.png"),
                    title=f"Standard | Episode {ep} | #Candidates={len(policy_cands)} | {self.reward_type.capitalize()} Reward",
                    show_legend=True
                )

            results['episode_returns'].append(ret)
            cum_reward += ret
            results['cumulative_rewards'].append(cum_reward)
            total_steps += steps
            results['timesteps'].append(total_steps)
            results['episodic_regrets'].append(reg)
            if not reached:
                for false_cand, bit in self._observed_false_candidates_in_td_log(td_log):
                    self.update_posterior(false_cand, False, bit=bit)
            print(f"  Return={ret}, Steps={steps}, Regret={reg:.3f}, Reached={reached}")

        results.update({
            'final_beta_params': self.beta_params,
        })
        return results

# --- Variant B: PSRL with Early Stoppage ---
class PSRLEarlyStopping(PSRLStandard):
    def __init__(self, *args, td_tolerance=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.td_tolerance = td_tolerance

    def run_episode(self, candidate_V, sampled_candidate, episode=None):
        state = self.origin_state
        env_model, V, pi = candidate_V
        total_reward = 0
        reached = False
        steps = 0
        episode_regret = 0.0
        td_log = []
        A_acc = 0.0       # advantage accumulator
        failure_recorded = False
        while not self.true_env.is_terminal(state) and steps < self.max_steps:
            probs = pi.get_action_probabilities(state)
            idx = np.random.choice(len(self.true_env.actions), p=probs)
            action = self.true_env.actions[idx]
            ns, r = self.true_env.get_next_state_and_reward(state, action)
            
            total_reward += r
            td, A_acc = self._detection_signal(V, state, action, ns, r, A_acc)
            opt = max(self.Q_star[state].values())
            actual = self.Q_star[state][action]
            episode_regret += opt - actual
            state = ns
            td_log.append((state, td))
            steps += 1

            if self.true_env.is_terminal(state):
                td_log.append((state, 0.0))
                reached = True
                self.update_posterior(self.true_goal, True, bit=state[2])
                break

            if td < self.td_tolerance:
                print(f"Early stop at step {steps} due to TD={td:.6f}")
                self.update_posterior(sampled_candidate, False, bit=state[2])
                failure_recorded = True
                break
        if not reached and not failure_recorded and steps >= self.max_steps:
            for false_cand, bit in self._observed_false_candidates_in_td_log(td_log):
                self.update_posterior(false_cand, False, bit=bit)
        
        policy_cands = [sampled_candidate]
        return total_reward, reached, episode_regret, steps, td_log, policy_cands

    def run(self):
        results = {
            'episode_returns': [],
            'cumulative_rewards': [],
            'episodic_regrets': [],
            'timesteps': []
        }
        cum_reward = 0
        total_steps = 0
        for ep in range(self.num_episodes):
            print(f"PSRL EarlyStopping - Episode {ep+1}:")
            cand, samp = self.sample_candidate()
            print(f"  Sampled {cand}")
            candidate_V = self.plan_policy(cand)
            ret, reached, reg, steps, td_log, policy_cands = self.run_episode(candidate_V, cand, ep+1)
            if self.vis_cache_dir:
                save_episode_visualization(
                    grid_mdp=self.true_env,
                    trajectory=td_log,
                    switches=None,  # No switches in early stopping
                    early_stop_idx=len(td_log) if not reached else None,  # Mark early stop if didn't reach goal
                    candidate_goals=self.candidate_goals,
                    policy_candidates=policy_cands,
                    save_path=os.path.join(self.vis_cache_dir, f"seed_{self.seed}_ep{ep}.png"),
                    title=f"Early Stopping | Episode {ep} | #Candidates={len(policy_cands)} | {self.reward_type.capitalize()} Reward",
                    show_legend=True
                )
            results['episode_returns'].append(ret)
            cum_reward += ret
            results['cumulative_rewards'].append(cum_reward)
            total_steps += steps
            results['timesteps'].append(total_steps)
            results['episodic_regrets'].append(reg)
            print(f"  Return={ret}, Steps={steps}, Regret={reg:.3f}, Reached={reached}")

        results.update({'final_beta_params': self.beta_params})
        return results

# --- Variant C: PSRL with Recovery ---
class PSRLWithRecovery(PSRLAgentBase):
    def __init__(self, *args, td_tolerance=0, prior_alpha=1.0, prior_beta=1.0, **kwargs):
        super().__init__(*args, prior_alpha=prior_alpha, prior_beta=prior_beta, **kwargs)
        self.td_tolerance = td_tolerance
        # Compute true optimal V and Q* for regret
        vi = ValueIteration(self.eval_env)
        self.true_V, self.true_policy = vi.run()
        self.Q_star = self._build_Q_star(self.true_V)

    def plan_policy(self, candidate):
        if self.reward_type == "stochastic":
            return self._stoch_vi(candidate)

        env = PSRLStandardGridWorld(
            width=self.width,
            height=self.height,
            gamma=self.gamma,
            candidate_goal=candidate,
            special_states=self.special_states,
            walls=self.walls,
            origin_state=self.origin_state,
            true_goal=self.true_goal,
            reward_type=self.reward_type,
            slip_prob=self.slip_prob,
        )
        V, pi = ValueIteration(env).run()
        return env, V, pi

    def run_episode(self, candidate_V, sampled_candidate, episode):
        state = self.origin_state
        env_model, V, pi = candidate_V
        curr_cand = sampled_candidate
        total_reward = 0
        reached = False
        steps = 0
        episode_regret = 0.0
        td_log = []
        switches = []
        policy_cands = [sampled_candidate]
        A_acc = 0.0       # advantage accumulator
        while not self.true_env.is_terminal(state) and steps < self.max_steps:
            curr_bit    = state[2]
            active_cand = curr_cand
            probs = pi.get_action_probabilities(state)
            idx = np.random.choice(len(self.true_env.actions), p=probs)
            action = self.true_env.actions[idx]
            ns, r = self.true_env.get_next_state_and_reward(state, action)
            
                
            total_reward += r
            td, A_acc = self._detection_signal(V, state, action, ns, r, A_acc)
            
            # Regret via Q*
            opt = max(self.Q_star[state].values())
            actual = self.Q_star[state][action]
            episode_regret += opt - actual
            state = ns
            td_log.append((state, td))
            steps += 1
            
            if self.true_env.is_terminal(state):
                td_log.append((state, 0.0))
                reached = True
                self.update_posterior(self.true_goal, True, bit=state[2])
                break

            false_cand = self._observed_false_candidate(state)
            if false_cand is not None:
                self.update_posterior(false_cand, False, bit=state[2])
                if false_cand == active_cand:
                    if steps >= self.max_steps:
                        break

                    switches.append(len(td_log) - 1)
                    A_acc = 0.0

                    curr_cand, samp = self.sample_candidate()
                    if self.reward_type == "stochastic":
                        env_model, V, pi = self._stoch_vi(curr_cand)
                    else:
                        env_model, V, pi = self.plan_policy(curr_cand)

                    if curr_cand not in policy_cands:
                        policy_cands.append(curr_cand)
                    continue

            if td < self.td_tolerance:
                # Recovery trigger
                self.update_posterior(active_cand, False, bit=curr_bit)

                if steps >= self.max_steps:
                    break
                    
                switches.append(len(td_log) - 1)
                A_acc = 0.0

                curr_cand, samp = self.sample_candidate()
                if self.reward_type == "stochastic":
                    env_model, V, pi = self._stoch_vi(curr_cand)
                else:
                    env_model, V, pi = self.plan_policy(curr_cand)

                if curr_cand not in policy_cands:
                    policy_cands.append(curr_cand)
                continue

        return total_reward, reached, episode_regret, steps, switches, td_log, policy_cands

    def run(self):
        results = {
            'episode_returns': [],
            'cumulative_rewards': [],
            'episodic_regrets': [],
            'timesteps': [],
            'policy_switch_counts': []
        }
        cum_reward = 0
        total_steps = 0
        for ep in range(self.num_episodes):
            print(f"PSRL + Recovery - Episode {ep+1}:")
            cand, samp = self.sample_candidate()
            print(f"  Sampled {cand}")
            candidate_V = self.plan_policy(cand)
            ret, reached, reg, steps, switches, td_log, policy_cands = self.run_episode(candidate_V, cand, ep+1)
            if self.vis_cache_dir:
                save_episode_visualization(
                    grid_mdp=self.true_env,
                    trajectory=td_log,
                    switches=switches,
                    early_stop_idx=None,
                    candidate_goals=self.candidate_goals,
                    policy_candidates=policy_cands,
                    save_path=os.path.join(self.vis_cache_dir, f"seed_{self.seed}_ep{ep}.png"),
                    title=f"Recovery | Episode {ep} | #Candidates={len(policy_cands)} | {self.reward_type.capitalize()} Reward",
                    show_legend=True
                )
            results['episode_returns'].append(ret)
            cum_reward += ret
            results['cumulative_rewards'].append(cum_reward)
            total_steps += steps
            results['timesteps'].append(total_steps)
            results['episodic_regrets'].append(reg)
            results['policy_switch_counts'].append(len(switches))
            print(f"  Return={ret}, Steps={steps}, Regret={reg:.3f}, Reached={reached}, Switches={len(switches)}")

        results.update({
            'final_beta_params': self.beta_params,
        })
        return results

# --- Variant D: PSRL with L2-Distance Weighted Recovery ---
class PSRLWeightedRecovery(PSRLWithRecovery):
    def __init__(self,
                 width, height, gamma,
                 candidate_goals, true_goal,
                 walls=None, special_states=None, origin_state=None,
                 reward_type='binary',
                 num_episodes=50, max_steps_per_episode=100,
                 prior_alpha=1.0, prior_beta=1.0,
                 td_tolerance=0, tau=1,
                 slip_prob=0.0, detection='td', gae_lambda=0.95):
        super().__init__(
            width, height, gamma,
            candidate_goals, true_goal,
            walls=walls,
            special_states=special_states,
            origin_state=origin_state,
            reward_type=reward_type,
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            td_tolerance=td_tolerance,
            slip_prob=slip_prob,
            detection=detection,
            gae_lambda=gae_lambda,
        )
        self.tau = tau

    def _weighted_draw(self, current_state, bit=None):
        candidates = list(self.candidate_goals)
        coords = np.array(candidates, dtype=float)
        current = np.array(current_state[:2], dtype=float)

        raw_dists = np.linalg.norm(coords - current, axis=1)
        dists = raw_dists / raw_dists.max() if raw_dists.max() > 0 else raw_dists
        bias = np.exp(self.tau * dists)
        beta_samples = np.array([
            np.random.beta(self.beta_params[c][0], self.beta_params[c][1])
            for c in candidates
        ])
        scores = beta_samples * bias
        idx = int(np.argmax(scores))
        return candidates[idx], float(beta_samples[idx])

    def run_episode(self, candidate_V, sampled_candidate, episode):
        state = self.origin_state

        # Modified initialization for stochastic case
        env_model, V, pi = candidate_V
        curr_cand = sampled_candidate

        total = 0; reached = False; steps = 0; regret = 0.0
        td_log, switches = [], []
        policy_cands     = [sampled_candidate]
        A_acc = 0.0       # advantage accumulator
        
        while not self.true_env.is_terminal(state) and steps < self.max_steps:
            # 1 - choose policy for current info-bit
            curr_bit    = state[2]
            active_cand = curr_cand
            # 2 - act and observe transition
            probs = pi.get_action_probabilities(state)
            a_idx = np.random.choice(len(self.true_env.actions), p=probs)
            action = self.true_env.actions[a_idx]
            ns, r  = self.true_env.get_next_state_and_reward(state, action)
            

            # 3 - book-keeping
            total += r
            td, A_acc = self._detection_signal(V, state, action, ns, r, A_acc)
            regret += max(self.Q_star[state].values()) - self.Q_star[state][action]
            steps  += 1
            state = ns
            td_log.append((state, td))
            # 4 - recovery trigger
            if self.true_env.is_terminal(state):
                td_log.append((state, 0.0))
                reached = True
                self.update_posterior(self.true_goal, True, bit=state[2])
                break

            false_cand = self._observed_false_candidate(state)
            if false_cand is not None:
                self.update_posterior(false_cand, False, bit=state[2])
                if false_cand == active_cand:
                    if steps >= self.max_steps:
                        break

                    switches.append(len(td_log) - 1)
                    A_acc = 0.0

                    curr_cand, samp = self._weighted_draw(state)
                    if self.reward_type == "stochastic":
                        env_model, V, pi = self._stoch_vi(curr_cand)
                    else:
                        env_model, V, pi = self.plan_policy(curr_cand)

                    if curr_cand not in policy_cands:
                        policy_cands.append(curr_cand)
                    continue

            if td < self.td_tolerance:
                self.update_posterior(active_cand, False, bit=curr_bit)

                if steps >= self.max_steps:
                    break
                    
                switches.append(len(td_log) - 1)
                A_acc = 0.0

                curr_cand, samp = self._weighted_draw(state)
                if self.reward_type == "stochastic":
                    env_model, V, pi = self._stoch_vi(curr_cand)
                else:
                    env_model, V, pi = self.plan_policy(curr_cand)

                if curr_cand not in policy_cands:
                    policy_cands.append(curr_cand)
                continue

        return total, reached, regret, steps, switches, td_log, policy_cands
    
    def run(self):
        results = {
            'episode_returns': [],
            'cumulative_rewards': [],
            'episodic_regrets': [],
            'timesteps': [],
            'policy_switch_counts': []
        }
        cum_reward = 0
        total_steps = 0
        for ep in range(self.num_episodes):
            print(f"PSRL + Weighted Recovery - Episode {ep+1}:")
            cand, samp = self.sample_candidate()
            print(f"  Sampled {cand}")
            candidate_V = self.plan_policy(cand)
            ret, reached, reg, steps, switches, td_log, policy_cands = self.run_episode(candidate_V, cand, ep+1)
            if self.vis_cache_dir:
                save_episode_visualization(
                    grid_mdp=self.true_env,
                    trajectory=td_log,
                    switches=switches,
                    early_stop_idx=None,
                    candidate_goals=self.candidate_goals,
                    policy_candidates=policy_cands,
                    save_path=os.path.join(self.vis_cache_dir, f"seed_{self.seed}_ep{ep}.png"),
                    title=f"Weighted Recovery | Episode {ep} | #Candidates={len(policy_cands)} | {self.reward_type.capitalize()} Reward",
                    show_legend=True
                )
            results['episode_returns'].append(ret)
            cum_reward += ret
            results['cumulative_rewards'].append(cum_reward)
            total_steps += steps
            results['timesteps'].append(total_steps)
            results['episodic_regrets'].append(reg)
            results['policy_switch_counts'].append(len(switches))
            print(f"  Return={ret}, Steps={steps}, Regret={reg:.3f}, Reached={reached}, Switches={len(switches)}")

        results.update({
            'final_beta_params': self.beta_params,
        })
        return results

# --- Variant E: PSRL with Shortest-Path Weighted Recovery ---   
class PSRLWeightedGraphRecovery(PSRLWeightedRecovery):
    def _compute_graph_distances(self, source, candidates):
        # BFS on grid to get dist(source -> every cell)
        H, W = self.height, self.width
        walls = set(self.walls)
        dist_map = {source: 0}
        q = deque([source])
        while q:
            i, j = q.popleft()
            d = dist_map[(i, j)]
            for di, dj in [(-1,0),(0,1),(1,0),(0,-1)]:
                ni, nj = i+di, j+dj
                if 0 <= ni < H and 0 <= nj < W and (ni,nj) not in walls and (ni,nj) not in dist_map:
                    dist_map[(ni, nj)] = d + 1
                    q.append((ni, nj))
        # build list of distances for each candidate (fallback to H*W if unreachable)
        maxd = H*W
        return np.array([ dist_map.get(c, maxd) for c in candidates ], dtype=float)

    def _weighted_draw(self, current_state, bit=None):
        cands = list(self.candidate_goals)
        source = tuple(current_state[:2])
        raw = self._compute_graph_distances(source, cands)
        dists = raw / raw.max() if raw.max() > 0 else raw
        bias = np.exp(self.tau * dists)
        beta_samples = np.array([
            np.random.beta(self.beta_params[c][0], self.beta_params[c][1])
            for c in cands
        ])
        scores = beta_samples * bias
        idx = int(np.argmax(scores))
        return cands[idx], float(beta_samples[idx])
        
    def run(self):
        results = {
            'episode_returns': [],
            'cumulative_rewards': [],
            'episodic_regrets': [],
            'timesteps': [],
            'policy_switch_counts': []
        }
        cum_reward = 0
        total_steps = 0
        for ep in range(self.num_episodes):
            print(f"PSRL + Weighted Graph Recovery - Episode {ep+1}:")
            cand, samp = self.sample_candidate()
            print(f"  Sampled {cand}")
            candidate_V = self.plan_policy(cand)
            ret, reached, reg, steps, switches, td_log, policy_cands = self.run_episode(candidate_V, cand, ep+1)
            if self.vis_cache_dir:
                save_episode_visualization(
                    grid_mdp=self.true_env,
                    trajectory=td_log,
                    switches=switches,
                    early_stop_idx=None,
                    candidate_goals=self.candidate_goals,
                    policy_candidates=policy_cands,
                    save_path=os.path.join(self.vis_cache_dir, f"seed_{self.seed}_ep{ep}.png"),
                    title=f"Weighted Graph Recovery | Episode {ep} | #Candidates={len(policy_cands)} | {self.reward_type.capitalize()} Reward",
                    show_legend=True
                )

            results['episode_returns'].append(ret)
            cum_reward += ret
            results['cumulative_rewards'].append(cum_reward)
            total_steps += steps
            results['timesteps'].append(total_steps)
            results['episodic_regrets'].append(reg)
            results['policy_switch_counts'].append(len(switches))
            print(f"  Return={ret}, Steps={steps}, Regret={reg:.3f}, Reached={reached}, Switches={len(switches)}")

        results.update({
            'final_beta_params': self.beta_params,
        })
        return results

# --- Variant F: PSRL with Direction-Angle Weighted Recovery ---
class PSRLWeightedDirectionalRecovery(PSRLWithRecovery):
    """
    On each TD-error triggered switch, draw a new candidate goal
    with probability proportional to  exp(tau * w_g) * BetaDraw_g,
    where   w_g = angle(move_vec, goal_vec) / 180  (in [0,1]),
    and `move_vec` is the agent's attempted step that produced
    the low-TD surprise.
    Antialigned goals (180 deg) get weight 1, aligned goals (0 deg) get 0.
    """
    def __init__(self,
                 width, height, gamma,
                 candidate_goals, true_goal,
                 walls=None, special_states=None, origin_state=None,
                 reward_type='binary',
                 num_episodes=50, max_steps_per_episode=100,
                 prior_alpha=1.0, prior_beta=1.0,
                 td_tolerance=0, tau=1.0,
                 slip_prob=0.0, detection='td', gae_lambda=0.95):
        super().__init__(
            width, height, gamma,
            candidate_goals, true_goal,
            walls=walls,
            special_states=special_states,
            origin_state=origin_state,
            reward_type=reward_type,
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            td_tolerance=td_tolerance,
            slip_prob=slip_prob,
            detection=detection,
            gae_lambda=gae_lambda,
        )
        self.tau = tau

    # -------- helper --------------------------------------------------
    @staticmethod
    def _angle_weight(move_vec, goal_vec):
        """Return w = theta/pi  in [0,1]   (theta in radians)."""
        # handle degenerate vectors gracefully
        nm = np.linalg.norm(move_vec)
        ng = np.linalg.norm(goal_vec)
        if nm == 0 or ng == 0:
            return 0.0       # treat as perfectly aligned
        cos = np.clip(np.dot(move_vec, goal_vec) / (nm * ng), -1.0, 1.0)
        theta = np.arccos(cos)          # in radians, 0..pi
        return theta / np.pi            # scale to 0..1

    def _weighted_draw(self, prev_state, move_vec, bit=None):
        """Draw a candidate with Beta score biased toward directions opposed to the failed move."""
        candidates = list(self.candidate_goals)
        bias = []
        for g in candidates:
            goal_vec = np.array(g, dtype=float) - np.array(prev_state[:2], dtype=float)
            w = self._angle_weight(move_vec, goal_vec)
            bias.append(np.exp(self.tau * w))
        bias = np.asarray(bias, dtype=float)
        beta_samples = np.array([
            np.random.beta(self.beta_params[c][0], self.beta_params[c][1])
            for c in candidates
        ])
        scores = beta_samples * bias
        idx = int(np.argmax(scores))
        return candidates[idx], float(beta_samples[idx])

    # -------- episode loop -------------------------------------------
    def run_episode(self, candidate_V, sampled_candidate, episode):
        state = self.origin_state

        env_model, V, pi = candidate_V
        curr_cand = sampled_candidate

        total = 0; reached = False; steps = 0; regret = 0.0
        td_log, switches = [], []
        policy_cands     = [sampled_candidate]
        A_acc = 0.0       # advantage accumulator
        while not self.true_env.is_terminal(state) and steps < self.max_steps:
            prev = state

            # choose policy for current bit
            curr_bit    = state[2]
            active_cand = curr_cand
            probs = pi.get_action_probabilities(prev)
            a_idx = np.random.choice(len(self.true_env.actions), p=probs)
            action = self.true_env.actions[a_idx]
            ns, r  = self.true_env.get_next_state_and_reward(prev, action)
            

            total += r
            td, A_acc = self._detection_signal(V, prev, action, ns, r, A_acc)
            regret += max(self.Q_star[prev].values()) - self.Q_star[prev][action]
            state = ns
            td_log.append((state, td))
            steps  += 1

            if self.true_env.is_terminal(state):
                td_log.append((state, 0.0))
                reached = True
                self.update_posterior(self.true_goal, True, bit=state[2])
                break

            move_vec = np.array(ns[:2]) - np.array(prev[:2])

            false_cand = self._observed_false_candidate(state)
            if false_cand is not None:
                self.update_posterior(false_cand, False, bit=state[2])
                if false_cand == active_cand:
                    if steps >= self.max_steps:
                        break

                    switches.append(len(td_log) - 1)
                    A_acc = 0.0

                    curr_cand, samp = self._weighted_draw(prev, move_vec)
                    if self.reward_type == "stochastic":
                        env_model, V, pi = self._stoch_vi(curr_cand)
                    else:
                        env_model, V, pi = self.plan_policy(curr_cand)

                    if curr_cand not in policy_cands:
                        policy_cands.append(curr_cand)
                    continue

            # recovery trigger
            if td < self.td_tolerance:
                self.update_posterior(active_cand, False, bit=curr_bit)

                if steps >= self.max_steps:
                    break
                    
                switches.append(len(td_log) - 1)
                A_acc = 0.0

                curr_cand, samp = self._weighted_draw(prev, move_vec)
                if self.reward_type == "stochastic":
                    env_model, V, pi = self._stoch_vi(curr_cand)
                else:
                    env_model, V, pi = self.plan_policy(curr_cand)

                if curr_cand not in policy_cands:
                    policy_cands.append(curr_cand)
                continue

        return total, reached, regret, steps, switches, td_log, policy_cands

    def run(self):
        results = {
            'episode_returns': [],
            'cumulative_rewards': [],
            'episodic_regrets': [],
            'timesteps': [],
            'policy_switch_counts': []
        }
        cum_reward = 0
        total_steps = 0
        for ep in range(self.num_episodes):
            print(f"PSRL + Weighted Directional Recovery - Episode {ep+1}:")
            cand, samp = self.sample_candidate()
            print(f"  Sampled {cand}")
            candidate_V = self.plan_policy(cand)
            ret, reached, reg, steps, switches, td_log, policy_cands = self.run_episode(candidate_V, cand, ep+1)
            if self.vis_cache_dir:
                save_episode_visualization(
                    grid_mdp=self.true_env,
                    trajectory=td_log,
                    switches=switches,
                    early_stop_idx=None,
                    candidate_goals=self.candidate_goals,
                    policy_candidates=policy_cands,
                    save_path=os.path.join(self.vis_cache_dir, f"seed_{self.seed}_ep{ep}.png"),
                    title=f"Weighted Directional Recovery | Episode {ep} | #Candidates={len(policy_cands)} | {self.reward_type.capitalize()} Reward",
                    show_legend=True
                )

            results['episode_returns'].append(ret)
            cum_reward += ret
            results['cumulative_rewards'].append(cum_reward)
            total_steps += steps
            results['timesteps'].append(total_steps)
            results['episodic_regrets'].append(reg)
            results['policy_switch_counts'].append(len(switches))
            print(f"  Return={ret}, Steps={steps}, Regret={reg:.3f}, Reached={reached}, Switches={len(switches)}")

        results.update({
            'final_beta_params': self.beta_params,
        })
        return results

# ================================================================== #
#  Baseline G: LEAST  (Liu et al., ICML 2025)                        #
#  Adaptive early termination via per-step V_plan median threshold.   #
#  Tabular adaptation: gradient modulator omega dropped (no neural net). #
# ================================================================== #
class PSRLWithLEAST(PSRLStandard):
    def __init__(self, *args, least_buffer_size=50, least_warmup=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.least_buffer_size = least_buffer_size
        self.least_warmup = least_warmup
        self._q_buffer = deque(maxlen=least_buffer_size)

    def run_episode(self, candidate_V, sampled_candidate, episode):
        state = self.origin_state
        env_model, V, pi = candidate_V
        total_reward = 0
        reached = False
        steps = 0
        episode_regret = 0.0
        td_log = []
        A_acc = 0.0
        q_trace = []          # V_plan(s) at each step for buffer

        while not self.true_env.is_terminal(state) and steps < self.max_steps:
            q_trace.append(V[state])

            probs = pi.get_action_probabilities(state)
            idx = np.random.choice(len(self.true_env.actions), p=probs)
            action = self.true_env.actions[idx]
            ns, r = self.true_env.get_next_state_and_reward(state, action)

            total_reward += r
            td, A_acc = self._detection_signal(V, state, action, ns, r, A_acc)
            opt = max(self.Q_star[state].values())
            actual = self.Q_star[state][action]
            episode_regret += opt - actual
            state = ns
            td_log.append((state, td))
            steps += 1

            if self.true_env.is_terminal(state):
                td_log.append((state, 0.0))
                reached = True
                self.update_posterior(self.true_goal, True, bit=state[2])
                break

            # --- LEAST adaptive threshold check ---
            if len(self._q_buffer) >= self.least_warmup:
                step_idx = len(q_trace) - 1
                step_vals = [ep[step_idx] for ep in self._q_buffer
                             if len(ep) > step_idx]
                if step_vals and q_trace[-1] < np.median(step_vals):
                    self.update_posterior(sampled_candidate, False, bit=state[2])
                    break

        self._q_buffer.append(q_trace)
        policy_cands = [sampled_candidate]
        return total_reward, reached, episode_regret, steps, td_log, policy_cands

    def run(self):
        results = {
            'episode_returns': [],
            'cumulative_rewards': [],
            'episodic_regrets': [],
            'timesteps': []
        }
        cum_reward = 0
        total_steps = 0
        for ep in range(self.num_episodes):
            print(f"PSRL + LEAST - Episode {ep+1}:")
            cand, samp = self.sample_candidate()
            print(f"  Sampled {cand}")
            candidate_V = self.plan_policy(cand)
            ret, reached, reg, steps, td_log, policy_cands = self.run_episode(
                candidate_V, cand, ep + 1)
            if self.vis_cache_dir:
                save_episode_visualization(
                    grid_mdp=self.true_env,
                    trajectory=td_log,
                    switches=None,
                    early_stop_idx=len(td_log) if not reached else None,
                    candidate_goals=self.candidate_goals,
                    policy_candidates=policy_cands,
                    save_path=os.path.join(self.vis_cache_dir,
                                           f"seed_{self.seed}_ep{ep}.png"),
                    title=(f"LEAST | Episode {ep} | "
                           f"#Candidates={len(policy_cands)} | "
                           f"{self.reward_type.capitalize()} Reward"),
                    show_legend=True,
                )
            results['episode_returns'].append(ret)
            cum_reward += ret
            results['cumulative_rewards'].append(cum_reward)
            total_steps += steps
            results['timesteps'].append(total_steps)
            results['episodic_regrets'].append(reg)
            if not reached:
                for false_cand, bit in self._observed_false_candidates_in_td_log(td_log):
                    self.update_posterior(false_cand, False, bit=bit)
            print(f"  Return={ret}, Steps={steps}, "
                  f"Regret={reg:.3f}, Reached={reached}")
        results['final_beta_params'] = self.beta_params
        return results


#  Baseline H: SEE-oracle  (Griesbach & D'Eramo, 2025)                #
#  Directed exploration via |TD-error| maximisation with mixed        #
#  behavior policy (Boltzmann over relative advantages).              #
#  Tabular adaptation:                                                #
#    - Exploitation policy: standard VI on planning model             #
#    - Exploration policy: maximum-reward VI on true env using        #
#      |expected_td| as the reward signal                             #
#    - Behaviour policy: Boltzmann mixture of both                    #
#    - Fingerprinting unnecessary (full tabular V available)          #
# ================================================================== #
class PSRLWithSEE(PSRLStandard):
    """Oracle tabular SEE diagnostic.

    This baseline intentionally uses the true environment to construct the
    exploration reward |E_true[delta]|. It should be reported as SEE-oracle,
    not as an agent-accessible SEE implementation.
    """

    # ---------- maximum-reward VI ------------------------------------- #
    def _max_reward_vi(self, R, env, gamma,
                       theta=1e-10, max_iter=1000):
        """V(s) = max_a  max( R(s,a), gamma * E[V(s')] )."""
        V = {s: 0.0 for s in env.states}
        for _ in range(max_iter):
            delta = 0.0
            new_V = {}
            for s in env.states:
                if env.is_terminal(s):
                    new_V[s] = 0.0
                    continue
                best = float('-inf')
                for a in env.actions:
                    future = sum(p * V[ns]
                                 for p, ns, _ in env.transition_probs(s, a))
                    val = max(R[s][a], gamma * future)
                    if val > best:
                        best = val
                new_V[s] = best
                delta = max(delta, abs(new_V[s] - V[s]))
            V = new_V
            if delta < theta:
                break
        # Build Q_explore
        Q = {}
        for s in env.states:
            Q[s] = {}
            for a in env.actions:
                if env.is_terminal(s):
                    Q[s][a] = 0.0
                else:
                    future = sum(p * V[ns]
                                 for p, ns, _ in env.transition_probs(s, a))
                    Q[s][a] = max(R[s][a], gamma * future)
        return V, Q

    # ---------- exploration policy ------------------------------------ #
    def _exploration_policy(self, V_plan):
        """Return (Q_explore, greedy_action_dict)."""
        R = {}
        for s in self.true_env.states:
            R[s] = {}
            for a in self.true_env.actions:
                R[s][a] = (0.0 if self.true_env.is_terminal(s)
                           else abs(expected_td(self.true_env, V_plan, s, a)))
        _, Q_explore = self._max_reward_vi(R, self.true_env, self.gamma)
        pi = {s: max(self.true_env.actions, key=lambda a: Q_explore[s][a])
              for s in self.true_env.states}
        return Q_explore, pi

    # ---------- exploitation Q from planning model -------------------- #
    @staticmethod
    def _build_Q_plan(env_model, V_plan, gamma):
        Q = {}
        for s in env_model.states:
            Q[s] = {}
            for a in env_model.actions:
                q = 0.0
                for p, ns, r in env_model.transition_probs(s, a):
                    q += p * (r + gamma * V_plan[ns])
                Q[s][a] = q
        return Q

    # ---------- Boltzmann action mixing ------------------------------- #
    def _mix_action(self, state, pi_exploit, Q_exploit, Q_explore, pi_explore):
        probs = pi_exploit.get_action_probabilities(state)
        a_exploit = self.true_env.actions[
            np.random.choice(len(self.true_env.actions), p=probs)]
        a_explore = pi_explore[state]
        if a_exploit == a_explore:
            return a_exploit
        A_exploit = Q_exploit[state][a_exploit] - Q_exploit[state][a_explore]
        A_explore = Q_explore[state][a_explore] - Q_explore[state][a_exploit]
        logits = np.array([A_exploit, A_explore])
        logits -= logits.max()
        e = np.exp(logits)
        if np.random.rand() < e[0] / e.sum():
            return a_exploit
        return a_explore

    # ---------- episode loop ------------------------------------------ #
    def run_episode(self, candidate_V, sampled_candidate, episode):
        state = self.origin_state
        env_model, V_plan, pi_exploit = candidate_V

        Q_exploit = self._build_Q_plan(env_model, V_plan, self.gamma)
        Q_explore, pi_explore = self._exploration_policy(V_plan)

        total_reward = 0; reached = False; steps = 0
        episode_regret = 0.0; td_log = []; A_acc = 0.0

        while not self.true_env.is_terminal(state) and steps < self.max_steps:
            action = self._mix_action(state, pi_exploit,
                                      Q_exploit, Q_explore, pi_explore)
            ns, r = self.true_env.get_next_state_and_reward(state, action)
            total_reward += r
            td, A_acc = self._detection_signal(V_plan, state, action, ns, r, A_acc)
            opt = max(self.Q_star[state].values())
            actual = self.Q_star[state][action]
            episode_regret += opt - actual
            state = ns
            td_log.append((state, td))
            steps += 1
            if self.true_env.is_terminal(state):
                td_log.append((state, 0.0))
                reached = True
                self.update_posterior(self.true_goal, True, bit=state[2])
                break

        policy_cands = [sampled_candidate]
        return total_reward, reached, episode_regret, steps, td_log, policy_cands

    def run(self):
        results = {
            'episode_returns': [],
            'cumulative_rewards': [],
            'episodic_regrets': [],
            'timesteps': []
        }
        cum_reward = 0
        total_steps = 0
        for ep in range(self.num_episodes):
            print(f"PSRL + SEE - Episode {ep+1}:")
            cand, samp = self.sample_candidate()
            print(f"  Sampled {cand}")
            candidate_V = self.plan_policy(cand)
            ret, reached, reg, steps, td_log, policy_cands = self.run_episode(
                candidate_V, cand, ep + 1)
            if self.vis_cache_dir:
                save_episode_visualization(
                    grid_mdp=self.true_env,
                    trajectory=td_log,
                    switches=None,
                    early_stop_idx=None,
                    candidate_goals=self.candidate_goals,
                    policy_candidates=policy_cands,
                    save_path=os.path.join(self.vis_cache_dir,
                                           f"seed_{self.seed}_ep{ep}.png"),
                    title=(f"SEE | Episode {ep} | "
                           f"#Candidates={len(policy_cands)} | "
                           f"{self.reward_type.capitalize()} Reward"),
                    show_legend=True,
                )
            results['episode_returns'].append(ret)
            cum_reward += ret
            results['cumulative_rewards'].append(cum_reward)
            total_steps += steps
            results['timesteps'].append(total_steps)
            results['episodic_regrets'].append(reg)
            if not reached:
                for false_cand, bit in self._observed_false_candidates_in_td_log(td_log):
                    self.update_posterior(false_cand, False, bit=bit)
            print(f"  Return={ret}, Steps={steps}, "
                  f"Regret={reg:.3f}, Reached={reached}")
        results['final_beta_params'] = self.beta_params
        return results
