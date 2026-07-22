try:
    from .Policy import Policy
except ImportError:
    from Policy import Policy


class ValueIteration:
    def __init__(self, env, theta=1e-5, max_iterations=1000):
        """
        Args:
            env: GridWorld environment.
            theta: Threshold for convergence.
            max_iterations: Maximum number of evaluation sweeps.
        """
        self.env = env
        self.theta = theta
        self.max_iterations = max_iterations

        # Value function initialized to 0.
        self.V = {s: 0.0 for s in env.states}

        # Best action set per state. Ties are retained and sampled uniformly.
        self.policy = {s: None for s in env.states}

    def run(self):
        """
        Perform value iteration until convergence or max_iterations.

        Returns:
            (V, policy): final value table and a Policy over greedy actions.
        """
        env = self.env
        gamma = env.gamma

        for _ in range(self.max_iterations):
            delta = 0
            new_V = self.V.copy()
            new_policy = self.policy.copy()

            for state in env.states:
                terminate, delta, new_V, new_policy = self._iterate_state(
                    state, env, gamma, delta, new_V, new_policy
                )
                if terminate == -1:
                    continue

            self.V = new_V
            self.policy = new_policy

            if delta < self.theta:
                break

        final_policy = self._build_policy_object()
        return self.V, final_policy

    def _iterate_state(self, state, env, gamma, delta, new_V, new_policy):
        if env.is_terminal(state):
            new_V[state] = 0.0
            new_policy[state] = None
            return -1, delta, new_V, new_policy

        best_value = float("-inf")
        best_actions = []
        for action in env.actions:
            if hasattr(env, "transition_probs"):
                q = 0.0
                for p, next_state, reward in env.transition_probs(state, action):
                    q += p * (reward + gamma * self.V[next_state])
            else:
                next_state, reward = env.get_next_state_and_reward(state, action)
                q = reward + gamma * self.V[next_state]

            if q > best_value:
                best_value = q
                best_actions = [action]
            elif q == best_value:
                best_actions.append(action)

        new_V[state] = best_value
        new_policy[state] = best_actions
        delta = max(delta, abs(best_value - self.V[state]))
        return 0, delta, new_V, new_policy

    def _build_policy_object(self):
        """
        Convert {state: best_actions} into a policy that is uniform over ties.
        """
        rules = []
        for state, best_actions in self.policy.items():
            if not best_actions:
                distribution = [0.25, 0.25, 0.25, 0.25]
            else:
                distribution = [0.0, 0.0, 0.0, 0.0]
                for act in best_actions:
                    idx = self.env.actions.index(act)
                    distribution[idx] = 1.0 / len(best_actions)

            condition = lambda s, st=state: s == st
            rules.append((condition, distribution))

        return Policy(
            rules=rules,
            default_action=[0.25, 0.25, 0.25, 0.25],
            actions=self.env.actions,
        )
