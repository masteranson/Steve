class IterativePolicyEvaluation:
    def __init__(self, env, policy, theta=1e-5, max_iterations=1000):
        """
        Args:
            env: GridWorld environment.
            theta: Threshold for convergence.
            max_iterations: Maximum number of evaluation sweeps.
        """
        self.env = env
        self.theta = theta
        self.max_iterations = max_iterations
        self.policy = policy

        # Value function initialized to 0
        self.V = {s: 0.0 for s in env.states}

    def run(self):
        """
        Perform iterative policy evaluation until convergence or until max_iterations
        Takes in a policy object defining action probabilities for a state.
        Returns final V
        """
        env = self.env
        gamma = self.env.gamma
        actions = self.env.actions

        for _ in range(self.max_iterations):
            delta = 0.0
            new_V = self.V.copy()

            for state in env.states:
                terminate, delta, new_V = self._iterate_state(state, env, gamma, delta, new_V, actions)
                if terminate == -1:
                    continue

            # Update value function
            self.V = new_V

            # Convergence check
            if delta < self.theta:
                break

        return self.V

    def _iterate_state(self, state, env, gamma, delta, new_V, actions):
        # Preserve the existing terminal-state convention used by this helper.
        if env.is_terminal(state):
            new_V[state] = 1.0
            return -1, delta, new_V

        # Evaluate V(s) = sum over a [ pi(a|s) * ( R(s,a) + gamma * V(s') ) ]
        action_probs = self.policy.get_action_probabilities(state)
        v_temp = 0.0

        use_tp = hasattr(env, 'transition_probs')

        for move_idx, prob in enumerate(action_probs):
            if prob == 0.0:
                continue
            # Get possible action a
            env_action = actions[move_idx]

            if use_tp:
                # Stochastic transitions: marginalise over P(s'|s,a)
                for p, next_state, reward in env.transition_probs(state, env_action):
                    v_temp += prob * p * (reward + gamma * self.V[next_state])
            else:
                # Deterministic fallback
                next_state, reward = env.get_next_state_and_reward(state, env_action)
                v_temp += prob * (reward + gamma * self.V[next_state])

        new_V[state] = v_temp
        delta = max(delta, abs(new_V[state] - self.V[state]))

        return 0, delta, new_V
