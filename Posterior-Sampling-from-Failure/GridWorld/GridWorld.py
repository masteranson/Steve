import numpy as np


class GridWorld:
    """
    H x W GridWorld environment

    - Actions: 'up', 'down', 'left', 'right'.
    - Moving "off the grid" or "into a wall" maintains location.
    - A list of special transitions can be provided as a list of tuples:
        (state, next_state, reward)
    - For all normal transitions, reward = 0.
    - Optional Russell & Norvig-style stochastic dynamics:
        with probability slip_prob the agent slips to one of the two
        perpendicular directions (equally likely).  slip_prob=0.0 recovers
        the original deterministic semantics.
    """
    # Perpendicular actions for each action index (used by slippage model)
    # actions = ['up', 'right', 'down', 'left']  -> indices 0,1,2,3
    _PERPENDICULARS = {
        0: (3, 1),   # up    -> left, right
        1: (0, 2),   # right -> up,   down
        2: (1, 3),   # down  -> right, left
        3: (2, 0),   # left  -> down,  up
    }

    def __init__(self,
                 width=5,
                 height=5,
                 gamma=0.9,
                 special_states=None,
                 walls=None,
                 origin_state=None,
                 terminal_states=None,
                 reward_type="binary",
                 slip_prob=0.0,
                 task_reward_only=False):
        """
        Args:
            width (int): Number of columns in the grid.
            height (int): Number of rows in the grid.
            gamma (float): Discount factor.
            special_states (list): A list of (state, next_state, reward) describing special transitions.
            walls (list): A list of states that are walls (barriers).
            terminal_states (list): A list of states that are terminal (absorbing).
            slip_prob (float): Probability of slipping to a perpendicular direction.
                0.0 = deterministic (default).  Russell & Norvig typical: 0.2.
                The intended action succeeds with prob (1 - slip_prob), and each
                of the two perpendicular directions is taken with prob slip_prob/2.
        """
        self.width = width
        self.height = height
        self.gamma = gamma
        self.reward_type = reward_type
        self.slip_prob = float(slip_prob)
        self.task_reward_only = bool(task_reward_only)
        self._max_dist = np.linalg.norm(
            np.array([0, 0]) - np.array([height - 1, width - 1])
        )

        # Actions
        self.actions = ['up', 'right', 'down', 'left']

        # If not provided, default to empty
        if special_states is None:
            special_states = []
        if walls is None:
            walls = []
        if terminal_states is None:
            terminal_states = []
        
        # All grid cells as states
        # Select info-bit from reward_type  (0 = binary, 1 = euclidean)
        _INFO_BIT_BY_REWARD = {
            "binary":     0,   # sparse flag-only feedback
            "euclidean":  1,   # distance-aware feedback
            "stochastic":  0, # stochastic distance-based feedback
        }
        self.info_bit = _INFO_BIT_BY_REWARD.get(reward_type, 0)

        # All grid cells as (x, y, b) states
        if reward_type == "stochastic":
            self.states = [
                (i, j, b)
                for i in range(height)
                for j in range(width)
                if (i, j) not in walls
                for b in (0, 1)
            ]
        else:
            self.states = [
                (i, j, self.info_bit)
                for i in range(height)
                for j in range(width)
                if (i, j) not in walls
            ]

        # Convert special states to dict: state -> (next_state, reward)
        self.special_dict = {}
        for (s, s_prime, r) in special_states:
            self.special_dict[s] = (s_prime, r)

        # Walls and terminal states as sets
        self.walls = set(walls)

        # Auto-expand 2-tuples into (x,y,b)
        conv = lambda s: s if len(s) == 3 else (s[0], s[1], self.info_bit)
        self.origin_state   = None if origin_state is None else conv(origin_state)
        self.terminal_states = set(conv(t) for t in terminal_states)

    def get_goal_locs(self):
        return self.terminal_states

    def is_wall(self, i, j):
        return (i, j) in self.walls

    def is_terminal(self, state):
        # Returns True if this state is terminal (absorbing), False otherwise.
        return state in self.terminal_states

    # ------------------------------------------------------------------ #
    #  Deterministic single-action outcome (no slippage)                  #
    # ------------------------------------------------------------------ #
    def _deterministic_step(self, state, action):
        """Return (next_state, reward) for *exactly* executing `action`."""
        if self.is_terminal(state):
            return (state, 0)

        if state in self.special_dict:
            s_prime, r = self.special_dict[state]
            return (s_prime, r)

        i, j, b = state
        new_i, new_j = i, j

        if action == 'up':
            new_i = max(i - 1, 0)
        elif action == 'right':
            new_j = min(j + 1, self.width - 1)
        elif action == 'down':
            new_i = min(i + 1, self.height - 1)
        elif action == 'left':
            new_j = max(j - 1, 0)

        next_state = (new_i, new_j, b)
        blocked = ((new_i, new_j) in self.walls) or (next_state == state)
        actual  = state if blocked else next_state

        goal_bonus = 1.0 if actual in self.terminal_states else 0.0

        if b == 1 and not self.task_reward_only:
            if len(self.terminal_states) == 0:
                d_min = 0.0
            else:
                d_min = min(
                    np.linalg.norm(np.array(actual[:2]) - np.array(g[:2]))
                    for g in self.terminal_states
                )
            euclid_penalty = -d_min / self._max_dist
            return (actual, goal_bonus + euclid_penalty)

        return (actual, goal_bonus)

    # ------------------------------------------------------------------ #
    #  Transition probability distribution (for VI / advantage calc)      #
    # ------------------------------------------------------------------ #
    def transition_probs(self, state, action):
        """
        Return list of (probability, next_state, reward) tuples.

        When slip_prob == 0 this collapses to a single deterministic outcome.
        With slip_prob > 0 (Russell & Norvig model):
            P(intended)      = 1 - slip_prob
            P(perpendicular) = slip_prob / 2   (each of two perpendiculars)
        """
        if self.slip_prob == 0.0:
            ns, r = self._deterministic_step(state, action)
            return [(1.0, ns, r)]

        if self.is_terminal(state):
            return [(1.0, state, 0)]

        a_idx = self.actions.index(action)
        p1_idx, p2_idx = self._PERPENDICULARS[a_idx]

        p_fwd  = 1.0 - self.slip_prob
        p_side = self.slip_prob / 2.0

        # Merge outcomes that land on the same cell (e.g. near walls/corners)
        outcomes = {}   # next_state -> (accumulated_prob, reward)
        for act_idx, prob in [(a_idx, p_fwd), (p1_idx, p_side), (p2_idx, p_side)]:
            ns, r = self._deterministic_step(state, self.actions[act_idx])
            if ns in outcomes:
                old_p, old_r = outcomes[ns]
                outcomes[ns] = (old_p + prob, old_r)
            else:
                outcomes[ns] = (prob, r)

        return [(p, ns, r) for ns, (p, r) in outcomes.items()]

    # ------------------------------------------------------------------ #
    #  Public step function (samples when stochastic)                     #
    # ------------------------------------------------------------------ #
    def get_next_state_and_reward(self, state, action):
        """
        Returns (next_state, reward).
        When slip_prob == 0 this is deterministic (original behavior).
        When slip_prob > 0 it samples from the R&N slippage distribution.
        """
        if self.slip_prob == 0.0:
            return self._deterministic_step(state, action)

        outcomes = self.transition_probs(state, action)
        probs = [o[0] for o in outcomes]
        idx = np.random.choice(len(outcomes), p=probs)
        return outcomes[idx][1], outcomes[idx][2]
