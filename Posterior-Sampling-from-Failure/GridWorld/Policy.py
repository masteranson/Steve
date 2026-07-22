class Policy:
    """
    Environment-agnostic class to define a policy. Stores a list of rules
    where each is a (condition, action_probs) pair:
      - condition(state) -> bool
      - action_probs -> [P(up), P(right), P(down), P(left)]
    """

    def __init__(self, rules=None, default_action=None, actions=None):
        """
        Args:
            rules (list): A list of tuples (condition, distribution).
                The condition receives the full state tuple.
            default_action (list): A default distribution if no conditions match.
                Defaults to uniform [0.25, 0.25, 0.25, 0.25].
        """
        self.rules = rules if rules else []
        self.default_action = default_action if default_action else [0.25, 0.25, 0.25, 0.25]
        self.actions = actions or ['up', 'right', 'down', 'left']

    def get_action_probabilities(self, state):
        """
        Returns the probability distribution for a given state
        in the form [P(up), P(right), P(down), P(left)].
        """
        # Return probabilities for first matching distribution
        for condition, distribution in self.rules:
            if condition(state):
                return distribution

        # Return default distribution if no conditions match
        return self.default_action
