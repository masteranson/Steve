import sys
from GridWorld import GridWorld
from ValueIteration import ValueIteration
from Policy import Policy
from IterativePolicyEvaluation import IterativePolicyEvaluation

def runValueIteration(width, height, gamma, special_states=None, walls=None, origin_state=None, terminal_states=None):
    # Example: special states from Sutton & Barto Ex 3.5
    # A = (0,1), A' = (4,1): reward +10
    # B = (0,3), B' = (2,3): reward +5
    # Ex. All actions from A lead to A', and all actions from B lead to B' with the given rewards

    # Walls behave as boundaries
    # An origin state defines where the agent starts
    # Terminal states stop all actions with reward=0 (absorbing)

    # Create GridWorld with special states, walls, origin states, terminal states
    env = GridWorld(
        width=width,
        height=height,
        gamma=gamma,
        special_states=special_states,
        walls=walls,
        origin_state=origin_state,
        terminal_states=terminal_states
    )

    # Run value iteration
    vi = ValueIteration(env)
    V, policy = vi.run()

    # Print optimal value function
    print("Optimal Value Function:")
    for state in sorted(env.states):
        print(f"State {state}: {V[state]:.10f}")

    # Print derived policy
    print("\nOptimal Policy:")
    for state in sorted(env.states):
        dist = policy.get_action_probabilities(state)
        # Map indices back to action names, e.g. env.actions = ['up','right','down','left']
        actions_with_probs = [
            (env.actions[i], p) 
            for i, p in enumerate(dist) 
            if p > 0
        ]
        print(f"State {state}: {actions_with_probs}")
    return V, policy

def runIterativePolicyEvaluation(width, height, gamma, special_states=None, walls=None, origin_state=None, terminal_states=None, rules=None, default_action=None):
    # Special states, walls, and terminal states behave as above

    # Create GridWorld with special states, walls, terminal states
    env = GridWorld(
        width=width,
        height=height,
        gamma=gamma,
        special_states=special_states,
        walls=walls,
        origin_state=origin_state,
        terminal_states=terminal_states
    )

    # Create policy
    policy = Policy(rules=rules, default_action=default_action)

    # Run policy evaluation
    ipe = IterativePolicyEvaluation(env, policy)
    V = ipe.run()

    # Print optimal value function
    print("Value Function:")
    for state in sorted(env.states):
        print(f"State {state}: {V[state]:.10f}")
    return V

def calculateTDError(width, height, gamma, policy, start_state, action_policy, special_states=None, walls=None, origin_state=None, terminal_states=None):
    """
    Computes a 1-step TD error for a given policy and action policy.
    Returns the 1-step TD error after following the sequence.
    """
    # Create GridWorld with special states, walls, origin states, terminal states
    env = GridWorld(
        width=width,
        height=height,
        gamma=gamma,
        special_states=special_states,
        walls=walls,
        origin_state=origin_state,
        terminal_states=terminal_states
    )
    
    ipe = IterativePolicyEvaluation(env, policy)
    V = ipe.run()

    expected_td_error = 0

    probs = action_policy.get_action_probabilities(start_state)
    for action in env.actions:
        prob = probs[env.actions.index(action)]

        # Take one action from start_state
        next_state, reward = env.get_next_state_and_reward(start_state, action)

        # Compute the 1-step return
        G = reward
    
        # If next_state is not terminal, add discounted bootstrap from V[next_state]
        if not env.is_terminal(next_state):
            G += gamma * V[next_state]

        # The TD error is the difference between G and V[start_state]
        td_error = G - V[start_state]
        expected_td_error += prob * td_error
    return expected_td_error


if __name__ == "__main__":
    args = sys.argv[1:]
    # Example from Sutton & Barto Ex 3.5
    # special_states = [
    #     ((0,1), (4,1), 10),
    #     ((0,3), (2,3), 5),
    # ]
    # V, policy = runValueIteration(5, 5, 0.75, special_states=special_states)

    # Four Rooms Problem
    walls = [
        (0,6),
        (1,6),
        # (2,6),
        # (3,6),
        (4,6),
        (5,6),
        (8,6),
        # (9,6),
        # (10,6),
        (11,6),
        (12,6),

        (6,0),
        (6,1),
        # (6,2),
        # (6,3),
        (6,4),
        (6,5),
        (6,6),

        (7,6),
        (7,7),
        (7,8),
        (7,9),
        (7,10),
        (7,11),
        (7,12)
        ]
    terminal_states = [(0,12)]
    origin_state = (10,2)

    # Rule probabilities are [P(up), P(right), P(down), P(left)]
    # Make sure they sum to 1
    # (Ex. (lambda x, y: x == 1 and 0 < y < 5, [0.5, 0.25, 0, 0.25]))]

    # Agent is drawn to the center of the dead end room @ state[8,8]
    faulty_policy_1 = [
    (lambda x, y: x == 10 and 1 < y < 9, [0, 1, 0, 0])
    ]

    # Agent is drawn to the top left corner of the left upper room @ state[0,0]
    faulty_policy_2 = [
    (lambda x, y: 5 < x < 11 and y == 2, [1, 0, 0, 0]),
    (lambda x, y: x == 5 and 0 < y < 3, [0, 0, 0, 1]),
    (lambda x, y: 0 < x < 6 and y == 0, [1, 0, 0, 0]),
    ]
    
    # Dictionary of policies
    policies = {
        "default": Policy(),
        "1": Policy(rules=faulty_policy_1),
        "2": Policy(rules=faulty_policy_2)
    }

    V, policy_optimal = runValueIteration(13, 13, 0.9, walls=walls, terminal_states=terminal_states, origin_state=origin_state)

    policies["optimal"] = policy_optimal

    # V = runIterativePolicyEvaluation(13, 13, 0.9, walls=walls, terminal_states=terminal_states, rules=policies[args[0]], origin_state=origin_state)
    
    # TD error calculations
    # Optimal policy, action from faulty policy 1
    td_1 = calculateTDError(13, 13, 0.9, policies["optimal"], (10,3), policies["1"], walls=walls, terminal_states=terminal_states, origin_state=origin_state)
    
    # Optimal policy, action from faulty policy 2
    td_2 = calculateTDError(13, 13, 0.9, policies["optimal"], (5,2), policies["2"], walls=walls, terminal_states=terminal_states, origin_state=origin_state)
    
    # Faulty policy 1, action from optimal policy
    td_3 = calculateTDError(13, 13, 0.9, policies["1"], (10,3), policies["optimal"], walls=walls, terminal_states=terminal_states, origin_state=origin_state)
    
    # Faulty policy 2, action from optimal policy
    td_4 = calculateTDError(13, 13, 0.9, policies["2"], (5,2), policies["optimal"], walls=walls, terminal_states=terminal_states, origin_state=origin_state)
    
    # Optimal policy, action from optimal policy
    td_5 = calculateTDError(13, 13, 0.9, policies["optimal"], (10,3), policies["optimal"], walls=walls, terminal_states=terminal_states, origin_state=origin_state)

    print(f"TD error for optimal policy, action from faulty policy 1: {td_1}")
    print(f"TD error for optimal policy, action from faulty policy 2: {td_2}")
    print(f"TD error for faulty policy 1, action from optimal policy: {td_3}")
    print(f"TD error for faulty policy 2, action from optimal policy: {td_4}")
    print(f"TD error for optimal policy, action from optimal policy: {td_5}")
