try:
    from .PSRL_Agents import PSRLStandardGridWorld
    from .visualization_utils import save_episode_visualization
except ImportError:
    from PSRL_Agents import PSRLStandardGridWorld
    from visualization_utils import save_episode_visualization

def main():
    # Environment preset: 15_candidates
    width = 13
    height = 13
    gamma = 0.9
    walls = [
        (0,6),(1,6),(4,6),(5,6),(8,6),(11,6),(12,6),
        (6,0),(6,1),(6,4),(6,5),(6,6),
        (7,6),(7,7),(7,8),(7,9),(7,10),(7,11),(7,12)
    ]
    origin = (10, 2, 0)
    candidate_goals = [
        (0,12), (0,1), (1,0), (0,4), (2,2),
        (4,0), (12,11), (11,12), (12,8), (10,10),
        (8,12), (11,7), (8,10), (6,10), (5,4), (4,5)
    ]
    true_goal = (0,12)  # arbitrarily choose one for init

    # Construct environment
    env = PSRLStandardGridWorld(
        width=width,
        height=height,
        gamma=gamma,
        candidate_goal=true_goal,
        reward_type='binary',
        special_states=None,
        walls=walls,
        origin_state=origin,
        true_goal=true_goal
    )

    # Save visualization
    save_episode_visualization(
        grid_mdp=env,
        trajectory=[(origin, 0.0)],
        save_path="plain_gridworld.png",
        title="Plain GridWorld - 15_Candidates",
        show_legend=True,
        candidate_goals=candidate_goals,
        policy_candidates=None
    )

if __name__ == "__main__":
    main()
