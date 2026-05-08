import csv
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from QLearning import FrozenLake5x5


ACTION_TO_CHAR = {0: "U", 1: "D", 2: "L", 3: "R"}


def generate_trajectories(
    env: FrozenLake5x5,
    policy: np.ndarray,
    n_episodes: int = 10,
    max_steps: int = 200,
    seed: int = 22,
    pause_frames: int = 20,
) -> List[Tuple[int, int, Tuple[int, int], Optional[int], float, bool]]:
    """Generates a list of steps representing trajectories of the given policy."""
    np.random.seed(seed)
    all_steps = []

    for episode in range(n_episodes):
        state = env.reset()
        all_steps.append((episode, 0, state, None, 0.0, False))

        done = False
        step_idx = 0
        action = None
        reward = 0.0

        for step_idx in range(1, max_steps + 1):
            action = int(policy[state[0], state[1]])
            reward, next_state, done = env.step(action)
            all_steps.append((episode, step_idx, next_state, action, reward, done))
            state = next_state
            if done:
                break

        # ---- Pause at end of episode ----
        for _ in range(pause_frames):
            all_steps.append((episode, step_idx, state, action, reward, done))

    return all_steps


def draw_policy_text(ax: plt.Axes, env: FrozenLake5x5, policy: np.ndarray) -> None:
    """Draws policy text (actions or terminal states) on the grid."""
    grid_size = env.n
    for row_idx in range(grid_size):
        for col_idx in range(grid_size):
            cell_char = env.g[row_idx][col_idx]
            if cell_char in "HG":
                text = cell_char
                font_weight = "bold"
                alpha_val = 1.0
            else:
                text = ACTION_TO_CHAR[int(policy[row_idx, col_idx])]
                font_weight = "normal"
                alpha_val = 0.8

            ax.text(
                col_idx,
                row_idx,
                text,
                ha="center",
                va="center",
                fontsize=12,
                color="black",
                alpha=alpha_val,
                fontweight=font_weight,
                zorder=3,
            )


def animate_trajectories(
    env: FrozenLake5x5, 
    steps: List[Tuple], 
    policy: np.ndarray, 
    interval: int = 100
) -> Tuple[plt.Figure, animation.FuncAnimation]:
    """Animates the previously generated trajectories on a grid figure."""
    grid_size = env.n

    grid = np.zeros((grid_size, grid_size))
    for row_idx in range(grid_size):
        for col_idx in range(grid_size):
            cell_char = env.g[row_idx][col_idx]
            grid[row_idx, col_idx] = -1 if cell_char == "H" else (2 if cell_char == "G" else 0)

    fig, ax = plt.subplots()
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)
    ax.set_xticks(np.arange(grid_size + 1) - 0.5)
    ax.set_yticks(np.arange(grid_size + 1) - 0.5)
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.imshow(grid, cmap="coolwarm", vmin=-1, vmax=2, zorder=0)
    draw_policy_text(ax, env, policy)

    ep0, t0, s0, a0, r0, d0 = steps[0]
    agent_dot, = ax.plot([s0[1]], [s0[0]], "ko", markersize=16, zorder=5)

    title = ax.set_title("")
    success_so_far = 0

    counted_episode = -1  # Avoid double counting

    def update(k: int):
        nonlocal success_so_far, counted_episode

        ep, t, s, a, r, done = steps[k]

        # Count success once per episode
        if done and r == 1.0 and ep != counted_episode:
            success_so_far += 1
            counted_episode = ep

        agent_dot.set_data([s[1]], [s[0]])

        a_str = "-" if a is None else ACTION_TO_CHAR[a]
        title.set_text(
            f"Episode {ep + 1} | Step {t} | a={a_str} r={r} | Success so far: {success_so_far}"
        )

        return (agent_dot, title)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(steps),
        interval=interval,
        blit=False,
        repeat=False,
    )

    return fig, ani


def load_policy(file_name: str = "submission_policy.csv") -> np.ndarray:
    """Loads the policy from a CSV file."""
    char_to_action = {"U": 0, "D": 1, "L": 2, "R": 3, "H": 0, "G": 0}
    with open(file_name) as file_obj:
        rows = list(csv.reader(file_obj))

    grid_size = len(rows)
    policy = np.zeros((grid_size, grid_size), dtype=int)

    for row_idx, row in enumerate(rows):
        for col_idx, item in enumerate(row):
            policy[row_idx, col_idx] = char_to_action[item]

    return policy


if __name__ == "__main__":
    frozen_lake_env = FrozenLake5x5()
    
    POLICY_FILE_NAME = "submission_policy.csv"
    read_policy = load_policy(POLICY_FILE_NAME)

    # 2 sec pause: interval=100ms → pause_frames=20
    generated_steps = generate_trajectories(
        frozen_lake_env,
        read_policy,
        n_episodes=100,
        max_steps=200,
        seed=22,
        pause_frames=20,
    )

    figure, anim = animate_trajectories(
        frozen_lake_env, generated_steps, read_policy, interval=100
    )
    plt.show()