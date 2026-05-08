import csv
from typing import List, Tuple

import numpy as np

# #################################################################
#  Don't change the code below
# ##################################################################

# 0=U, 1=D, 2=L, 3=R
ACTION_TO_CHAR = {0: "U", 1: "D", 2: "L", 3: "R"}


class FrozenLake5x5:
    """A 5x5 Frozen Lake environment."""

    def __init__(self) -> None:
        # Start in the middle; goal at bottom-right; a few holes
        self.g: List[str] = [
            "SFFFF",
            "FHFFH",
            "FFFFH",
            "FFHFF",
            "HFFFG",
        ]
        self.n: int = 5
        self.num_actions: int = 4  # UP, DOWN, LEFT, RIGHT
        self.step_number: int = 0
        self.d: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.s: Tuple[int, int]
        self.reset()

    def reset(self) -> Tuple[int, int]:
        """Resets the environment to the initial state."""
        self.s = (0, 0)
        self.step_number = 0
        return self.s

    def _move(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Calculates the next state given a state and an action."""
        row, col = state
        d_row, d_col = self.d[int(action)]
        next_row, next_col = row + d_row, col + d_col

        if 0 <= next_row < self.n and 0 <= next_col < self.n:
            return (next_row, next_col)
        return state

    def step(self, action: int) -> Tuple[float, Tuple[int, int], bool]:
        """
        Takes a step in the environment.

        Args:
            action (int): The action to take (0=Up, 1=Down, 2=Left, 3=Right).

        Returns:
            Tuple[float, Tuple[int, int], bool]: A tuple containing the reward,
            the next state, and a boolean indicating if the episode is done.
        """
        row, col = self.s
        if self.g[row][col] in "HG":
            return 0.0, self.s, True

        # Up: up 3/4, left/right 1/8 each
        # Down/Left/Right: intended 1/2, perpendicular 1/4 each
        if action == 0:
            choices = [(0, 0.75), (2, 0.125), (3, 0.125)]
        elif action == 1:
            choices = [(1, 0.4), (2, 0.3), (3, 0.3)]
        elif action == 2:
            choices = [(2, 0.4), (0, 0.3), (1, 0.3)]
        else:
            choices = [(3, 0.4), (0, 0.3), (1, 0.3)]

        actions, probabilities = zip(*choices)
        chosen_action = np.random.choice(actions, p=probabilities)

        next_state = self._move(self.s, chosen_action)
        self.s = next_state

        cell = self.g[next_state[0]][next_state[1]]
        reward = 1.0 if cell == "G" else 0.0

        done = cell in "HG" or self.step_number >= 200
        self.step_number += 1

        return reward, next_state, done


def save_policy(env: FrozenLake5x5, policy: np.ndarray, file_name: str) -> None:
    """Saves the policy to a CSV file."""
    with open(file_name, "w") as file_writer:
        for row_idx in range(env.n):
            row_data = []
            for col_idx in range(env.n):
                cell_char = env.g[row_idx][col_idx]
                if cell_char in "HG":
                    row_data.append(cell_char)
                else:
                    action_idx = int(policy[row_idx, col_idx])
                    row_data.append(ACTION_TO_CHAR[action_idx])
            file_writer.write(",".join(row_data) + "\n")


def eval_policy(
    file_name: str, env: FrozenLake5x5, episodes: int = 50_000, max_steps: int = 200, seed: int = 22
) -> Tuple[float, float, float]:
    """Evaluates a policy loaded from a CSV file."""
    char_to_action = {"U": 0, "D": 1, "L": 2, "R": 3, "H": 0, "G": 0}

    # ---- Load policy as 2D array ----
    with open(file_name, newline="") as file_reader:
        rows = list(csv.reader(file_reader))

    grid_size = len(rows)
    policy = np.zeros((grid_size, grid_size), dtype=int)

    for row_idx, row in enumerate(rows):
        for col_idx, cell in enumerate(row):
            policy[row_idx, col_idx] = char_to_action[cell]
            print(cell, end=", ")
        print()

    np.random.seed(seed)
    successes = 0
    failures = 0
    incompletes = 0

    for _ in range(episodes):
        state = env.reset()

        for _ in range(max_steps):
            action = policy[state[0], state[1]]  # Direct 2D indexing
            reward, state, done = env.step(action)

            if done:
                if reward == 1.0:
                    successes += 1
                else:
                    failures += 1
                break
        else:
            incompletes += 1

    return successes / episodes, failures / episodes, incompletes / episodes


# #################################################################
#  Don't change the code above
# ##################################################################


def q_learning(env: FrozenLake5x5, max_steps: int = 200) -> np.ndarray:
    """
    Performs Q-Learning to find an optimal policy.

    Environment Interactions:
    env.n = 5  # number of rows == number of cols
    env.reset()  # returns (0, 0), start state
    env.step(action)  # returns reward, next_state, done (True or False)

    Parameters for Q-Learning:
    alpha, epsilon, gamma, Q-table initialization, seed_value, etc.

    Returns:
        np.ndarray: Policy 2D array of size 5x5. Each value is an action in {0, 1, 2, 3}.
                    Example: policy[1, 2] contains the action at second row, third column.
    """
    episodes = 30_000
    print(f"Num Episodes: {episodes}, Max Steps: {max_steps}")

    # You may change this seed number.
    random_seed = 1122
    np.random.seed(random_seed)

    grid_size = env.n
    num_actions = env.num_actions  # UP, DOWN, LEFT, RIGHT

    # Initial random policy (will be replaced later)
    policy = np.random.randint(0, num_actions, size=(grid_size, grid_size), dtype=int)

    # ###############################################
    # YOUR IMPLEMENTATION GOES HERE.
    q_table = np.zeros((grid_size, grid_size, num_actions))

    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.99995

    for _ in range(episodes):
        state = env.reset()

        for _ in range(max_steps):
            row, col = state

            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.randint(0, num_actions)
            else:
                action = int(np.argmax(q_table[row, col]))

            reward, next_state, done = env.step(action)
            next_row, next_col = next_state

            # Q-learning target and update
            if done and env.g[next_row][next_col] in "HG":
                td_target = reward
            else:
                best_next_action = np.argmax(q_table[next_row, next_col])
                td_target = reward + gamma * q_table[next_row, next_col, best_next_action]

            td_error = td_target - q_table[row, col, action]
            q_table[row, col, action] += alpha * td_error

            state = next_state

            if done:
                break

        # Decay epsilon over episodes
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    # At the end:
    # policy = greedy policy derived from learned Q-table
    for r in range(grid_size):
        for c in range(grid_size):
            policy[r, c] = int(np.argmax(q_table[r, c]))
    # ###############################################

    return policy


if __name__ == "__main__":
    POLICY_FILE_NAME = "submission_policy.csv"
    frozen_lake_env = FrozenLake5x5()

    learned_policy = q_learning(frozen_lake_env, max_steps=200)

    print("Policy grid (U/D/L/R, terminals shown as H/G):")
    save_policy(frozen_lake_env, learned_policy, POLICY_FILE_NAME)

    success_rate, failure_rate, incomplete_rate = eval_policy(
        POLICY_FILE_NAME, frozen_lake_env, episodes=10_000, max_steps=200, seed=24
    )

    print(f"\nsuccess_rate        = {success_rate}")
    print(f"failure_rate        = {failure_rate}")
    print(f"did_not_finish_rate = {incomplete_rate}")
