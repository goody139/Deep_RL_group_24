import numpy as np
import pandas as pd
from gridworld import gridworld


def visualize(q_table, actions = ["left", "right", "up", "down"]):
    """
    Visualizes the current state.
    """

    for slice, action in zip(q_table, actions):
        print(action, end=":")
        A = pd.DataFrame(slice)
        A.columns = [''] * A.shape[1]
        print(A.to_string(index=False))
        print()

    print("\n\n\n\n")

def q_diff(prev_q, current_q):
    return prev_q - current_q



def sarsa(grid, n, gamma, alpha, epsilon):
    actions = ["left", "right", "up", "down"]

    # initialize random q table
    q_table = np.random.rand(4, grid.size[0], grid.size[1])
    agent_x, agent_y = grid.a_pos
    old_q_table = np.copy(q_table)

    # sample many trajectories
    for sample in range(2000):
        is_terminal = False
        grid.visualize()
        grid.reset()
        grid.visualize()
        visualize(q_table)

        print("difference")
        visualize(q_diff(old_q_table, q_table))
        old_q_table = np.copy(q_table)

        # until the terminal state is reached
        while not is_terminal:

            # sample action according to policy - maximize q value
            if np.random.random() < (1 - epsilon):
                q_values = q_table[:, agent_x, agent_y]
                max_index = np.argmax(q_values)
                action = actions[max_index]
            else:
                action = actions[np.random.randint(0, len(actions))]

            # perform step to get reward
            agent_pos, reward, is_terminal = grid.step(action)
            agent_x, agent_y = agent_pos

            state_grid = np.copy(grid.current_state)
            state_a_pos = agent_pos
            state_action = action

            # calculate expected reward
            q_new = reward  # k=0

            for k in range(1, n):  # k=1 to n-1
                # sample action according to policy - maximize q value
                if np.random.random() < (1 - epsilon):
                    q_values = q_table[:, agent_x, agent_y]
                    max_index = np.argmax(q_values)
                    action = actions[max_index]
                else:
                    action = actions[np.random.randint(0, len(actions))]


                # perform step to get reward
                agent_pos, reward, is_terminal = grid.step(action)
                agent_x, agent_y = agent_pos

                q_new += reward * (gamma**k)


            # sample action according to policy - maximize q value
            if np.random.random() < (1 - epsilon):
                q_values = q_table[:, agent_x, agent_y]
                max_index = np.argmax(q_values)
                action = actions[max_index]
            else:
                action = actions[np.random.randint(0, len(actions))]

            # add q value as estimation for the rest of the path
            print("action", action)
            action_idx = np.where(np.array(actions) == action)[0][0]
            q_new += q_table[action_idx, agent_x, agent_y] * (gamma**n)

            # reset grid and agent position to actual current state
            agent_pos = state_a_pos
            agent_x, agent_y = agent_pos
            grid.set_to_state(np.copy(state_grid), state_a_pos)
            # print("indices", action_idx, agent_x, agent_y)

            # update q value
            action_idx = np.where(np.array(actions) == action)[0][0]
            q_old = q_table[action_idx, agent_x, agent_y]
            # print("q to update", q_table[action_idx, agent_x, agent_y])
            # print("updating to ", q_old + alpha * (q_new - q_old))
            q_table[action_idx, agent_x, agent_y] = q_old + alpha * (q_new - q_old)

    return q_table

def apply_policy(grid, q_table, epsilon=0.1):
    actions = ["left", "right", "up", "down"]
    agent_x, agent_y = grid.a_pos

    is_terminal = False
    while not is_terminal:
        grid.visualize()
        if np.random.random() < (1 - epsilon):
            q_values = q_table[:, agent_x, agent_y]
            max_index = np.argmax(q_values)
            action = actions[max_index]
        else:
            action = actions[np.random.randint(0, len(actions))]

        agent_pos, _, _ = grid.step(action)
        agent_x, agent_y = agent_pos



g = gridworld()
q = sarsa(g, n=1, gamma=0.5, alpha=0.5, epsilon=0.1)
g.visualize()
g.reset()
g.visualize()
visualize(q)



print("applying policy")
apply_policy(g, q)

















