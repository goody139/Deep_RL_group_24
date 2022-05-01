import numpy as np
from gridworld import gridworld
import pandas as pd


def visualize(q_table):
    """
    Visualizes the current state.
    """
    actions = ["left", "right", "up", "down"]
    for slice, action in zip(q_table, actions):
        print(action, end=":")
        A = pd.DataFrame(slice)
        A.columns = [''] * A.shape[1]
        print(A.to_string(index=False))
        print()

    print("\n\n\n\n")


def sarsa(grid, n=1, gamma=0.5, alpha=0.9):

    grid = gridworld()
    grid.visualize()

    # n, discount factor, learning rate, min error (stopping condition?)

    # initialize q table
    q_table = np.random.rand(grid.size[0], grid.size[1], 4)  # grid size times 4 possible actions

    # for forever?
    for i in range(1000):
        grid.reset()
        grid.visualize()
        visualize(q_table)



        is_terminal = False
        prev_positions = []
        a_pos = grid.a_pos
        prev_actions = []
        prev_rs = []
        i = 0
        # sample random trajectory
        while not is_terminal:
            # sample random action
            actions = ["left", "right", "up", "down"]
            prev_positions.append(a_pos)

            # chose greedy action
            possible_action_rewards = []
            for a in actions:
                try:
                    m = np.array(grid.moves[a])
                    a_pos = np.array(a_pos)

                    # print(a_pos+m)
                    if -1 in a_pos+m:
                        # print("raise")
                        raise IndexError
                    # print((a_pos+m)[0], (a_pos+m)[1])
                    # print("t\n",[grid.current_state[(a_pos+m)[0], (a_pos+m)[1]], a])
                    if grid.current_state[(a_pos + m)[0], (a_pos + m)[1]] is not None:
                        possible_action_rewards.append([grid.current_state[(a_pos+m)[0], (a_pos+m)[1]], a])
                    # print("pos actions", possible_action_rewards)
                except IndexError:
                    pass
            # print("pos rewards", np.array(possible_action_rewards)[:, 0])
            # print("max", np.max(np.array(np.array(possible_action_rewards)[:, 0], dtype=int)))
            value_col = np.array(np.array(possible_action_rewards)[:, 0], dtype=int)
            max_indices = np.where(value_col == np.max(value_col))

            # print(max_indices[0])
            x_idx=max_indices[0][np.random.randint(0, len(max_indices))]
            # print("x", x_idx)
            print(np.array(possible_action_rewards))
            print("max indices", max_indices)
            print("np.random.randint(0, len(max_indices))", np.random.randint(0, len(max_indices)))
            action = possible_action_rewards[max_indices[0][np.random.randint(0, len(max_indices))]][1]
            # print(action)

            # chose random action
            # action = actions[np.random.randint(0, len(actions))]
            # print(action)
            prev_actions.append(action)

            a_pos, reward, is_terminal = grid.step(action)


            prev_rs.append(reward)
            # print(reward)

            # calculate new q value
            if i >= n:
                new_q = 0
                for ii in range(n):
                    # add reward from prev steps, weighted by gamma
                    new_q += prev_rs[i-(n-ii)] + gamma ** n-(n-ii)
                new_q += q_table[a_pos, np.where(np.array(actions)==action)[0][0]]
                # print("i",i, "n",n, "prev_actions[i-n]",prev_actions[i-n], "np.where(np.array(actions)==prev_actions[i-n])", np.where(np.array(actions)==prev_actions[i-n])[0][0])
                old_q = q_table[prev_positions[i-n], [np.where(np.array(actions)==prev_actions[i-n])[0][0]]]
                # update q value

                q_table[prev_positions[i-n],[np.where(np.array(actions)==prev_actions[i-n])[0][0]]] = old_q + alpha * (new_q-old_q)

            i += 1
            # grid.visualize()






sarsa(gridworld(nb_non_deterministic=0))

