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


def q_diff(prev_q, current_q):
    return prev_q - current_q


def sarsa(grid, n=1, gamma=0.5, alpha=0.9):


    # n, discount factor, learning rate, min error (stopping condition?)

    # initialize q table
    q_table = np.random.rand(4, grid.size[0], grid.size[1])  # grid size times 4 possible actions
    prev_q = np.copy(q_table)

    # for forever?
    for i in range(1):
        grid.reset()
        grid.visualize()
        visualize(q_table)
        print(q_table)



        is_terminal = False
        prev_positions = []
        a_pos = grid.a_pos
        prev_actions = []
        prev_rs = []
        i = 0
        t=0
        while not is_terminal and t < 10:
            t+=1
            print("step ", i)
            grid.visualize()
            visualize(q_table)
            # sample random action
            actions = ["left", "right", "up", "down"]
            prev_positions.append(a_pos)

            # chose greedy action
            # possible_action_rewards = []
            # for a in actions:
            #     try:
            #         m = np.array(grid.moves[a])
            #         a_pos = np.array(a_pos)
            #
            #         # print(a_pos+m)
            #         if -1 in a_pos+m:
            #             # print("raise")
            #             raise IndexError
            #         # print((a_pos+m)[0], (a_pos+m)[1])
            #         # print("t\n",[grid.current_state[(a_pos+m)[0], (a_pos+m)[1]], a])
            #         if grid.current_state[(a_pos + m)[0], (a_pos + m)[1]] is not None:
            #             possible_action_rewards.append([grid.current_state[(a_pos+m)[0], (a_pos+m)[1]], a])
            #         # print("pos actions", possible_action_rewards)
            #     except IndexError:
            #         pass
            # # print("pos rewards", np.array(possible_action_rewards)[:, 0])
            # # print("max", np.max(np.array(np.array(possible_action_rewards)[:, 0], dtype=int)))
            # value_col = np.array(np.array(possible_action_rewards)[:, 0], dtype=int)
            # max_indices = np.where(value_col == np.max(value_col))
            #
            # # print(max_indices[0])
            # x_idx=max_indices[0][np.random.randint(0, len(max_indices))]
            # # print("x", x_idx)
            # print(np.array(possible_action_rewards))
            # print("max indices", max_indices)
            # print("np.random.randint(0, len(max_indices))", np.random.randint(0, len(max_indices)))

            max_q = None
            max_idx = None
            for j in range(4):
                v = q_table[a_pos[0], a_pos[1], j]
                if not max_q:
                    max_q = v
                    max_idx = j
                elif v > max_q:
                    max_q = v
                    max_idx = j

            action = actions[max_idx]




            # action = possible_action_rewards[max_indices[0][np.random.randint(0, len(max_indices))]][1]
            print("action chosen in step", i, ":", action)

            # chose random action
            # action = actions[np.random.randint(0, len(actions))]
            # print(action)
            prev_actions.append(action)

            grid.visualize()
            a_pos, reward, is_terminal = grid.step(action)
            grid.visualize()


            prev_rs.append(reward)
            # print(reward)

            # calculate new q value
            if i >= n:
                new_q = 0
                for ii in range(n):
                    # add reward from prev steps, weighted by gamma
                    new_q += prev_rs[i-(n-ii)] * gamma ** (n-(n-ii))
                print("new q idx",np.where(np.array(actions)==action)[0][0], a_pos[0], a_pos[1])
                new_q += q_table[np.where(np.array(actions)==action)[0][0], a_pos[0], a_pos[1]]
                # print("i",i, "n",n, "prev_actions[i-n]",prev_actions[i-n], "np.where(np.array(actions)==prev_actions[i-n])", np.where(np.array(actions)==prev_actions[i-n])[0][0])
                old_q = q_table[[np.where(np.array(actions)==prev_actions[i-n])[0][0]][0], prev_positions[i-n][0], prev_positions[i-n][1]]

                # update q value

                q_table[prev_positions[i-n],[np.where(np.array(actions)==prev_actions[i-n])[0][0]]] = old_q + alpha * (new_q-old_q)
                print("q table index", [np.where(np.array(actions)==prev_actions[i-n])[0][0]][0], prev_positions[i-n][0], prev_positions[i-n][1])
                print(q_table[[np.where(np.array(actions)==prev_actions[i-n])[0][0]][0], prev_positions[i-n][0], prev_positions[i-n][1]])

            i += 1
            # grid.visualize()

            visualize(q_diff(prev_q, q_table))
            prev_q = np.copy(q_table)








sarsa(gridworld(nb_non_deterministic=0))

