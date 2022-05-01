import numpy as np
import random
import pandas as pd


def manhattan_dist(x, y):
    """
    Calculates the manhattan or cityblock distance between two 2d points.

    :param x: 2d point
    :param y: 2d point
    :return: manthattan distance between x and y
    """
    return np.abs(x[0] - y[0]) + np.abs(x[1] - y[1])


def max_manhattan_dist(x, size):
    """
    Calculates the manhattan or cityblock distance between two 2d points.

    :param x: 2d point
    :param y: 2d point
    :return: manthattan distance between x and y
    """
    return np.max(manhattan_dist(x, (0, 0)), manhattan_dist(x, (0, size[1])), manhattan_dist(x, (size[0], 0)),
                  manhattan_dist(x, size))


class gridworld():

    def __init__(self, size=(5, 5), agent_start=(0, 0), terminal_pos=None, nb_blocked=None, nb_negative=None,
                 nb_non_deterministic=None, template_grid=None, pos_reward=10, neg_reward=-5):
        # check if the values are valid
        if len(size) != 2 or size[0] < 2 or size[1] < 2:
            raise ValueError("grid must be two dimensional and have at least have size (2,2)")

        if agent_start and (
                len(agent_start) != 2 or not 0 <= agent_start[0] < size[0] or not 0 <= agent_start[1] < size[1]):
            raise ValueError("agent_start must be a 2d point withing the grid")

        if terminal_pos and (
                len(terminal_pos) != 2 or not 0 <= terminal_pos[0] < size[0] or not 0 <= terminal_pos[1] < size[1]):
            raise ValueError("terminal_pos must be a 2d point withing the grid")

        if nb_blocked and (
                (agent_start and terminal_pos and nb_blocked >= size[0] * size[1] - manhattan_dist(terminal_pos,
                                                                                                   agent_start))
                or (agent_start and not terminal_pos and nb_blocked >= size[0] * size[1] - max_manhattan_dist(
            agent_start, size))
                or (terminal_pos and not agent_start and nb_blocked >= size[0] * size[1] - max_manhattan_dist(
            terminal_pos, size))
                or (not agent_start and not terminal_pos and nb_blocked >= size[0] * size[1] - size[0] - size[1])
        ):
            raise ValueError("too many points blocked, there need to be enough free points to reach the terminal state")

        # optional checks to fulfill task requirements of the gridworld
        if agent_start and terminal_pos and agent_start[0] == terminal_pos[0] and agent_start[1] == terminal_pos[1]:
            raise ValueError("agent_start and terminal_pos have to be different points")
        if nb_negative and nb_negative < 2:
            raise ValueError("there should be at least two tiles with a negative reward")
        if nb_non_deterministic and nb_non_deterministic < 2:
            raise ValueError("there should be at least 2 tiles with a non-deterministic state transition function")
        if pos_reward < 1:
            raise ValueError("positive reward must be positive")
        if neg_reward > -1:
            raise ValueError("negative reward must be negative")

        # initialize according to the parameters
        self.size = size
        self.fields = size[0] * size[1]
        self.current_state = None
        self.original_grid = None
        self.original_a_pos = None
        self.a_pos = None
        self.t_pos = None

        # all possible movements
        self.moves = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

        if template_grid:
            # TODO agent pos
            self.original_grid = template_grid
            self.current_state = template_grid

        else:
            print("no template")
            # create array of size size
            grid = np.zeros(shape=size, dtype=object)

            # place agent
            if agent_start:
                grid[agent_start] = "A"
                self.a_pos = agent_start
            else:
                # select random start point
                print("creating start")
                while True:
                    (x, y) = (np.random.randint(0, self.size[0]), np.random.randint(0, self.size[1]))
                    if not terminal_pos or terminal_pos[0] != x or terminal_pos[1] != y:
                        break

                grid[x, y] = "A"
                self.a_pos = (x, y)

            # place terminal state
            if terminal_pos:
                grid[terminal_pos] = pos_reward
                self.t_pos = terminal_pos
            else:
                # select random position for terminal state
                print("creating end")
                while True:
                    (x, y) = (np.random.randint(0, self.size[0]), np.random.randint(0, self.size[1]))
                    if self.a_pos[0] != x or self.a_pos[1] != y:
                        break
                grid[x, y] = pos_reward
                self.t_pos = (x, y)

            # create a path according to the number of blocked parts
            print("start blocking")

            # generate number to be blocked if not given
            min_path_len = manhattan_dist(self.a_pos, self.t_pos)

            if not nb_blocked:
                nb_blocked = np.random.randint(0, self.fields - min_path_len)

            max_path_len = self.fields - nb_blocked
            path_len = np.random.randint(min_path_len, max_path_len + 1)

            current_x = self.a_pos[0]
            current_y = self.a_pos[1]
            goal_x = self.t_pos[0]
            goal_y = self.t_pos[1]
            remaining_steps = path_len
            delta_x = goal_x - current_x
            delta_y = goal_y - current_y

            print("current pos:", current_x, current_y)
            print("agent:", self.a_pos[0], self.a_pos[1])
            print("goal:", goal_x, goal_y)

            print("grid\n", grid)
            end_reached = False
            while current_x != goal_x or current_y != goal_y:
                print()
                print("while")
                print("steps:", remaining_steps)

                # movement options for next step
                m = []

                # check whether the path needs to be direct
                print("abs delta:", np.abs(delta_x) + np.abs(delta_y))
                if np.abs(delta_x) + np.abs(delta_y) <= remaining_steps +1:
                    print("restricting movements")
                    # path needs to be direct, only allow direct movements towards the goal
                    if delta_x > 0:
                        m.append((self.moves["down"]))
                    if delta_x < 0:
                        m.append((self.moves["up"]))
                    if delta_y > 0:
                        m.append((self.moves["right"]))
                    if delta_y < 0:
                        m.append((self.moves["left"]))

                    m = np.array(m)

                else:
                    # all movements are allowed except when at the edge of the grid
                    m.append((self.moves["up"]))
                    m.append((self.moves["down"]))
                    m.append((self.moves["left"]))
                    m.append((self.moves["right"]))

                    rm_indices = []

                    if current_x == 0:
                        rm_indices.append(0)  # remove up
                    if current_x == self.size[0] - 1:
                        rm_indices.append(1)  # remove down
                    if current_y == 0:
                        rm_indices.append(2)  # remove left
                    if current_y == self.size[1] - 1:
                        rm_indices.append(3)  # remove right

                    m = np.array(m)
                    mask = np.ones(len(m), bool)
                    mask[rm_indices] = 0
                    m = m[mask]

                # sample random movement
                print("m", m)
                move = m[np.random.randint(0, len(m))]
                #print("move", move)
                current_x += move[0]
                current_y += move[1]
                delta_x = goal_x - current_x
                delta_y = goal_y - current_y
                remaining_steps -= 1

                print("current pos:", current_x, current_y)
                print("agent:", self.a_pos[0], self.a_pos[1])
                print("goal:", goal_x, goal_y)
                print("delta:", delta_x, delta_y)

                # set path to "p" - except agent start and terminal state
                if not (current_x == self.a_pos[0] and current_y == self.a_pos[1]) \
                        and not (current_x == self.t_pos[0] and current_y == self.t_pos[1]):
                    grid[current_x, current_y] = "p"
                elif current_x == self.t_pos[0] and current_y == self.t_pos[1]:
                    print("on end", current_x, current_y)
                    end_reached = True
                else:
                    print("on start", current_x, current_y)

                vis(grid)

                if remaining_steps == 0:
                    print("done walking, goal reached:", end_reached)

            # randomly block fields that are not path, start, or end
            available_to_block = np.where(grid == 0)
            print("available len", len(available_to_block[0]), "nb blocked", nb_blocked)
            to_block = random.sample(range(len(available_to_block[0])), nb_blocked)
            grid[available_to_block[0][to_block], available_to_block[1][to_block]] = None
            print("end blocking")

            # reset path to 0
            grid[np.where(grid == "p")] = 0


            # randomly set negative rewards
            available_to_set = np.where(grid == 0)
            if not nb_negative:
                nb_negative = np.random.randint(0, len(available_to_set[0]))
            to_set = random.sample(range(len(available_to_set[0])), nb_negative)
            grid[available_to_set[0][to_set], available_to_set[1][to_set]] = neg_reward


            # randomly chose which states have s non-deterministic state transition
            self.non_deterministic = np.zeros(self.size)
            available_to_set = np.where(grid != None) # leave as !=, do NOT ude is not, does not wok that way and will produce an error
            if not nb_non_deterministic:
                nb_non_deterministic = np.random.randint(0, len(available_to_set[0]))
            to_set = random.sample(range(len(available_to_set[0])), nb_non_deterministic)
            self.non_deterministic[available_to_set[0][to_set], available_to_set[1][to_set]] = 1


            self.current_state = grid
            self.original_grid = np.copy(grid)
            self.original_a_pos = self.a_pos



    def reset(self):
        """
            Resets the gridworld to its initial state.

            :return: The initial state of the grid.
            """
        self.current_state = np.copy(self.original_grid)
        self.a_pos = self.original_a_pos

        return np.copy(self.original_grid)

    def non_deterministic_transition(self, possible_actions, action):

        # chose action 0.8 percent of the time, else randomly chose another state
        other_actions = possible_actions[np.where(possible_actions!=action)]
        if np.random.random(1) < 0.8:
            return action
        else:
            return other_actions[np.random.randint(0, len(other_actions))]


    def step(self, action):
        """
        Applies the state transition dynamics and reward dynamics based on the state of the environment and the action
        argument.

        :param action: String indictating the agent movement, one of "left", "right", "up" "down"
        :return: (1) The agent position (should be new state?), (2) the reward of this step, (3) a boolean indicating whether this state is terminal.
        """
        current_x = self.a_pos[0]
        current_y = self.a_pos[1]

        # check if action is legal
        # try:
        #     s = self.current_state[current_x+self.moves[action][0], current_y+self.moves[action][1]]
        #     if s is None:
        #         raise ValueError("Action cannot be taken, the path is blocked")
        # except IndexError:
        #     raise ValueError("Action cannot be taken as it would lead off the grid")

        # possible moves for this state -- !!!!!!!!!!!!!!!!!!!!!!!!!!!! check if the path is blocked
        m = []
        # all movements are allowed except when at the edge of the grid
        m.append((self.moves["up"]))
        m.append((self.moves["down"]))
        m.append((self.moves["left"]))
        m.append((self.moves["right"]))


        # rm_indices = []
        #
        # for i, mv in enumerate(m):
        #     try:
        #         s = self.current_state[current_x+mv[0], current_y+mv[1]]
        #         if s is None:
        #             rm_indices.append(i)
        #     except IndexError:
        #         rm_indices.append(i)
        #
        # m = np.array(m)
        # mask = np.ones(len(m), bool)
        # mask[rm_indices] = 0
        # m = m[mask]

        # define non-deterministic state transition function
        if self.non_deterministic[current_x, current_y]:
            # movements other that the currently chosen action
            m = np.array(m)
            m = np.delete(m, np.where(m==self.moves[action]))

            move = self.non_deterministic_transition(m, self.moves[action])
        else:
            move = self.moves[action]

        # set the current position to its original value / 0 if it is the agent's start
        value = self.original_grid[current_x, current_y]
        if type(value) is not int:
            value = 0

        # print("value", value, "setting at:", current_x, current_y)
        self.current_state[current_x, current_y] = value

        # check if movement is legal, if not just stay at the current position
        try:
            s = self.current_state[current_x+move[0], current_y+move[1]]
            if s is None or current_x+move[0] < 0 or current_y+move[1] < 0:
                move = (0, 0)
        except IndexError:
            move = (0, 0)

        # determine reward
        current_x += move[0]
        current_y += move[1]

        reward = self.current_state[current_x, current_y]
        if move == (0, 0):
            reward = 0

        # update agent position
        # print("current values:",current_x, current_y)
        self.current_state[current_x, current_y] = "A"
        self.a_pos = (current_x, current_y)

        is_terminal = False
        if current_x == self.t_pos[0] and current_y == self.t_pos[1]:
            is_terminal = True


        # print("reward", reward, "move", move, "current pos", current_x, current_y, "apos", self.a_pos)
        return self.a_pos, reward, is_terminal

    def visualize(self):
        """
        Visualizes the current state.
        """
        A = pd.DataFrame(self.current_state)
        A.columns = [''] * A.shape[1]
        print(A.to_string(index=False))
        # with np.printoptions(np.set_printoptions(formatter={'float': '{: 0.3f}'.format})):
        #     print(self.current_state)


def vis(d):
    A = pd.DataFrame(d)
    A.columns = [''] * A.shape[1]
    print(A.to_string(index=False))

g = gridworld()
g.visualize()
print("\n\n\n")
print("orig",g.original_grid)
print("right")
g.step("right")
print("orig",g.original_grid)
g.visualize()
print()

print("right")
g.step("right")
print("orig",g.original_grid)
g.visualize()
print()

print("left")
g.step("left")
g.visualize()
print()

g.reset()
print("down")
g.step("down")
g.visualize()
