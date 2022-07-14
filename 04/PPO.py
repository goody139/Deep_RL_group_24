
import time
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

#####################  hyper parameters  ####################

ENV_NAME = 'CarRacing-v0'     # environment name
ALG_NAME = 'PPO'
RANDOMSEED = 1                # random seed

EP_MAX = 1000                 # total number of episodes for training
EP_LEN = 500                  # total number of steps for each episode
GAMMA = 0.9                   # reward discount
A_LR = 0.0001                 # learning rate for actor
C_LR = 0.001                  # learning rate for critic
BATCH = 64                    # update batchsize

A_UPDATE_STEPS = 10           # actor update steps
C_UPDATE_STEPS = 10           # critic update steps

EPS = 1e-8                    # epsilon

METHOD = dict(name='clip', epsilon=0.2)                                              # choose the method for optimization

# ppo-penalty parameters
KL_TARGET = 0.01
LAM = 0.5

# ppo-clip parameters
EPSILON = 0.2


###############################  PPO  ####################################
class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()

        self.fc1 = tf.keras.layers.Conv2D(name = "c1",kernel_size = (2,2), filters = 10, activation="relu", input_shape = (96, 96, 3))
        self.fc2 = tf.keras.layers.Conv2D(name = "c2", kernel_size = (2,2),filters = 10, activation="relu", kernel_regularizer='l2')
        self.gap = tf.keras.layers.GlobalAveragePooling2D()

        self.out = tf.keras.layers.Dense(name = "c3", units = 3, activation="tanh")

    def call(self, x, training=False):

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.gap(x)

        val = self.out(x)

        return val

class Actor(tf.keras.Model):
    def __init__(self, action_dim, action_bound):
        super(Actor, self).__init__()

        self.action_dim = action_dim
        self.action_bound = action_bound

        self.fc1 = tf.keras.layers.Conv2D(name = "c1",kernel_size = (2,2), filters = 10, activation="relu", input_shape = (1, 96, 96, 3))
        self.fc2 = tf.keras.layers.Conv2D(name = "c2", kernel_size = (2,2),filters = 10, activation="relu", kernel_regularizer='l2')
        self.gap = tf.keras.layers.GlobalAveragePooling2D()

        self.a = tf.keras.layers.Dense(name = "c3", units = action_dim, activation="tanh")
        self.sigma = tf.keras.layers.Dense(name = "c4",  units = action_dim, activation="softplus")

    def call(self, x, training=False):

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.gap(x)

        mu = self.a(x)
        sig = self.sigma(x)

        return mu, sig


class PPO:
    def __init__(self, action_dim, action_bound, method='clip'):

        self.critic = Critic()

        self.actor = Actor(action_dim, action_bound)
        self.actor_old = Actor(action_dim, action_bound)

        self.actor_opt = tf.keras.optimizers.Adam(A_LR)
        self.critic_opt = tf.keras.optimizers.Adam(C_LR)

        self.state_buffer, self.action_buffer = [], []
        self.reward_buffer, self.cumulative_reward_buffer = [], []
        self.new_state = []

        self.action_bound = action_bound

    @tf.function
    def choose_action(self, s):
        '''
        Choose action
        :param s: state
        :return: clipped act
        '''

        s = tf.convert_to_tensor([s], dtype=tf.float32)

        mu, sigma = self.actor(s)

        pi = tfp.distributions.Normal(mu, sigma)

        a = tf.squeeze(pi.sample(1), axis=0)[0]
        
        return a


    def store_transition(self, state, action, reward, new_state):
        """
        Store state, action, reward at each step
        :param state:
        :param action:
        :param reward:
        :return: None
        """
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.new_state.append(new_state)

    @tf.function
    def a_train(self, state, action, new_state, reward):
        '''
        update the policy network
        '''

        state = tf.convert_to_tensor(state, np.float32)
        action = tf.convert_to_tensor(action, np.float32)
        state_ = tf.convert_to_tensor(new_state, np.float32)
        reward = tf.convert_to_tensor(reward, np.float32)


        with tf.GradientTape() as tape:

            adv = reward + self.critic(state_) - self.critic(state)

            mu, sigma = self.actor(state, training=True)
            pi = tfp.distributions.Normal(mu, sigma)

            mu_old, sigma_old = self.actor_old(state, training=True)
            oldpi = tfp.distributions.Normal(mu_old, sigma_old)

            ratio = pi.prob(action) / (oldpi.prob(action) + EPS)

            surr = ratio * adv

            aloss = -tf.reduce_mean(
                tf.minimum(surr, tf.clip_by_value(
                    ratio, 1. - METHOD['epsilon'], 1. + METHOD['epsilon']) * adv)
            )

        a_gard = tape.gradient(aloss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_gard, self.actor.trainable_weights))


    def update_old_pi(self):
        '''
        更新actor_old参数。
        '''
        for pi, oldpi in zip(self.actor.trainable_weights, self.actor_old.trainable_weights):
            oldpi.assign(pi)

    @tf.function
    def c_train(self, reward, state):
        """
        update the critic network
        """

        reward = tf.convert_to_tensor(reward, dtype=np.float32)

        with tf.GradientTape() as tape:
            # calculate the td-error
            advantage = reward - self.critic(state)
            loss = tf.reduce_mean(tf.square(advantage))

        grad = tape.gradient(loss, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(grad, self.critic.trainable_weights))


    def update(self):
        '''
        '''
        s = np.array(self.state_buffer, np.float32)
        a = np.array(self.action_buffer, np.float32)
        r = np.array(self.cumulative_reward_buffer, np.float32)
        n = np.array(self.new_state, np.float32)


        # adv = (adv - adv.mean())/(adv.std()+1e-6)  # sometimes helpful


        for _ in range(A_UPDATE_STEPS):
            self.a_train(s, a, n, r)

        # update critic
        for _ in range(C_UPDATE_STEPS):
            self.c_train(r, s)

        self.update_old_pi()

        self.state_buffer.clear()
        self.action_buffer.clear()
        self.cumulative_reward_buffer.clear()
        self.reward_buffer.clear()
        self.new_state.clear()


    def finish_path(self, next_state, done):
        """
        Calculate cumulative reward
        :param next_state:
        :return: None
        """
        if done: v_s_ = 0
        else: v_s_ = self.critic(np.array([next_state], dtype=np.float32))[0, 0]
        discounted_r = []
        for r in self.reward_buffer[::-1]:
            v_s_ = r + GAMMA * v_s_
            discounted_r.append(v_s_)
        discounted_r.reverse()
        discounted_r = np.array(discounted_r)[:, np.newaxis]
        self.cumulative_reward_buffer.extend(discounted_r)
        self.reward_buffer.clear()




if __name__ == '__main__':

    env = gym.make(ENV_NAME).unwrapped

    # reproducible
    env.seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)
    tf.random.set_seed(RANDOMSEED)

    ppo = PPO(
        action_dim = env.action_space.shape[0],
        action_bound = env.action_space.high,
    )


    all_ep_r = []

    RENDER = 0
    for episode in range(EP_MAX):

        state = env.reset()
        buffer_s, buffer_a, buffer_r = [], [], []
        episode_reward = 0
        t0 = time.time()

        if episode % 25 == 0:
            RENDER = 1

        for t in range(EP_LEN):
            if RENDER:
                env.render()
        
            action = ppo.choose_action(state)

            state_, reward, done, _ = env.step(action.numpy())

            ppo.store_transition(state, action, reward, state_)

            state = state_

            episode_reward += reward
            if done:
                break

            if (t + 1) % BATCH == 0 or t == EP_LEN - 1:
                ppo.finish_path(state_, done)
                ppo.update()

        if episode == 0:
            all_ep_r.append(episode_reward)
        else:
            all_ep_r.append(all_ep_r[-1] * 0.9 + episode_reward * 0.1)
        print(
            'Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                episode, EP_MAX, episode_reward,
                time.time() - t0
            )
        )

        RENDER = 0

    ppo.save_ckpt()
    plt.plot(all_ep_r)
    plt.show()
