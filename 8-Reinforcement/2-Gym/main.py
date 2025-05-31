import gymnasium as gym
import random
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

env = gym.make("CartPole-v1", render_mode="human")

obs, info = env.reset()

done = False
# for i in range(100):
#     while not done:
#         env.render()
#         obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
#         done = terminated or truncated
#         print(f"{obs} -> {reward}")
#
# env.close()

print(env.observation_space.low)
print(env.observation_space.high)


def discretize(x):
    return tuple((x / np.array([0.25, 0.25, 0.01, 0.1])).astype(int))


def create_bins(i, num):
    return np.arange(num + 1) * (i[1] - i[0]) / num + i[0]


print("Sample bins for interval (-5,5) with 10 bins\n", create_bins((-5, 5), 10))

ints = [(-5, 5), (-2, 2), (-0.5, 0.5), (-2, 2)]  # intervals of values for each parameter
nbins = [20, 20, 10, 10]  # number of bins for each parameter
bins = [create_bins(ints[i], nbins[i]) for i in range(4)]


def discretize_bins(x):
    return tuple(np.digitize(x[i], bins[i]) for i in range(4))



while not done:
    env.render()
    obs, rew, truncated, terminated, info = env.step(env.action_space.sample())
    done = terminated or truncated
    print(discretize_bins(obs))
    # print(discretize(obs))
env.close()

Q = {}
actions = (0, 1)


def qvalues(state):
    return [Q.get((state, a), 0) for a in actions]


# hyperparameters
alpha = 0.3
gamma = 0.9
epsilon = 0.90


def probs(v,eps=1e-4):
    v = v-v.min()+eps
    v = v/v.sum()
    return v

Qmax = 0
cum_rewards = []
rewards = []
for epoch in range(100000):
    obs = env.reset()
    done = False
    cum_reward=0
    # == do the simulation ==
    while not done:
        s = discretize(obs)
        if random.random()<epsilon:
            # exploitation - chose the action according to Q-Table probabilities
            v = probs(np.array(qvalues(s)))
            a = random.choices(actions,weights=v)[0]
        else:
            # exploration - randomly chose the action
            a = np.random.randint(env.action_space.n)

        obs, rew, truncated , terminated,info = env.step(a)
        done = truncated or terminated
        cum_reward+=rew
        ns = discretize(obs)
        Q[(s,a)] = (1 - alpha) * Q.get((s,a),0) + alpha * (rew + gamma * max(qvalues(ns)))
    cum_rewards.append(cum_reward)
    rewards.append(cum_reward)
    # == Periodically print results and calculate average reward ==
    if epoch%5000==0:
        print(f"{epoch}: {np.average(cum_rewards)}, alpha={alpha}, epsilon={epsilon}")
        if np.average(cum_rewards) > Qmax:
            Qmax = np.average(cum_rewards)
            Qbest = Q
        cum_rewards=[]


