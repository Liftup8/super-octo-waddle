<<<<<<< HEAD
import gym
gym.logger.set_level(40)
import random
import torch
import numpy as np
import gym_dubins_airplane
from collections import deque
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import random
import time
RenderOpt = True  # enabling render
# Creates environment

env = gym.make('dubinsAC2D-v0')
env.seed(0)

from agent import Agent

agent = Agent(state_size=8, action_size=15, seed=0)


def dqn(n_episodes=10000,
        max_t=1000,
        eps_start=1.0,
        eps_end=0.001,
        eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of training timesteps per episode, indicates the maximum number actions in one episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    score_mean = []
    scores_window = deque(maxlen=100)
    eps = eps_start  # initialize epsilon

    start_time = time.time()

    for i_episode in range(1, n_episodes + 1):
        # episode loop
        # get current state from environment
        state = env.reset()
        score = 0  # score at the start of new episode
        damage_red = 0  # damage dealt to red AC at the start of new episode
        # play a sequence (1 episode)
        for t in range(max_t):
            # training time step loop
            # select the action for each state
            action = agent.act(state, eps)
            # execute action, get reward, new state and whether the sequence can be continued
            # env.step(action): Step the environment by one action timestep. Return
            # observation, reward and done
            if RenderOpt:
                env.render()
                time.sleep(0.01)
            next_state, reward_t, reward_s, done, damage, _ = env.step(action)
            reward = reward_t + reward_s
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward  # total score in current episode (scalar + potential for current time)
            damage_red += damage  # total damage on red aircraft in current episode
            if damage_red == 3:  # health bar implementation on red AC
                done = True
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        score_mean.append(np.mean(scores_window))
        eps = max(eps_end, eps_decay * eps)
        if damage_red == 3:
            print("Blue wins! Dominated!")
            print(
                '\rEpisode {}\tReward in Episode: {:.5f} \tAverage Score {:.5f}\n\n'
                .format(i_episode, score, np.mean(scores_window)),
                end="")
        else:
            print(
                '\rEpisode {}\tReward in Episode: {:.5f} \tAverage Score {:.5f}\n\n'
                .format(i_episode, score, np.mean(scores_window)),
                end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_window)))
            elapsed_time = time.time() - start_time
            print("Duration: ", elapsed_time)
        if np.mean(scores_window) >= 10 and i_episode >= 100:
            print(
                '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'
                .format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    elapsed_time = time.time() - start_time
    print("Training duration: ", elapsed_time)
    return score_mean

    env.close()


score_mean = dqn()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(score_mean)
plt.ylabel('Average Reward')
plt.xlabel('Episode #')
plt.savefig('training_result.png')
=======
<<<<<<< HEAD
import gym
gym.logger.set_level(40)
import random
import torch
import numpy as np
import gym_dubins_airplane
from collections import deque
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import random
import time
RenderOpt = False  # enabling render
# Creates environment

# env = gym.make('LunarLander-v2')
env = gym.make('dubinsAC2D-v0')
env.seed(0)

print('State shape:',env.observation_space.shape)
print('Number of actions:',env.action_space.n)

from state_norm import state_norm
from config import Config

from agent import Agent

agent = Agent(state_size=8, action_size=Config.action_size, seed=0)


def dqn(n_episodes=20000,
        max_t=1000,
        eps_start=1.0,
        eps_end=0.001,
        eps_decay=0.998):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of training timesteps per episode, indicates the maximum number actions in one episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    score_mean=[]
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon

    start_time = time.time()

    for i_episode in range(1, n_episodes + 1):  # episode loop
        # get current state from environment
        state = env.reset()  # It returns an initial observation (random)
        score = 0  # score at the start of new episode
        damage_red = 0  # damage dealt to red AC at the start of new episode
        if i_episode % 100 == 0:
          print('\nEpsilon in episode {}: {:.5f}'.format(i_episode, eps))        
        # play a sequence (1 episode)
        for t in range(max_t):  # training time step loop
            # select the action for each state
            action = agent.act(state_norm(state), eps)
            # execute action, get reward, new state and whether the sequence can be continued
            # env.step(action): Step the environment by one action timestep. Return
            # observation, reward and done
            if RenderOpt:
                env.render()  # will display a popup window
                time.sleep(0.01)
            next_state, reward_t, reward_s, done, damage = env.step(action)
            reward = reward_t + reward_s
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward  # total score in current episode (scalar + potential for current time)
            damage_red += damage  # total damage on red aircraft in current episode
            if damage_red == 3:  # health bar implementation on red AC
                done = True
            if done:
                break
        scores_window.append(
            score
        )  # save most recent score, adds collected reward at the end of episode
        scores.append(score)  # save most recent score
        score_mean.append(np.mean(scores_window))
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon at each episode
        # if damage_red == 3:
            # print("Blue wins! Dominated!")
            # print('\rEpisode {}\tReward in Episode: {:.5f} \tAverage Score {:.5f}\n\n'.format(i_episode, score, np.mean(scores_window)),end="")
            # print('-----------------------------------------------------')
        # else:
            # print('\rEpisode {}\tReward in Episode: {:.5f} \tAverage Score {:.5f}\n\n'.format(i_episode, score, np.mean(scores_window)),end="")
            # print('-----------------------------------------------------')
        # if i_episode % 100 == 0:
            # print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            # elapsed_time = time.time() - start_time
            # print("Duration: ", elapsed_time)
        # if np.mean(scores_window) >= 10 and i_episode >= 100:
            # print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            # torch.save(agent.qnetwork_local.state_dict(),'checkpoint.pth')  # Function to save model parameters
            # break
    torch.save(agent.qnetwork_local.state_dict(),'checkpoint.pth')  # Function to save model parameters
    elapsed_time = time.time() - start_time
    print("Training duration: ", elapsed_time)
    return score_mean

    env.close()


score_mean = dqn()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(score_mean)
plt.ylabel('Average Reward')
plt.xlabel('Episode #')
plt.savefig('training_result.png')
=======
import gym
gym.logger.set_level(40)
import random
import torch
import numpy as np
import gym_dubins_airplane
from collections import deque
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import random
import time
RenderOpt = True  # enabling render
# Creates environment

env = gym.make('dubinsAC2D-v0')
env.seed(0)

from agent import Agent

agent = Agent(state_size=8, action_size=15, seed=0)


def dqn(n_episodes=10000,
        max_t=1000,
        eps_start=1.0,
        eps_end=0.001,
        eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of training timesteps per episode, indicates the maximum number actions in one episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    score_mean = []
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon

    start_time = time.time()

    for i_episode in range(1, n_episodes + 1):
        # episode loop
        # get current state from environment
        state = env.reset()  # It returns an initial observation (random)
        score = 0  # score at the start of new episode
        damage_red = 0  # damage dealt to red AC at the start of new episode
        # play a sequence (1 episode)
        for t in range(max_t):
            # training time step loop
            # select the action for each state
            action = agent.act(state, eps)
            # execute action, get reward, new state and whether the sequence can be continued
            # env.step(action): Step the environment by one action timestep. Return
            # observation, reward and done
            if RenderOpt:
                env.render()  # will display a popup window
                time.sleep(0.01)
            next_state, reward_t, reward_s, done, damage, _ = env.step(action)
            reward = reward_t + reward_s
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward  # total score in current episode (scalar + potential for current time)
            damage_red += damage  # total damage on red aircraft in current episode
            if damage_red == 3:  # health bar implementation on red AC
                done = True
            if done:
                break
        scores_window.append(
            score
        )  # save most recent score, adds collected reward at the end of episode
        scores.append(score)  # save most recent score
        score_mean.append(np.mean(scores_window))
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon at each episode
        if damage_red == 3:
            print("Blue wins! Dominated!")
            print(
                '\rEpisode {}\tReward in Episode: {:.5f} \tAverage Score {:.5f}\n\n'
                .format(i_episode, score, np.mean(scores_window)),
                end="")
        else:
            print(
                '\rEpisode {}\tReward in Episode: {:.5f} \tAverage Score {:.5f}\n\n'
                .format(i_episode, score, np.mean(scores_window)),
                end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_window)))
            elapsed_time = time.time() - start_time
            print("Duration: ", elapsed_time)
        if np.mean(scores_window) >= 10 and i_episode >= 100:
            print(
                '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'
                .format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(),
                       'checkpoint.pth')  # Function to save model parameters
            break
    elapsed_time = time.time() - start_time
    print("Training duration: ", elapsed_time)
    return score_mean

    env.close()


score_mean = dqn()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(score_mean)
plt.ylabel('Average Reward')
plt.xlabel('Episode #')
plt.savefig('training_result.png')
>>>>>>> 6819adb566d3adb52b4ba0d843df0a5e09f4af63
>>>>>>> baf535a02bf8c1d22686c247a37d9aa21a5ccce8
