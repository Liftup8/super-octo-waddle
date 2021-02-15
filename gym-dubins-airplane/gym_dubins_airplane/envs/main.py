# Main function of the RL air combat training programe
#
# Hasan ISCI - 27.12.2020

import gym
gym.logger.set_level(40)
import gym_dubins_airplane
from matplotlib import pyplot as plt
import numpy as np
DebugInfo = True
RenderSteps = True
import time

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # To test directly aircraft model
    # simple_ac = ACEnvironment2D()
    # simple_ac.takeaction(0.,0.,5.)

    env = gym.make('dubinsAC2D-v0',
                   actions='discrete')  # returns the environment
    state = env.reset()

    reward_history = []
    score = 0
    damage_red = 0
    # temp_r = temp_s = temp_d = 0
    # index_di = 0
    # Run environment with arbitrary actions for testing purposes
    for i in range(10000):

        state, reward, terminate, damage, info = env.step(
            env.action_space.sample())  # take a random action
        score += reward
        damage_red += damage

        reward_history.append(reward)

        if RenderSteps:
            env.render()
        if DebugInfo:
            # if(reward == temp_r and score == temp_s and damage == temp_d):
            # index_di += 1
            # else:
            # print({'reward': temp_r, 'score': temp_s, 'damage': temp_d}, "x", index_di)
            # temp_r = reward
            # temp_s = score
            # temp_d = damage
            # print({'reward': reward, 'score': score, 'damage': damage_red})
            print({'reward': reward, 'score': score, 'damage': damage_red})
        if damage_red == 3:  # health bar implementation on red AC (aynı anda hepsini vuruyor, incelenecek)
            print("Blue wins! Dominated!")
            terminate = True
        if terminate:

            plt.plot(np.array(reward_history))

            reward_history.clear()

            state = env.reset()
            score = 0
            damage_red = 0
    env.close()  # closes the environment rendering window
