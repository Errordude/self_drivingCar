# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 15:13:35 2020

@author: Andrii Patrikei
"""

import gym
import matplotlib as plt
import matplotlib.pyplot as plt
from IPython import display
import time
env = gym.make("MountainCar-v0")
env.reset()


# new_obs, reward, is_done, _ = env.step(2)

# # print("new observation code:", new_obs)
# # print("reward:", reward)
# # print("is game over?:", is_done)

# plt.pyplot.imshow(env.render('rgb_array'))
# # print("Observation space:", new_obs)
# # print("Action space:", env.action_space)
# # # Note: as you can see, the car has moved to the right slightly (around 0.0005)


# create env manually to set time limit. Please don't change this.
TIME_LIMIT = 60
TIME_LIMIT_1 = 30
env = gym.wrappers.TimeLimit(
    gym.envs.classic_control.MountainCarEnv(),
    max_episode_steps=1000,
)
s = env.reset()
actions = {'left': 0, 'stop': 1, 'right': 2}

plt.figure(figsize=(4, 3))
display.clear_output(wait=True)




for t in range(30):
    plt.gca().clear()
    
    # change the line below to reach the flag
    s, r, done, _ = env.step(actions['right'])
    
    # draw game image on display
    plt.imshow(env.render('rgb_array'))
    # time.sleep(0.1)
    display.clear_output(wait=True)
    display.display(plt.gcf())
    # env.close()

for t in range(40):
    plt.gca().clear()
    
    # change the line below to reach the flag
    s1, r1, done1, _1 = env.step(actions['left'])
    

    # draw game image on display
    plt.imshow(env.render('rgb_array'))
    # time.sleep(0.1)
    display.clear_output(wait=True)
    display.display(plt.gcf())
    # env.close()
for t in range(50):
    plt.gca().clear()
    
    # change the line below to reach the flag
    s, r, done, _ = env.step(actions['right'])
    
    # draw game image on display
    plt.imshow(env.render('rgb_array'))
    # time.sleep(0.1)
    display.clear_output(wait=True)
    display.display(plt.gcf())
    # env.close()

    if done:
        print('done')
        break
else:
    print("your time is up")
assert s[0] > 0.47
print("You solved it!")
display.clear_output(wait=True)