import numpy as np 
import cv2 
import gym
import random
import time
import bpy, bpy_extras
import time
import numpy as np
import os
from mathutils import Vector
import bmesh
from math import pi
import os
import sys
sys.path.append(os.getcwd())
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from sklearn.neighbors import NearestNeighbors
from gym import Env, spaces
from render import *

class SpaghettiEnv(Env):
    def __init__(self):
        super(SpaghettiEnv, self).__init__()
        # Define a 2-D observation space
        self.observation_shape = (64, 64, 3)
        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape), 
                                            high = np.ones(self.observation_shape),
                                            dtype = np.float16)
        
        # Define an action space ranging from 0 to 2
        self.action_space = spaces.Discrete(2,) # push or group

        self.noodles = None
        self.pusher, self.fork = initialize_sim()
        self.current_render = None
        self.action_ctr = 0
        self.max_action_count = 10
    
    def reset(self):
        self.action_ctr = 0
        num_noodles = np.random.randint(5,20)
        self.noodles = reset_sim(self.pusher, self.fork, num_noodles)
        obs = render(0)
        self.current_render = obs
        return obs

    def render(self, mode = "human"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            cv2.imshow("img", self.current_render)
            cv2.waitKey(10)
        elif mode == "rgb_array":
            return self.current_render

    def get_action_meanings(self):
        return {0: "Group", 1: "Twirl"}

    def close(self):
        cv2.destroyAllWindows()
        
    def step(self, action):
        # Flag that marks the termination of an episode
        
        # Assert that it is a valid action 
        assert self.action_space.contains(action), "Invalid Action"

        initial_area, initial_num_noodles = get_coverage_pickup_stats(self.noodles)

        if action == 0:
            take_push_action(self.pusher, self.noodles)
        elif action == 1:
            take_twirl_action(self.fork, self.noodles)

        area, num_noodles = get_coverage_pickup_stats(self.noodles)

        area_reward = initial_area - area
        pickup_reward = initial_num_noodles - num_noodles
        reward = area_reward + pickup_reward

        obs = render(0)
        self.current_render = obs
        self.action_ctr += 1

        done = (self.action_ctr >= self.max_action_count) or (num_noodles <= 2)

        if done:
            clear_noodles()

        return obs, reward, done, [area_reward, pickup_reward]

if __name__ == '__main__':
    env = SpaghettiEnv()
    obs = env.reset()
    env.render()
    while True:
        # Take a random action
        print(env.action_ctr)
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(reward, info)

        if done == True:
            break
        
        # Render the game
        env.render()

    print('here')
    done = False
    obs = env.reset()
    env.render()
    while True:
        # Take a random action
        print(env.action_ctr)
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(reward, info)

        if done == True:
            break
        
        # Render the game
        env.render()
        
    
    env.close()
