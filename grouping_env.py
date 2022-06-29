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

class GroupingEnv(Env):
    def __init__(self):
        super(GroupingEnv, self).__init__()
        self.observation_shape = (64, 64, 3)
        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape), 
                                            high = np.ones(self.observation_shape),
                                            dtype = np.float16)

        self.action_high = np.array([3.0, 3.0, 0.3, 0.3], dtype=np.float32)
        self.action_space = spaces.Box(low=-self.action_high, high=self.action_high, dtype=np.float32)
        
        self.noodles = None
        self.pusher, self.fork = initialize_sim()
        self.current_render = None
        self.action_ctr = 0
        self.max_action_count = 10
        self.initial_num_noodles = 0
    
    def reset(self):
        self.action_ctr = 0
        num_noodles = np.random.randint(5,20)
        self.initial_num_noodles = num_noodles
        self.noodles = reset_sim(self.pusher, self.fork, num_noodles)
        obs = render(0)
        self.current_render = obs
        return obs

    def render(self, mode = "human"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            cv2.imshow("img", self.current_render)
            cv2.waitKey(5)
        elif mode == "rgb_array":
            return self.current_render

    def get_action_meanings(self):
        return {0: "Group", 1: "Twirl"}

    def close(self):
        cv2.destroyAllWindows()
        
    def step(self, action):
        # Flag that marks the termination of an episode
        
        action = np.clip(action, -self.action_high, self.action_high)
        # Assert that it is a valid action 
        assert self.action_space.contains(action), "Invalid Action"

        initial_area, _ = get_coverage_pickup_stats(self.noodles)

        take_push_action(self.pusher, self.noodles, action[:2], action[2:])

        area, num_noodles = get_coverage_pickup_stats(self.noodles)

        reward = initial_area - area

        obs = render(0)
        self.current_render = obs
        self.action_ctr += 1

        print('\ninitial #: %d, '%self.initial_num_noodles, \
                'action %d: %s, '%(self.action_ctr, str(np.round(action, 2).tolist())), 'reward: %f, '%reward)

        done = (self.action_ctr >= self.max_action_count) or (num_noodles <= 0)
        if done:
            clear_noodles()

        return obs, reward, done, None

if __name__ == '__main__':
    env = GroupingEnv()
    obs = env.reset()
    env.render()

    images = [obs]
    actions = [0]
    rewards = [0.0]
    while True:
        # Take a random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        images.append(obs)
        actions.append(action)
        rewards.append(reward)

        if done == True:
            break
        
        # Render the game
        env.render()

    env.close()

    print('done, showing states')
    for action,img,reward in zip(actions,images,rewards):
        print(action,reward)
        cv2.putText(img, 'R: %.2f'%reward, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1, cv2.LINE_AA)
        cv2.imshow('img', img)
        cv2.waitKey(0)

