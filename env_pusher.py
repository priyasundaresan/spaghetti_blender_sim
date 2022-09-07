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
from render_pusher import *
from scipy.interpolate import interp1d

class BimanualAcquisEnv(Env):
    def __init__(self, random_seed=0):
        super(BimanualAcquisEnv, self).__init__()
        # Define a 2-D observation space
        self.blender_render_size = (256,256,3)
        self.observation_shape = (64, 64, 3)
        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape), 
                                            high = np.ones(self.observation_shape),
                                            dtype = np.float16)
        
        # Define an action space ranging from 0 to 2
        self.action_space = spaces.Discrete(2,) # push or group
        self.items = None
        self.pusher, self.scooper = initialize_sim()
        self.current_render = None
        self.action_ctr = 0
        self.max_action_count = 10
        self.initial_num_items = 0
        self.random_seed = random_seed
    
    def reset(self, deterministic=False):
        self.action_ctr = 0
        #num_noodles = np.random.randint(5,20)
        num_items = 12
        self.initial_num_items = num_items
        self.items = reset_sim(self.pusher, self.scooper, num_items, deterministic=deterministic, random_seed=self.random_seed)
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
        return {0: "Group", 1: "Scoop"}

    def close(self):
        cv2.destroyAllWindows()
        
    def step(self, action):
        # Flag that marks the termination of an episode
        action = int(action)
        
        # Assert that it is a valid action 
        assert self.action_space.contains(action), "Invalid Action"

        initial_area, initial_num_items = get_coverage_pickup_stats(self.items)

        if action == 0:
            self.items, action_pixels = take_push_action(self.pusher, self.scooper, self.items)
        elif action == 1:
            self.items, action_pixels = take_scoop_action(self.pusher, self.scooper, self.items)

        area, num_items = get_coverage_pickup_stats(self.items)

        pickup_reward = initial_num_items - num_items # reward for picking up noodles
        area_reward = initial_area - area # reward for minimizing coverage

        #pickup_interp = interp1d([0,self.initial_num_items],[-5,5],kind='quadratic')

        #if pickup_reward:
        #    area_reward = 0
        #
        #if pickup_reward == 1:
        #    pickup_reward = -5

        if pickup_reward == 1:
            pickup_reward = -5

        reward = 2*area_reward + pickup_reward

        obs = render(0)
        self.current_render = obs
        self.action_ctr += 1

        print('\ninitial #: %d, '%self.initial_num_items, 'curr #: %d, '%num_items, \
                'action %d: %s, '%(self.action_ctr, self.get_action_meanings()[int(action)]), 'area reward: %f, '%area_reward, 'pickup_reward: %d'%pickup_reward, 'reward: %f'%reward)

        done = (self.action_ctr >= self.max_action_count) or (num_items <= 0)

        if done:
            clear_items()

        #action_pixels = np.array(action_pixels)//(self.blender_render_size[0]/self.observation_shape[0])
        #action_pixels = np.array(action_pixels)*(self.observation_shape[0]/self.blender_render_size[0])
        action_pixels = np.array(action_pixels)
        total_item_pickup = self.initial_num_items - num_items
        total_coverage = area

        return obs, reward, done, (action_pixels, total_item_pickup, total_coverage)

if __name__ == '__main__':
    env = BimanualAcquisEnv()
    obs = env.reset()
    env.render()

    images = [obs]
    actions = [0]
    while True:
        # Take a random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        images.append(obs)
        actions.append(action)

        if done == True:
            break
        
        # Render the game
        env.render()

    env.close()

    print('done, showing states')
    for action,img in zip(actions,images):
        cv2.putText(img, str(action), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)
        print(img.shape)
        #cv2.imshow('img', img)
        #cv2.waitKey(0)

