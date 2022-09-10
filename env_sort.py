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
from gym import Env, spaces
from render_sort import *
from scipy.interpolate import interp1d

class BlockSortEnv(Env):
    def __init__(self, random_seed=0):
        super(BlockSortEnv, self).__init__()
        # Define a 2-D observation space
        self.blender_render_size = (256,256,3)
        self.observation_shape = (64, 64, 3)
        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape), 
                                            high = np.ones(self.observation_shape),
                                            dtype = np.float16)
        
        # Define an action space ranging from 0 to 2
        self.action_space = spaces.Discrete(2,) # push or group
        self.items = None
        self.pusher = initialize_sim()
        self.current_render = None
        self.action_ctr = 0
        self.max_action_count = 8
        self.initial_num_items = 0
        self.random_seed = random_seed
    
    def reset(self, deterministic=False):
        self.action_ctr = 0
        num_items = 10
        self.initial_num_items = num_items
        self.items, self.colors = reset_sim(self.pusher, num_items, deterministic=deterministic, random_seed=self.random_seed)
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
        return {0: "Pick-Place", 1: "Push"}

    def close(self):
        cv2.destroyAllWindows()
        
    def step(self, action):
        # Flag that marks the termination of an episode
        action = int(action)
        
        # Assert that it is a valid action 
        assert self.action_space.contains(action), "Invalid Action"

        red_correct_prev, red_incorrect_prev, blue_correct_prev, blue_incorrect_prev = get_reward_stats(self.items, self.colors)
        #pick_item, place_point, push_start, push_end = get_action_candidates(self.items, self.colors)
        pick_item, place_point = get_pick_action_candidates(self.items, self.colors)
        push_start, push_end = get_push_action_candidates(self.items, self.colors)

        if action == 0 or push_start is None:
            action_pixels = take_pick_place_action(pick_item, place_point)
        elif action == 1:
            action_pixels = take_push_action(self.pusher, push_start, push_end, self.items)

        red_correct, red_incorrect, blue_correct, blue_incorrect = get_reward_stats(self.items, self.colors)
        #pick_item, place_point, push_start, push_end = get_action_candidates(self.items, self.colors)
        pick_item, place_point = get_pick_action_candidates(self.items, self.colors)
        push_start, push_end = get_push_action_candidates(self.items, self.colors)

        correct_reward = (red_correct - red_correct_prev) + (blue_correct - blue_correct_prev)
        incorrect_reward = (red_incorrect_prev - red_incorrect) + (blue_incorrect_prev - blue_incorrect)

        reward = 1*correct_reward + 1*incorrect_reward 
        if (correct_reward + incorrect_reward) == 0:
            reward = -10

        obs = render(0)
        self.current_render = obs
        self.action_ctr += 1

        counted_items = red_correct + blue_correct + red_incorrect + blue_incorrect

        print('\naction %d: %s, '%(self.action_ctr, self.get_action_meanings()[int(action)]), 'reward: %f, '%reward, \
                                   '\nred_correct: %d'%red_correct, 'red_incorrect: %d'%red_incorrect, \
                                   'blue_correct: %d'%blue_correct, 'blue_incorrect: %d'%blue_incorrect, 'num counted items: %d'%(counted_items))

        #done = (self.action_ctr >= self.max_action_count) or (pick_item is None)
        done = (self.action_ctr >= self.max_action_count) or (red_correct + blue_correct == len(self.items))
        if done:
            clear_items()

        #action_pixels = np.array(action_pixels)//(self.blender_render_size[0]/self.observation_shape[0])
        action_pixels = np.array(action_pixels)
        return obs, reward, done, (action_pixels, red_correct + blue_correct, 0.0)

if __name__ == '__main__':
    env = BlockSortEnv()
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
    
    if not os.path.exists('images'):
        os.mkdir('images')

    print('done, showing states')
    for i,(action,img) in enumerate(zip(actions,images)):
        cv2.putText(img, str(action), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)
        print(img.shape)
        cv2.imwrite('%s/%05d.jpg'%('images', i), img)
        #cv2.imshow('img', img)
        #cv2.waitKey(0)

