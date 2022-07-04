import sys
import os
import cv2
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), '..'))

#import pdb
import torch
from tqdm import trange
from functools import partial
from collections import defaultdict

from torch.distributions import Normal, kl
from torch.distributions.kl import kl_divergence

from utils import *
from memory import *
from rssm_model import *
from rssm_policy import *
from rollout_generator import RolloutGenerator

action_size = 1
obs_size = (64,64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def img_inference(cv_img, policy):
    mapping = {0: "Group", 1:"Twirl"}
    obs = to_tensor_obs(cv_img)
    act = policy.poll(obs.to(device)).flatten()
    vis = cv_img.copy()
    cv2.putText(vis, '%s'%mapping[act.item()], (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 1, cv2.LINE_AA)
    return act, vis

def eval(path_to_checkpoint, imgs_dir):
    if not os.path.exists('preds'):
        print('here')
        os.mkdir('preds')

    rssm_model = RecurrentStateSpaceModel(action_size).to(device)
    rssm_model.load_state_dict(torch.load(path_to_checkpoint))
    policy = RSSMPolicy(
        rssm_model, 
        planning_horizon=10,
        num_candidates=1024,
        num_iterations=1,
        top_candidates=1,
        device=device
    )
    print('loaded model, policy')

    policy.reset()
    for idx, fn in enumerate(os.listdir(imgs_dir)):
        img = cv2.imread(os.path.join(imgs_dir, fn))
        act, vis = img_inference(img, policy)
        cv2.imwrite('preds/%03d.jpg'%idx, vis)
        print(act)
    
    #rollout_gen = RolloutGenerator(
    #    env,
    #    device,
    #    policy=policy,
    #    episode_gen=lambda : Episode(partial(postprocess_img, depth=5)),
    #    max_episode_steps=env.env.max_action_count,
    #)
    #eval_episode, eval_frames, eval_metrics, eval_act_seq = rollout_gen.rollout_eval()


if __name__ == '__main__':
    path_to_checkpoint = 'results/ckpt_100.pth'
    imgs_dir = 'real_segmasks'
    eval(path_to_checkpoint, imgs_dir)
