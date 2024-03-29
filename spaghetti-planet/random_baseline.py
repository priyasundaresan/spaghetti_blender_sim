import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), '..'))

#import pdb
import torch
from tqdm import trange
from functools import partial
from collections import defaultdict
from torch.distributions import Normal, kl
from torch.distributions.kl import kl_divergence
from env import SpaghettiEnv
from env_pusher import BimanualAcquisEnv
from env_sort import BlockSortEnv
from utils import *
from memory import *
from rollout_generator import RolloutGenerator

def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]  # get all args after "--"
    random_seed = int(argv[-1])

    set_seed_everywhere(random_seed)

    #env = SpaghettiEnv(random_seed=random_seed)
    #env = BimanualAcquisEnv(random_seed=random_seed)
    env = BlockSortEnv(random_seed=random_seed)

    env = TorchImageEnvWrapper(env, bit_depth=5, act_rep=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rollout_gen = RolloutGenerator(
        env,
        device,
        policy=None,
        episode_gen=lambda : Episode(partial(postprocess_img, depth=5)),
        max_episode_steps=env.env.max_action_count,
    )
    #res_dir = 'results_random/'
    res_dir = 'results_random_randomseed_%d/'%random_seed

    summary = TensorBoardMetrics(f'{res_dir}/')

    act_sequences = []
    for i in trange(10, leave=False):
        print('\ROLLOUT: %d'%i)
        metrics = {}
        eval_episode, eval_frames, eval_metrics, eval_act_seq = rollout_gen.rollout_baseline()
        act_sequences.append(eval_act_seq)
        visualize_episode(eval_frames, eval_episode, res_dir, f'vid_{i+1}', env.env.get_action_meanings())
        summary.update(eval_metrics)

    np.save(f'{res_dir}/eval_act_seqs.npy', np.array(act_sequences)) 
    print('DONE')
    exit()

if __name__ == '__main__':
    main()
