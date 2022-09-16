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

from env_sort import BlockSortEnv
from utils import *
from memory import *
from rssm_model import *
from rssm_policy import *
from rollout_generator import RolloutGenerator

#def train(memory, rssm, optimizer, device, N=32, H=50, beta=1.0, grads=False):
def train(memory, rssm, optimizer, device, N=32, H=1, beta=1.0, grads=False):
    """
    Training implementation as indicated in:
    Learning Latent Dynamics for Planning from Pixels
    arXiv:1811.04551

    (a.) The Standard Varioational Bound Method
        using only single step predictions.
    """
    free_nats = torch.ones(1, device=device)*3.0
    batch = memory.sample(N, H, time_first=True)
    x, u, r, t  = [torch.tensor(x).float().to(device) for x in batch]
    preprocess_img(x, depth=5)
    e_t = bottle(rssm.encoder, x)
    h_t, s_t = rssm.get_init_state(e_t[0])
    kl_loss, rc_loss, re_loss = 0, 0, 0
    states, priors, posteriors, posterior_samples = [], [], [], []

    for i, a_t in enumerate(torch.unbind(u, dim=0)):
        h_t = rssm.deterministic_state_fwd(h_t, s_t, a_t)
        states.append(h_t)
        priors.append(rssm.state_prior(h_t))
        posteriors.append(rssm.state_posterior(h_t, e_t[i + 1]))
        posterior_samples.append(Normal(*posteriors[-1]).rsample())
        s_t = posterior_samples[-1]
    prior_dist = Normal(*map(torch.stack, zip(*priors)))
    posterior_dist = Normal(*map(torch.stack, zip(*posteriors)))
    states, posterior_samples = map(torch.stack, (states, posterior_samples))
    rec_loss = F.mse_loss(
        bottle(rssm.decoder, states, posterior_samples), x[1:],
        reduction='none'
    ).sum((2, 3, 4)).mean()
    kld_loss = torch.max(
        kl_divergence(posterior_dist, prior_dist).sum(-1),
        free_nats
    ).mean()
    rew_loss = F.mse_loss(
        bottle(rssm.pred_reward, states, posterior_samples), r
    )
    optimizer.zero_grad()
    nn.utils.clip_grad_norm_(rssm.parameters(), 1000., norm_type=2)
    (beta*kld_loss + rec_loss + rew_loss).backward()
    optimizer.step()
    metrics = {
        'losses': {
            'kl': kld_loss.item(),
            'reconstruction': rec_loss.item(),
            'reward_pred': rew_loss.item()
        },
    }
    if grads:
        metrics['grad_norms'] = {
            k: 0 if v.grad is None else v.grad.norm().item()
            for k, v in rssm.named_parameters()
        }
    return metrics


def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]  # get all args after "--"
    random_seed = int(argv[-1])

    set_seed_everywhere(random_seed)

    env = BlockSortEnv(random_seed=random_seed)
    #env = OneHotAction(env)
    env = TorchImageEnvWrapper(env, bit_depth=5, act_rep=1)
    print('action size', env.action_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rssm_model = RecurrentStateSpaceModel(env.action_size).to(device)
    optimizer = torch.optim.Adam(rssm_model.parameters(), lr=1e-3, eps=1e-4)

    planning_horizon = 8
    policy = RSSMPolicy(
        rssm_model, 
        planning_horizon=planning_horizon,
        num_candidates=2**planning_horizon,
        num_iterations=1,
        top_candidates=1,
        device=device
    )

    rollout_gen = RolloutGenerator(
        env,
        device,
        policy=policy,
        episode_gen=lambda : Episode(partial(postprocess_img, depth=5)),
        max_episode_steps=env.env.max_action_count,
    )
    mem = Memory(100)
    mem.append(rollout_gen.rollout_n(1, random_policy=True))
    res_dir = 'results_randomseed_%d/'%random_seed
    summary = TensorBoardMetrics(f'{res_dir}/')
    #for i in trange(100, desc='Epoch', leave=False):
    act_sequences = []
    #for i in trange(100, desc='Epoch', leave=False):
    #for i in trange(100, desc='Epoch', leave=False):
    for i in trange(50, desc='Epoch', leave=False):
        print('\nEPOCH: %d'%i)
        metrics = {}
        for _ in trange(150, desc='Iter ', leave=False):
            train_metrics = train(mem, rssm_model.train(), optimizer, device)
            for k, v in flatten_dict(train_metrics).items():
                if k not in metrics.keys():
                    metrics[k] = []
                metrics[k].append(v)
                metrics[f'{k}_mean'] = np.array(v).mean()
        
        summary.update(metrics)
        mem.append(rollout_gen.rollout_once(explore=True))
        eval_episode, eval_frames, eval_metrics, eval_act_seq = rollout_gen.rollout_eval(hardcode_last_action=False)
        act_sequences.append(eval_act_seq)
        mem.append(eval_episode)
        visualize_episode(eval_frames, eval_episode, res_dir, f'vid_{i+1}', env.env.get_action_meanings())
        np.savez_compressed('%s/%03d.npz'%(res_dir, i), act_seq=eval_act_seq)
        try:
            summary.update(eval_metrics)
        except:
            continue

        if (i + 1) % 25 == 0:
            torch.save(rssm_model.state_dict(), f'{res_dir}/ckpt_{i+1}.pth')

    np.save(f'{res_dir}/eval_act_seqs.npy', np.array(act_sequences)) 
    print('DONE')
    exit()

    #pdb.set_trace()

if __name__ == '__main__':
    main()
