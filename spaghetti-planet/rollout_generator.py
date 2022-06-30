import pdb
import numpy as np
import torch
from collections import defaultdict

from tqdm import trange
from torchvision.utils import make_grid

from memory import Episode # this needs modification!

class RolloutGenerator:
    """Rollout generator class."""
    def __init__(self,
        env,
        device,
        policy=None,
        max_episode_steps=None,
        episode_gen=None,
        name=None,
    ):
        self.env = env
        self.device = device
        self.policy = policy
        self.episode_gen = episode_gen or Episode
        self.name = name or 'Rollout Generator'
        self.max_episode_steps = max_episode_steps
        if self.max_episode_steps is None:
            self.max_episode_steps = self.env.max_episode_steps
        self.discount = 0.9

    def rollout_once(self, random_policy=False, explore=False) -> Episode:
        """Performs a single rollout of an environment given a policy
        and returns and episode instance.
        """
        if self.policy is None and not random_policy:
            random_policy = True
            print('Policy is None. Using random policy instead!!')
        if not random_policy:
            self.policy.reset()
        eps = self.episode_gen()
        obs = self.env.reset()
        des = f'{self.name} Ts'
        for t in trange(self.max_episode_steps, desc=des, leave=False):
            if random_policy:
                act = self.env.sample_random_action()
            else:
                act = self.policy.poll(obs.to(self.device), explore).flatten()

            if t == self.max_episode_steps-1:
                act = torch.Tensor([1.0])

            nobs, reward, terminal, (action_pixels, total_noodle_pickup)  = self.env.step(act)

            reward = self.discount**t * reward # priya new

            eps.append(obs, act, reward, terminal, action_pixels)
            obs = nobs
            if terminal:
                eps.append(self.env.render(), act, reward, terminal, action_pixels)
                break
        eps.terminate(nobs)
        return eps 

    def rollout_n(self, n=1, random_policy=False) -> [Episode]:
        """
        Performs n rollouts.
        """
        if self.policy is None and not random_policy:
            random_policy = True
            print('Policy is None. Using random policy instead!!')
        des = f'{self.name} EPS'
        ret = []
        for _ in trange(n, desc=des, leave=False):
            ret.append(self.rollout_once(random_policy=random_policy))
        return ret

    def rollout_eval_n(self, n):
        metrics = defaultdict(list)
        episodes, frames, act_sequences = [], [], []
        for _ in range(n):
            e, f, m, a = self.rollout_eval()
            episodes.append(e)
            act_sequences.append(a)
            frames.append(f)
            for k, v in m.items():
                metrics[k].append(v)
        return episodes, frames, metrics, act_sequences

    def rollout_baseline(self, policy=None):
        eps = self.episode_gen()
        obs = self.env.reset()
        des = f'{self.name} Eval Ts'
        frames = []
        metrics = {}
        act_r = []
        act_seq = []
        eps_reward = 0
        eps_pickup = 0
        for t in trange(self.max_episode_steps, desc=des, leave=False):
            if policy is None:
                act = self.env.sample_random_action()
            else:
                act = torch.Tensor([0.0]) if (t in [0,1,2,4,5,6,8]) else torch.Tensor([1.0])
            if t == self.max_episode_steps-1:
                # always twirl last
                act = torch.Tensor([1.0])

            nobs, reward, terminal, (action_pixels, total_noodle_pickup) = self.env.step(act)

            reward = self.discount**t * reward # priya new

            act_seq.append([act.item(), total_noodle_pickup])
            frames.append(make_grid([nobs + 0.5], nrow=1).numpy())
            eps.append(obs, act, reward, terminal, action_pixels)
            print('\neval step %d:'%t, act.item(), reward, terminal, total_noodle_pickup)
            act_r.append(reward)
            eps_reward += reward
            eps_pickup = total_noodle_pickup
            obs = nobs
            if terminal:
                eps.append(self.env.render(), act, reward, terminal, action_pixels)
                break
        eps.terminate(nobs)
        metrics['eval/episode_reward'] = eps_reward
        metrics['eval/noodle_pickup'] = eps_pickup
        return eps, np.stack(frames), metrics, act_seq

    def rollout_eval(self):
        assert self.policy is not None, 'Policy is None!!'
        self.policy.reset()
        eps = self.episode_gen()
        #obs = self.env.reset()
        obs = self.env.reset(deterministic=True)
        des = f'{self.name} Eval Ts'
        frames = []
        metrics = {}
        rec_losses = []
        pred_r, act_r = [], []
        act_seq = []
        eps_reward = 0
        eps_pickup = 0
        for t in trange(self.max_episode_steps, desc=des, leave=False):
            with torch.no_grad():
                act = self.policy.poll(obs.to(self.device)).flatten()
                dec = self.policy.rssm.decoder(
                    self.policy.h,
                    self.policy.s
                ).squeeze().cpu().clamp_(-0.5, 0.5)
                rec_losses.append(((obs - dec).abs()).sum().item())
                frames.append(make_grid([obs + 0.5, dec + 0.5], nrow=2).numpy())
                pred_r.append(self.policy.rssm.pred_reward(
                    self.policy.h, self.policy.s
                ).cpu().flatten().item())

            if t == self.max_episode_steps-1:
                # always twirl last
                act = torch.Tensor([1.0])

            nobs, reward, terminal, (action_pixels, total_noodle_pickup) = self.env.step(act)

            print('\nDISCOUNTED REWARD', reward, self.discount**t)
            reward = self.discount**t * reward # priya new

            act_seq.append([act.item(), total_noodle_pickup])

            eps.append(obs, act, reward, terminal, action_pixels)
            print('\neval step %d:'%t, act.item(), reward, terminal, total_noodle_pickup)
            act_r.append(reward)
            eps_reward += reward
            eps_pickup = total_noodle_pickup
            obs = nobs
            if terminal:
                eps.append(self.env.render(), act, reward, terminal, action_pixels)
                break
        eps.terminate(nobs)
        metrics['eval/episode_reward'] = eps_reward
        metrics['eval/noodle_pickup'] = eps_pickup
        metrics['eval/reconstruction_loss'] = rec_losses
        metrics['eval/reward_pred_loss'] = abs(
            np.array(act_r)[:-1] - np.array(pred_r)[1:]
        )
        return eps, np.stack(frames), metrics, act_seq
