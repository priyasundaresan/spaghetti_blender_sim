import pdb
import random
import torch
import itertools

from torch.distributions import Normal

def enumerate_binary_actions(planning_horizon, device):
    actions = list(itertools.product([0, 1], repeat=planning_horizon))
    return torch.Tensor(actions).to(device)

class RSSMPolicy:
    def __init__(self,
            model,
            planning_horizon,
            num_candidates,
            num_iterations,
            top_candidates,
            device
        ):
        super().__init__()
        self.rssm = model
        self.N = num_candidates
        self.K = top_candidates
        self.T = num_iterations
        self.H = planning_horizon
        self.d = self.rssm.action_size
        self.device = device
        self.state_size = self.rssm.state_size
        self.latent_size = self.rssm.latent_size
        self.binary_actions = enumerate_binary_actions(self.H, self.device)
        self.discount = 0.9

    def reset(self):
        self.h = torch.zeros(1, self.state_size).to(self.device)
        self.s = torch.zeros(1, self.latent_size).to(self.device)
        self.a = torch.zeros(1, self.d).to(self.device)

    #def _poll(self, obs):
    #    self.mu = torch.zeros(self.H, self.d).to(self.device)
    #    self.stddev = torch.ones(self.H, self.d).to(self.device)
    #    # observation could be of shape [CHW] but only 1 timestep
    #    assert len(obs.shape) == 3, 'obs should be [CHW]'
    #    self.h, self.s = self.rssm.get_init_state(
    #        self.rssm.encoder(obs[None]),
    #        self.h, self.s, self.a
    #    )
    #    for _ in range(self.T):
    #        rwds = torch.zeros(self.N).to(self.device)
    #        actions = Normal(self.mu, self.stddev).sample((self.N,))
    #        h_t = self.h.clone().expand(self.N, -1)
    #        s_t = self.s.clone().expand(self.N, -1)
    #        for a_t in torch.unbind(actions, dim=1):
    #            h_t = self.rssm.deterministic_state_fwd(h_t, s_t, a_t)
    #            s_t = self.rssm.state_prior(h_t, sample=True)
    #            rwds += self.rssm.pred_reward(h_t, s_t)
    #        _, k = torch.topk(rwds, self.K, dim=0, largest=True, sorted=False)
    #        self.mu = actions[k].mean(dim=0)
    #        self.stddev = actions[k].std(dim=0, unbiased=False)
    #    self.a = self.mu[0:1]

    #def poll(self, observation, explore=False):
    #    with torch.no_grad():
    #        self._poll(observation)
    #        if explore:
    #            self.a += torch.randn_like(self.a)*0.3
    #        return self.a

    def _poll(self, obs):
        assert len(obs.shape) == 3, 'obs should be [CHW]'
        self.h, self.s = self.rssm.get_init_state(
            self.rssm.encoder(obs[None]),
            self.h, self.s, self.a
        )

        rwds = torch.zeros(self.N).to(self.device)
        #actions = enumerate_binary_actions(self.H, self.device)
        h_t = self.h.clone().expand(self.N, -1)
        s_t = self.s.clone().expand(self.N, -1)
        #for a_t in torch.unbind(actions, dim=1):

        t = 0
        for a_t in torch.unbind(self.binary_actions, dim=1):
            h_t = self.rssm.deterministic_state_fwd(h_t, s_t, a_t)
            s_t = self.rssm.state_prior(h_t, sample=True)
            pred_reward = self.rssm.pred_reward(h_t, s_t)
            rwds += pred_reward*(self.discount**t)
            t += 1
        _, k = torch.topk(rwds, self.K, dim=0, largest=True, sorted=False)
        best_action_seq = self.binary_actions[k].squeeze()
        best_action = best_action_seq[0].view(self.a.shape)
        self.a = best_action
        

    def poll(self, observation, explore=False):
        with torch.no_grad():
            self._poll(observation)
            if explore:
                #if random.random() < 0.3:
                if random.random() < 0.5:
                    self.a = 1-self.a
            return self.a
