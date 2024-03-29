import os
import sys
import pdb
import cv2
import gym
import torch
import pickle
import pathlib
import numpy as np

from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
import random

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def to_tensor_obs(image):
    """
    Converts the input np img to channel first 64x64 dim torch img.
    """
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)
    #image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    return image


def postprocess_img(image, depth):
    """
    Postprocess an image observation for storage.
    From float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
    """
    image = np.floor((image + 0.5) * 2 ** depth)
    return np.clip(image * 2**(8 - depth), 0, 2**8 - 1).astype(np.uint8)


def preprocess_img(image, depth):
    """
    Preprocesses an observation inplace.
    From float32 Tensor [0, 255] to [-0.5, 0.5]
    Also adds some noise to the observations !!
    """
    image.div_(2 ** (8 - depth)).floor_().div_(2 ** depth).sub_(0.5)
    image.add_(torch.randn_like(image).div_(2 ** depth)).clamp_(-0.5, 0.5)
    

def bottle(func, *tensors):
    """
    Evaluates a func that operates in N x D with inputs of shape N x T x D 
    """
    n, t = tensors[0].shape[:2]
    #out = func(*[x.view(n*t, *x.shape[2:]) for x in tensors])
    out = func(*[x.reshape(n*t, *x.shape[2:]) for x in tensors])
    
    return out.view(n, t, *out.shape[1:])


def get_combined_params(*models):
    """
    Returns the combine parameter list of all the models given as input.
    """
    params = []
    for model in models:
        params.extend(list(model.parameters()))
    return params


def save_video(frames, path, name):
    """
    Saves a video containing frames.
    """
    frames = (frames*255).clip(0, 255).astype('uint8').transpose(0, 2, 3, 1)
    _, H, W, _ = frames.shape
    #writer = cv2.VideoWriter(
    #    str(pathlib.Path(path)/f'{name}.mp4'),
    #    cv2.VideoWriter_fourcc(*'mp4v'), 25., (W, H), True
    #)
    writer = cv2.VideoWriter(
        str(pathlib.Path(path)/f'{name}.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'), 0.5, (W, H), True
    )
    for frame in frames[..., ::-1]:
        writer.write(frame)
    writer.release()

def visualize_episode(frames, episode, path, name, mapping=None):
    frames = (frames*255).clip(0, 255).astype('uint8').transpose(0, 2, 3, 1)
    _, H, W, _ = frames.shape

    #if W == 134:
    if W == 64*2:
        writer = cv2.VideoWriter(
            str(pathlib.Path(path)/f'{name}.mp4'),
            cv2.VideoWriter_fourcc(*'mp4v'), 0.5, (256*2, 256), True
        )
    else:
        writer = cv2.VideoWriter(
            str(pathlib.Path(path)/f'{name}.mp4'),
            cv2.VideoWriter_fourcc(*'mp4v'), 0.5, (256, 256), True
        )

    for i, frame in enumerate(frames[..., ::-1]):
        reward = episode.r[i]

        action = episode.u[i]
        action_pixels = episode.action_pixels[i]

        vis = frame.copy()
        H1,W1,_ = vis.shape
        if W1 == 64*2:
            vis = cv2.resize(vis, (256*2, 256))
        else:
            vis = cv2.resize(vis, (256, 256))
        vis = cv2.flip(vis, 0)
        H,W,C = vis.shape

        if len(action_pixels)>1:
            u1,v1 = action_pixels.astype(int)[0]
            u2,v2 = action_pixels.astype(int)[1]
            cv2.arrowedLine(vis, (u1, v1), (u2, v2), (255,0,0), 2)
        else:
            u1,v1 = action_pixels.astype(int)[0]
            cv2.circle(vis, (u1, v1), 2, (255,0,0), -1)
        cv2.putText(vis, 'Action: %s, Reward: %.2f'%(mapping[action], reward), (10,15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,0), 1, cv2.LINE_AA)

        writer.write(vis)
    writer.release()

def save_frames(target, pred_prior, pred_posterior, name, n_rows=5):
    """
    Saves the target images with the generated prior and posterior predictions. 
    """
    image = torch.cat([target, pred_prior, pred_posterior], dim=3)
    save_image(make_grid(image + 0.5, nrow=n_rows), f'{name}.png')


def get_mask(tensor, lengths):
    """
    Generates the masks for batches of sequences.
    Time should be the first axis.
    input:
        tensor: the tensor for which to generate the mask [N x T x ...]
        lengths: lengths of the seq. [N] 
    """
    mask = torch.zeros_like(tensor)
    for i in range(len(lengths)):
        mask[i, :lengths[i]] = 1.
    return mask


def load_memory(path, device):
    """
    Loads an experience replay buffer.
    """
    with open(path, 'rb') as f:
        memory = pickle.load(f)
        memory.device = device
        for e in memory.data:
            e.device = device
    return memory


def flatten_dict(data, sep='.', prefix=''):
    """Flattens a nested dict into a dict.
    eg. {'a': 2, 'b': {'c': 20}} -> {'a': 2, 'b.c': 20}
    """
    x = {}
    for key, val in data.items():
        if isinstance(val, dict):
            x.update(flatten_dict(val, sep=sep, prefix=key))
        else:
            x[f'{prefix}{sep}{key}'] = val
    return x


class TensorBoardMetrics:
    """Plots and (optionally) stores metrics for an experiment.
    """
    def __init__(self, path):
        self.writer = SummaryWriter(path)
        self.steps = defaultdict(lambda: 0)
        self.summary = {}

    def assign_type(self, key, val):
        if isinstance(val, (list, tuple)):
            fun = lambda k, x, s: self.writer.add_histogram(k, np.array(x), s)
            self.summary[key] = fun
        elif isinstance(val, (np.ndarray, torch.Tensor)):
            self.summary[key] = self.writer.add_histogram
        elif isinstance(val, float) or isinstance(val, int):
            self.summary[key] = self.writer.add_scalar
        else:
            raise ValueError(f'Datatype {type(val)} not allowed')
    
    def update(self, metrics: dict):
        metrics = flatten_dict(metrics)
        for key_dots, val in metrics.items():
            key = key_dots.replace('.', '/')
            if self.summary.get(key, None) is None:
                self.assign_type(key, val)
            self.summary[key](key, val, self.steps[key])
            self.steps[key] += 1


def apply_model(model, inputs, ignore_dim=None):
    pass

def plot_metrics(metrics, path, prefix):
    for key, val in metrics.items():
        lineplot(np.arange(len(val)), val, f'{prefix}{key}', path)

def lineplot(xs, ys, title, path='', xaxis='episode'):
    MAX_LINE = Line(color='rgb(0, 132, 180)', dash='dash')
    MIN_LINE = Line(color='rgb(0, 132, 180)', dash='dash')
    NO_LINE = Line(color='rgba(0, 0, 0, 0)')
    MEAN_LINE = Line(color='rgb(0, 172, 237)')
    std_colour = 'rgba(29, 202, 255, 0.2)'
    if isinstance(ys, dict):
        data = []
        for key, val in ys.items():
            xs = np.arange(len(val))
            data.append(Scatter(x=xs, y=np.array(val), name=key))
    elif np.asarray(ys, dtype=np.float32).ndim == 2:
        ys = np.asarray(ys, dtype=np.float32)
        ys_mean, ys_std = ys.mean(-1), ys.std(-1)
        ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std
        l_max = Scatter(x=xs, y=ys.max(-1), line=MAX_LINE, name='Max')
        l_min = Scatter(x=xs, y=ys.min(-1), line=MIN_LINE, name='Min')
        l_stu = Scatter(x=xs, y=ys_upper, line=NO_LINE, showlegend=False)
        l_mean = Scatter(
            x=xs, y=ys_mean, fill='tonexty', fillcolor=std_colour,
            line=MEAN_LINE, name='Mean'
        )
        l_stl = Scatter(
            x=xs, y=ys_lower, fill='tonexty', fillcolor=std_colour,
            line=NO_LINE, name='-1 Std. Dev.', showlegend=False
        )
        data = [l_stu, l_mean, l_stl, l_min, l_max]
    else:
        data = [Scatter(x=xs, y=ys, line=MEAN_LINE)]
    plotly.offline.plot({
        'data': data,
        'layout': dict(
            title=title,
            xaxis={'title': xaxis},
            yaxis={'title': title}
            )
        }, filename=os.path.join(path, title + '.html'), auto_open=False
    )

class TorchImageEnvWrapper:
    """
    Torch Env Wrapper that wraps a gym env and makes interactions using Tensors.
    Also returns observations in image form.
    """
    def __init__(self, env, bit_depth, observation_shape=None, act_rep=2, discrete=True):
        self.env = env
        self.bit_depth = bit_depth
        self.action_repeats = act_rep
        self.discrete = discrete

    def reset(self, deterministic=False):
        self.env.reset(deterministic=deterministic)
        x = to_tensor_obs(self.env.render(mode='rgb_array'))
        preprocess_img(x, self.bit_depth)
        return x

    def step(self, u):
        u, rwds = u.cpu().detach().numpy(), 0
        for _ in range(self.action_repeats):
            _, r, d, i = self.env.step(u)
            rwds += r
        x = to_tensor_obs(self.env.render(mode='rgb_array'))
        preprocess_img(x, self.bit_depth)
        return x, rwds, d, i

    def render(self):
        return to_tensor_obs(self.env.render(mode='rgb_array'))
        #return self.env.render(mode)

    def close(self):
        self.env.close()

    @property
    def observation_size(self):
        return (3, 64, 64)
        #return (3, 128, 128)

    @property
    def action_size(self):
        if self.discrete:
            return 1
        else:
            return self.env.action_space.shape[0]

    def sample_random_action(self):
        return torch.tensor(self.env.action_space.sample())

class OneHotAction:

  def __init__(self, env):
    assert isinstance(env.action_space, gym.spaces.Discrete)
    self._env = env

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    shape = (self._env.action_space.n,)
    space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
    space.sample = self._sample_action
    return space

  def step(self, action):
    index = np.argmax(action).astype(int)
    reference = np.zeros_like(action)
    reference[index] = 1
    #if not np.allclose(reference, action):
    #  raise ValueError(f'Invalid one-hot action:\n{action}')
    return self._env.step(index)

  def reset(self):
    return self._env.reset()

  def _sample_action(self):
    actions = self._env.action_space.n
    #index = self._random.randint(0, actions)
    index = np.random.randint(0, actions)
    reference = np.zeros(actions, dtype=np.float32)
    reference[index] = 1.0
    return reference
