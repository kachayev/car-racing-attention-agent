import argparse
import math
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, Tuple

import gym
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
import torch
from torch import nn
from torchvision import transforms


#
#  More TODO items to improve the code:
#   * better logger for fitness/evaluation functions (maybe tensorboard/wandb)
#   * checkpoint/resume logic could be definitely improved
#   * agent implementation doesn't require torch, btw
#


#
# modules / build blocks for the solution
#
class LSTMController(nn.Module):

    def __init__(
        self,
        input_dim,
        num_hidden,
        output_dim,
        output_activation: str = "tanh"
    ):
        super().__init__()
        self._hidden_size = num_hidden
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self._hidden_size,
            num_layers=1,
        )
        self.fc = nn.Linear(
            in_features=self._hidden_size,
            out_features=output_dim,
        )
        if output_activation == 'tanh':
            self.activation = nn.Tanh()
        elif output_activation == 'softmax':
            self.activation = nn.Softmax(dim=-1)
        else:
            raise ValueError("unsupported activation function")
        self.reset()
        self.eval()

    def forward(self, x):
        x, self._hidden = self.lstm(x.view(1, 1, -1), self._hidden)
        x = self.fc(x)
        x = self.activation(x)
        return x

    def reset(self):
        self._hidden = (
            torch.zeros((1, 1, self._hidden_size)),
            torch.zeros((1, 1, self._hidden_size)),
        )


class SelfAttention(nn.Module):

    def __init__(self, data_dim, dim_q):
        super().__init__()
        self.fc_q = nn.Linear(data_dim, dim_q)
        self.fc_k = nn.Linear(data_dim, dim_q)
        self.eval()

    def forward(self, X):
        _, _, K = X.size()
        queries = self.fc_q(X) # (B, T, Q)
        keys = self.fc_k(X) # (B, T, Q)
        dot = torch.bmm(queries, keys.transpose(1, 2)) # (B, T, T)
        scaled = torch.div(dot, math.sqrt(K))
        return scaled


class CarRacingAgent(nn.Module):
    """CarRacing agents described in 'Neuroevolution of Self-Interpretable Agents'
    https://arxiv.org/pdf/2003.08165v2.pdf

    Implementation is based on the original code here:
    https://github.com/google/brain-tokyo-workshop
    """
    def __init__(
        self,
        image_size,
        query_dim,
        output_dim,
        output_activation,
        num_hidden,
        patch_size,
        patch_stride,
        top_k,
        data_dim,
        normalize_positions: bool = True,
    ):
        super().__init__()
        self._image_size = image_size
        self._patch_size = patch_size
        self._patch_stride = patch_stride
        self._top_k = top_k
        self._normalize_positions = normalize_positions

        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        n = int((image_size - patch_size) / patch_stride + 1)
        offset = self._patch_size // 2
        patch_centers = []
        for i in range(n):
            patch_center_row = offset + i * patch_stride
            for j in range(n):
                patch_center_col = offset + j * patch_stride
                patch_centers.append([patch_center_row, patch_center_col])
        self._patch_centers = torch.tensor(patch_centers).float()
        self.attention = SelfAttention(
            data_dim=data_dim * self._patch_size ** 2,
            dim_q=query_dim,
        )
        self.controller = LSTMController(
            input_dim=self._top_k * 2,
            output_dim=output_dim,
            num_hidden=num_hidden,
            output_activation=output_activation,
        )
        self.eval()

    def forward(self, x):
        x = x.permute(1, 2, 0)
        _, _, C = x.size()
        patches = x.unfold(0, self._patch_size, self._patch_stride).permute(0, 3, 1, 2)
        patches = patches.unfold(2, self._patch_size, self._patch_stride).permute(0, 2, 1, 4, 3)
        patches = patches.reshape((-1, self._patch_size, self._patch_size, C))
        flattened_patches = patches.reshape((1, -1, C * self._patch_size ** 2))
        attention_matrix = self.attention(flattened_patches)
        patch_importance_matrix = torch.softmax(attention_matrix.squeeze(), dim=-1)
        patch_importance = patch_importance_matrix.sum(dim=0)
        ix = torch.argsort(patch_importance, descending=True)
        top_k_ix = ix[:self._top_k]
        centers = self._patch_centers[top_k_ix].flatten(0, -1)
        if self._normalize_positions:
            centers = centers / self._image_size
        return centers

    def step(self, obs):
        with torch.no_grad():
            x = self._transform(obs)
            centers = self.forward(x)
        return centers, None

    def reset(self):
        self.controller.reset()


class CarRacingWrapper(gym.Wrapper):

    def __init__(self, env, base_agent, steps_cap=0, neg_reward_cap=0):
        super().__init__(env)
        self.env = env
        self.steps_cap = steps_cap
        self.neg_reward_cap = neg_reward_cap
        self.action_range = (env.action_space.high - env.action_space.low) / 2.
        self.action_mean = (env.action_space.high + env.action_space.low) / 2.
        self.neg_reward_seq = 0
        self.steps_count = 0
        self.base_agent = base_agent

    def reset(self):
        self.base_agent.reset()
        obs, flag = self.env.reset()
        obs = self.overwrite_obs(obs)
        self.neg_reward_seq = 0
        self.steps_count = 0
        return obs, flag

    def overwrite_obs(self, obs):
        centers, _ = self.base_agent.step(obs)
        return centers

    def overwrite_terminate_flag(self, reward):
        if self.neg_reward_cap == 0:
            # no need to terminate early
            return False
        self.neg_reward_seq = 0 if reward >= 0 else self.neg_reward_seq + 1
        out_of_tracks = 0 < self.neg_reward_cap < self.neg_reward_seq
        overtime = 0 < self.steps_cap <= self.steps_count
        return out_of_tracks or overtime

    def step(self, action):
        self.steps_count += 1
        action = action * self.action_range + self.action_mean
        obs, reward, done, timeout, info = self.env.step(action)
        obs = self.overwrite_obs(obs)
        done = done or self.overwrite_terminate_flag(reward)
        return obs, reward, done, timeout, info


#
# Training loop
#
def rollout(env, agent) -> Tuple[float, Dict[str, Any]]:
    total_reward, done, steps = 0, False, 0
    obs, _ = env.reset()
    agent.reset()
    while not done:
        action, _ = agent.step(obs)
        obs, reward, done, _, _ = env.step(action)
        steps += 1
        total_reward += reward
    return total_reward, {"steps": steps}


def make_base_env(base_agent_params, evaluate: bool = False, render: bool = False):
    render_mode = "human" if render else None
    env = gym.make("CarRacing-v2", verbose=False, render_mode=render_mode)
    kwargs = dict(neg_reward_cap=20, steps_cap=1000) if not evaluate else {}
    base_agent = make_base_agent(base_agent_params)
    env = CarRacingWrapper(env, base_agent, **kwargs)
    return env


def make_env(base_agent_params: np.ndarray, seed: int, rank: int):
    def _init() -> gym.Env:
        env = make_base_agent(base_agent_params)
        env.seed(seed+rank)
        return env
    set_random_seed(seed)
    return _init


def make_base_agent(base_agent_params):
    agent = CarRacingAgent(
        image_size=96,
        query_dim=4,
        output_dim=3,
        output_activation="tanh",
        num_hidden=16,
        patch_size=7,
        patch_stride=4,
        top_k=10,
        data_dim=3,
        normalize_positions=True,
    )
    to_torch(base_agent_params, agent)
    return agent


#
# CMA-ES helpers (generic)
#
def from_torch(module: nn.Module):
    return np.concatenate([p.data.numpy().flatten() for p in module.parameters()])


def to_torch(params, module: nn.Module):
    ps, ts = list(module.parameters()), torch.Tensor(params)
    for p, p0 in zip(ps, ts.split([e.numel() for e in ps])):
        p.data.copy_(p0.view(p.size()))
    return module


def train(args):
    with np.load(args.base_from_pretrained) as data:
        base_agent_params = data['params'].flatten()
    if args.num_workers is None:
        args.num_workers = mp.cpu_count()-1
    env = SubprocVecEnv([make_env(base_agent_params, args.seed, i) for i in range(args.num_workers)])
    print(f"Running: {args.num_workers} workers")
    model = RecurrentPPO(
        'MlpLstmPolicy',
        env,
        verbose=1,
        learning_rate=0.003,
        policy_kwargs=dict(lstm_hidden_size=16, net_arch=[dict(pi=[], vf=[])])
    )
    print(model.policy)
    checkpoint_callback = CheckpointCallback(
        save_freq=100,
        save_path="sb3_logs/",
        name_prefix="exp2-topK-ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )    
    model.learn(total_timesteps=args.total_timesteps, log_interval=100, callback=checkpoint_callback)


def parse_args():
    parser = argparse.ArgumentParser("RL agent training with PPO")
    parser.add_argument("--seed", type=int, default=1143)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=256)
    parser.add_argument("--init-sigma", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--num-rollouts", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--num-eval-rollouts", type=int, default=64)
    parser.add_argument("--logs-dir", type=str, default="es_logs/exp1_topK_ppo_v0")
    parser.add_argument("--from-pretrained", type=Path, default=None)
    parser.add_argument("--base-from-pretrained", type=Path)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
    # if args.from_pretrained:
    #     with np.load(args.from_pretrained) as data:
    #         params = data['params'].flatten()
    #         evaluate(params, render=True)
    # else:
    #     train(args)
