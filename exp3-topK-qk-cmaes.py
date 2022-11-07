import argparse
from functools import partial
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
from typing import Any, Dict, Tuple

import cma
import gym
import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torchvision import transforms


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
        query_dim,
        output_dim,
        output_activation,
        num_hidden,
        patch_size,
        data_dim,
        top_k,
    ):
        super().__init__()
        self.attention = SelfAttention(
            data_dim=data_dim * patch_size ** 2,
            dim_q=query_dim,
        )
        self.controller = LSTMController(
            input_dim=top_k * 2,
            output_dim=output_dim,
            num_hidden=num_hidden,
            output_activation=output_activation,
        )
        self.eval()

    def forward(self, x):
        return self.controller(x).squeeze()

    def step(self, centers):
        with torch.no_grad():
            action = self.forward(centers).numpy()
        return action, None

    def reset(self):
        self.controller.reset()


class Exp3Agent(nn.Module):

    def __init__(
        self,
        image_size,
        query_dim,
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
        pass


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
        self.neg_reward_seq = 0
        self.steps_count = 0
        return self.env.reset()

    def overwrite_terminate_flag(self, reward):
        if self.neg_reward_cap == 0:
            # no need to terminate early
            return False
        self.neg_reward_seq = 0 if reward >= 0 else self.neg_reward_seq + 1
        out_of_tracks = 0 < self.neg_reward_cap < self.neg_reward_seq
        overtime = 0 < self.steps_cap <= self.steps_count
        return out_of_tracks or overtime

    def overwrite_action(self, centers):
        action, _ = self.base_agent.step(centers)
        return action

    def step(self, action):
        action = self.overwrite_action(action)
        self.steps_count += 1
        action = action * self.action_range + self.action_mean
        obs, reward, done, timeout, info = self.env.step(action)
        done = done or self.overwrite_terminate_flag(reward)
        return obs, reward, done, timeout, info


#
# ES training loop (strategy, init params, loop, eval, checkpoints)
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


def make_env(base_agent_params, evaluate: bool = False, render: bool = False):
    render_mode = "human" if render else None
    env = gym.make("CarRacing-v2", verbose=False, render_mode=render_mode)
    kwargs = dict(neg_reward_cap=20, steps_cap=1000) if not evaluate else {}
    base_agent = make_base_agent(base_agent_params)
    env = CarRacingWrapper(env, base_agent, **kwargs)
    return env


def make_base_agent(base_agent_params):
    agent = CarRacingAgent(
        query_dim=4,
        output_dim=3,
        output_activation="tanh",
        num_hidden=16,
        patch_size=7,
        top_k=10,
        data_dim=3,
    )
    vector_to_parameters(torch.Tensor(base_agent_params), agent.parameters())
    return agent


def make_agent(params=None):
    agent = Exp3Agent(
        image_size=96,
        query_dim=4,
        patch_size=7,
        patch_stride=4,
        top_k=10,
        data_dim=3,
        normalize_positions=True,
    )
    if params is not None:
        vector_to_parameters(torch.Tensor(params), agent.parameters())
    return agent


#
# CMA-ES helpers (generic)
#

# XXX: save all models based on the iteration?
def save_checkpoint(folder, es, best_solution):
    os.makedirs(folder, exist_ok=True)
    with open(f"{folder}/best.pkl", "wb") as f:
        pickle.dump({"es": es, "best": best_solution}, f)


def load_checkpoint(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
        return data["es"], data["best"]


def get_fitness(base_agent_params, n_samples: int, params: np.ndarray, verbose: bool = False) -> float:
    env = make_env(base_agent_params)
    agent = make_agent(params)
    rewards = np.array([rollout(env, agent)[0] for _ in range(n_samples)])
    avg_reward = rewards.mean()
    if verbose:
        print(f"Fitness min/mean/max: {rewards.min():.2f}/{avg_reward:.2f}/{rewards.max():.2f}")
    return params, -avg_reward


def evaluate(base_agent_params, params, render: bool = False) -> float:
    env = make_env(base_agent_params, evaluate=True, render=render) # no need for early termination when evaluating
    agent = make_agent(params)
    reward, _ = rollout(env, agent)
    return reward


# NOTE: multiprocessing module uses pickle that fails when dealing
# with lambdas (globally visible function is required)
def evaluate_cb(base_agent_params, params, _idx: int, verbose: bool = True) -> float:
    reward = evaluate(base_agent_params, params)
    if verbose:
        print(f"Evaluation reward: {reward}")
    return reward


def train(args):
    with np.load(args.base_from_pretrained) as data:
        base_agent_params = data['params'].flatten()
    if args.resume:
        es, best_ever = load_checkpoint(args.resume)
    else:
        init_agent = make_agent()
        print(init_agent)
        init_params = parameters_to_vector(init_agent.parameters()).numpy()
        es = cma.CMAEvolutionStrategy(
            init_params,
            args.init_sigma,
            {"popsize": args.population_size, "seed": args.seed, "maxiter": args.max_iter}
        )
        best_ever = cma.optimization_tools.BestSolution()
    if not args.num_workers:
        args.num_workers = mp.cpu_count() - 1
    current_step = 0
    with mp.Pool(processes=args.num_workers) as pool:
        while not es.stop():
            current_step += 1
            solutions = es.ask()
            fitness = list(pool.imap_unordered(partial(get_fitness, base_agent_params, args.num_rollouts, verbose=args.verbose), solutions))
            es.tell(*zip(*fitness))
            es.disp()
            best_ever.update(es.best)
            save_checkpoint(args.logs_dir, es, best_ever)
            if 0 == current_step % args.eval_every:
                fitness = pool.map(
                    partial(evaluate_cb, base_agent_params, es.result.xfavorite, verbose=args.verbose),
                    range(args.num_eval_rollouts)
                )
                print(f"Evaluation: step={current_step} fitness={np.mean(fitness)}")
        es.result_pretty()


def parse_args():
    parser = argparse.ArgumentParser("RL agent training with ES")
    parser.add_argument("--seed", type=int, default=1143)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=256)
    parser.add_argument("--init-sigma", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--num-rollouts", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--num-eval-rollouts", type=int, default=64)
    parser.add_argument("--logs-dir", type=str, default="es_logs/exp3_topK_qk_cmaes_v0")
    parser.add_argument("--from-pretrained", type=Path, default=None)
    parser.add_argument("--base-from-pretrained", type=Path)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)