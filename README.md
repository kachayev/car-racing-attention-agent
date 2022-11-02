# Neuroevolution of Self-Interpretable Agents

Train the agent to drive a car over proceduraly generated track (`CarRacing-v2`). The model consists of a self-attention applied to input signlas paired with tiny LSTM. Optimization is done using Covariance Matrix Adaptation Evolution Strategy (CMA-ES).

The model and training algorithm presented here is an attempt to reproduce the result from the paper ["Neuroevolution of Self-Interpretable Agents"](https://arxiv.org/abs/2003.08165) by Yujin Tang, Duong Nguyen, and David Ha. The official code could be find [here](https://github.com/google/brain-tokyo-workshop/tree/master/AttentionAgent).

Both neural networks needed for the solution are implemented in `PyTorch`, CMA-ES is done with `cma` package. The implementation followed the design principles behind [`CleanRL`](https://github.com/vwxyzjn/cleanrl): a single script that contains all the coded necessary (including various helpers) to make it much easier to grasp the solution in its entirety.

## Discussion

Model and training hyper parameters are kept the same (to match the paper as close as possible).

After 4 days of training on a 32 CPUs machine, it gets to the agent with evaluation performance 903 +/- 12 points. Which is slighly worse then the final result stated in the paper (914 +/- 15) but it still outperformce agents describe in previous papers (e.g. GA and PPO).

## Install

The easiest way to prepare the environment is to use [`conda`](https://docs.conda.io/en/latest/miniconda.html):

```shell
% conda create -n cmaes python==3.9
% conda activate cmaes
% conda install numpy==1.23.3 pytorch==1.10.2 torchvision==0.11.3
% python -m pip install 'gym[box2d]'==0.26.2 matplotlib==3.6.1
```

## Training

There's a single script to run:

```shell
% python train.py
CarRacingAgent(
  (attention): SelfAttention(
    (fc_q): Linear(in_features=147, out_features=4, bias=True)
    (fc_k): Linear(in_features=147, out_features=4, bias=True)
  )
  (controller): LSTMController(
    (lstm): LSTM(20, 16)
    (fc): Linear(in_features=16, out_features=3, bias=True)
    (activation): Tanh()
  )
)
(128_w,256)-aCMA-ES (mu_w=66.9,w_1=3%) in dimension 3667 (seed=1143, Wed Jan 26 16:38:16 2022)
...
```

The script automatically scales to the number of cores available on your machine by spawning multiple processes. To put a limit on max CPUs used for the training, specifie desired limit using `--num-workers` option.

The full list of configuration options:

```shell
% python train.py --help
usage: RL agent training with ES [-h] [--seed SEED] [--resume RESUME] [--num-workers NUM_WORKERS] [--population-size POPULATION_SIZE] [--init-sigma INIT_SIGMA] [--max-iter MAX_ITER]
                                 [--num-rollouts NUM_ROLLOUTS] [--eval-every EVAL_EVERY] [--num-eval-rollouts NUM_EVAL_ROLLOUTS] [--logs-dir LOGS_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED
  --resume RESUME
  --num-workers NUM_WORKERS
  --population-size POPULATION_SIZE
  --init-sigma INIT_SIGMA
  --max-iter MAX_ITER
  --num-rollouts NUM_ROLLOUTS
  --eval-every EVAL_EVERY
  --num-eval-rollouts NUM_EVAL_ROLLOUTS
  --logs-dir LOGS_DIR
```

## Evaluation

To make sure both the agent and the environemnt are compatible with the original paper, the repo also contains set of original weights re-packed to the suitable format (see `pretrained/` folder with the archive). To run the agent use `--from-pretrained` flag:

```shell
% python train.py --from-pretrained pretrained/original.npz
```

## Experiments

### Exp 1. Fix Q/K

Fix Q/K from the pre-trained solution, use CMA-ES to learn LSTM (the controller). Doesn't get the same performance. Takes a long time, seems to be stuck after 175+ iterations:

```
  179  45824 -8.881190132640309e+02 1.0e+00 8.08e-02  8e-02  8e-02 5298:03.3
```

How to launch:

```shell
% python exp1-topK-lstm-cmaes.py \
  --base-from-pretrained pretrained/original.npz \
  --resume es_logs/exp1_topK_cmaes_v0/best.pkl \
  --eval-every 25 \
  --verbose
```