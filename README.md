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
% python -m pip install 'gym[box2d]'==0.26.2 matplotlib==3.6.1 cma==3.2.2
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
Iterat #Fevals   function value  axis ratio  sigma  min&max std  t[m:s]
...
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

### Exp 3. Fix LSTM

Fix LSTM from the pre-trained solution, use CMA-ES to learn Q/K linear layers. Almost instantaneously emits a pretty good policy (after a few iterations): 

```
Iterat #Fevals   function value  axis ratio  sigma  min&max std  t[m:s]
...
   10   2560 -8.760685376761674e+02 1.0e+00 8.47e-02  8e-02  8e-02 995:20.5
...
   20   5120 -9.032011463678441e+02 1.0e+00 8.42e-02  8e-02  8e-02 2696:20.2   
```

How to launch:

```shell
% python exp3-topK-qk-cmaes.py \
  --base-from-pretrained pretrained/original.npz \
  --resume es_logs/exp3_topK_qk_cmaes_v0/best.pkl \
  --eval-every 25 \
  --verbose
```

**Exp 3.1.** To test out the hypothesis that `SelfAttention` would be more robust if linear layers (Q/K) have no bias, we also run experiment with `bias=False` and `query_dim=2` (max level compression for the information from each patch).

To run:
```shell
% python exp3-topK-qk-cmaes.py \
  --base-from-pretrained pretrained/original.npz \
  --eval-every 25 \
  --verbose \
  --query-dim 2 \
  --no-bias

Exp3Agent(
  (attention): SelfAttention(
    (fc_q): Linear(in_features=147, out_features=2, bias=False)
    (fc_k): Linear(in_features=147, out_features=2, bias=False)
  )
)
(128_w,256)-aCMA-ES (mu_w=66.9,w_1=3%) in dimension 588 (seed=1143, Sun Nov 20 23:48:31 2022)
```

Quite a few of randomly generated agents from the very first population of the algorithm, in fact, show decent performance on the task:

```shell
Fitness min/mean/max: 120.45/352.30/733.91
Fitness min/mean/max: 11.77/212.05/527.22
Fitness min/mean/max: 134.75/498.31/901.46
Fitness min/mean/max: 96.20/383.97/900.35
```

After the first iteration we get agent with a quite strong performance, rapid improvements to follow:

```
Iterat #Fevals   function value  axis ratio  sigma  min&max std  t[m:s]
    1    256 -7.360439490799304e+02 1.0e+00 9.50e-02  9e-02  9e-02 22:38.5
...
   10   2560 -8.451596726780031e+02 1.0e+00 8.65e-02  9e-02  9e-02 857:11.7
```

### Exp 4. Frame Stacking instead of LSTM

Take pre-trained `Q/K` layers, learn MLP policy over stacked frames using the same algorithm (frame stacking to replace recurrency in the policy). Learning is rapid though somewhat unstable:

```
Iterat #Fevals   function value  axis ratio  sigma  min&max std  t[m:s]
...
   37   9472 -8.008960468870334e+02 1.0e+00 9.65e-02  1e-01  1e-01 610:18.9
```

How to launch:

```shell
% python exp4-topK-stack-cmaes.py \
  --base-from-pretrained pretrained/original.npz \
  --resume es_logs/exp4_topK_stack_cmaes_v0/best.pkl \
  --eval-every 25 \
  --verbose
```

To train using different number of frames (the default is 2), use `--num-frames` argument. The network size will be automatically adjusted accordinly to a new state space shaping.

```shell
% python exp4-topK-stack-cmaes.py \
  --base-from-pretrained pretrained/original.npz \
  --num-frames 4

Exp4Agent(
  (controller): Sequential(
    (0): Linear(in_features=80, out_features=16, bias=True)
    (1): ReLU()
    (2): Linear(in_features=16, out_features=3, bias=True)
    (3): Tanh()
  )
)
(128_w,256)-aCMA-ES (mu_w=66.9,w_1=3%) in dimension 1347 (seed=1143, Mon Nov  7 23:56:22 2022)
```

### Exp 5. RNN instead of LSTM 

There are multiple interesting angle here:
* number of neurons is much smaller compared to LSTM
* `ReLU` is used as RNN's non-linearity function, imposing additional inductive bias on the solution

It learns blazingly fast compared to LSMT:

```
Iterat #Fevals   function value  axis ratio  sigma  min&max std  t[m:s]
...
   28   7168 -6.381508565694667e+02 1.0e+00 9.50e-02  9e-02  1e-01 436:05.6
...
   43  11008 -8.484943160119002e+02 1.0e+00 1.06e-01  1e-01  1e-01 1233:14.8
...
  111  28416 -8.902925877126818e+02 1.1e+00 1.16e-01  1e-01  1e-01 8106:54.6
```

How to run:

```shell
% python exp5-topK-rnn-cmaes.py \
  --base-from-pretrained pretrained/original.npz \
  --eval-every 25 \
  --verbose

Exp5Agent(
  (rnn): RNN(20, 16)
  (fc): Linear(in_features=16, out_features=3, bias=True)
  (activation): Tanh()
)
(128_w,256)-aCMA-ES (mu_w=66.9,w_1=3%) in dimension 659 (seed=1143, Thu Nov 09 22:11:20 2022)
```