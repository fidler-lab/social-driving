# social-driving
Design multi-agent environments and simple reward functions such that social
driving behavior emerges


## Installation

We use MPI and OpenAI SpinningUp in our implementation of PPO. These need to
be setup before installing `sdriving`. Please follow the installation
instructions on [this page](https://spinningup.openai.com/en/latest/user/installation.html)
for proper setup.

Additionally pytorch needs to be installed. Follow the instructions given
[here](https://pytorch.org/get-started/locally/).

Finally install sdriving using the following instructions.

```
$ git clone https://github.com/fidler-lab/social-driving.git sdriving
$ cd sdriving
$ python setup.py install
```

## Environment Rollouts

Use the [rollout.py](https://github.com/fidler-lab/social-driving/blob/master/sdriving/scripts/rollout.py)
script for testing agents in a
[registered environment](https://github.com/fidler-lab/social-driving/blob/master/sdriving/envs/__init__.py)

```
usage: rollout.py [-h] [-s SAVE_DIR] [-m MODEL_SAVE_PATH]
                  [-tep NUM_TEST_EPISODES] --env ENV [--env-kwargs ENV_KWARGS]
                  [--dummy-run] [--no-render] [--algo ALGO] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -s SAVE_DIR, --save-dir SAVE_DIR
  -m MODEL_SAVE_PATH, --model-save-path MODEL_SAVE_PATH
  -tep NUM_TEST_EPISODES, --num-test-episodes NUM_TEST_EPISODES
  --env ENV
  --env-kwargs ENV_KWARGS
  --dummy-run
  --no-render
  --algo ALGO
  -v, --verbose
```

## Training Agents

Use the [train.py](https://github.com/fidler-lab/social-driving/blob/master/sdriving/scripts/train.py)
script for training agents in a
[registered environment](https://github.com/fidler-lab/social-driving/blob/master/sdriving/envs/__init__.py)

```
usage: train.py [-h] --eid EID -s SAVE_DIR [-e EPOCHS] [--seed SEED]
                [--cpu CPU] [-rs REPLAY_SIZE] [--gamma GAMMA]
                [--target-kl TARGET_KL] [--vf-lr VF_LR] [--pi-lr PI_LR]
                [--clip-ratio CLIP_RATIO] [-se STEPS_PER_EPOCH] [--lam LAM]
                [-tep NUM_TEST_EPISODES] [-tpi TRAIN_PI_ITERS]
                [-tvi TRAIN_V_ITERS] [-f SAVE_FREQ] [--ac-kwargs AC_KWARGS]
                [--logger-kwargs LOGGER_KWARGS] --env ENV
                [--env-kwargs ENV_KWARGS] [--resume] [--render-train]
                [--render-test] [--model-checkpoint MODEL_CHECKPOINT]
                [--tboard] [--centralized-critic]

optional arguments:
  -h, --help            show this help message and exit
  --eid EID
  -s SAVE_DIR, --save-dir SAVE_DIR
  -e EPOCHS, --epochs EPOCHS
  --seed SEED
  --cpu CPU
  -rs REPLAY_SIZE, --replay-size REPLAY_SIZE
  --gamma GAMMA
  --target-kl TARGET_KL
  --vf-lr VF_LR
  --pi-lr PI_LR
  --clip-ratio CLIP_RATIO
  -se STEPS_PER_EPOCH, --steps-per-epoch STEPS_PER_EPOCH
  --lam LAM
  -tep NUM_TEST_EPISODES, --num-test-episodes NUM_TEST_EPISODES
  -tpi TRAIN_PI_ITERS, --train_pi_iters TRAIN_PI_ITERS
  -tvi TRAIN_V_ITERS, --train_v_iters TRAIN_V_ITERS
  -f SAVE_FREQ, --save-freq SAVE_FREQ
  --ac-kwargs AC_KWARGS
  --logger-kwargs LOGGER_KWARGS
  --env ENV
  --env-kwargs ENV_KWARGS
  --resume
  --render-train
  --render-test
  --model-checkpoint MODEL_CHECKPOINT
  --tboard
  --centralized-critic, -cc
```

#### Notes

1. For now, the only available agent is a Centralized PPO for MultiAgent
   Models. So pass the `-cc` argument by default
2. Emperically 16 tasks can be run on a single GPU. So the number of GPUs
   available must be >= Number of CPU // 16 
