# Social Driving
Design multi-agent environments and simple reward functions such that social driving behavior emerges

## Table of Contents

* [Installation](#installation)
* [Agents Module](#agents-module)
    * [Training an Agent](#training-an-agent)
    * [Trainer Description](#trainer-description)
* [Scripts Module](#scripts-module)
    * [Generating Rollouts](#generating-rollouts)
* [Environments Module](#environments-module)
    * [Available Environments](#available-environments)
    * [Environment Configuration](#environment-configuration)
    * [Writing New Environments](#writing-new-environments)
* [TSIM (Traffic Simulator) Module](#tsim-traffic-simulator-module)
* [Additional Suggestions for Debugging](#additional-suggestions-for-debugging)

## Installation

Create a new conda environment

```
$ conda create --name sdriving python=3.7
$ conda activate sdriving
```

Follow the instructions given [here](https://pytorch.org/get-started/locally/) to install pytorch.

Install the [Nuscenes DevKit](https://github.com/nutonomy/nuscenes-devkit/) for access to realistic simulation environments.

Finally install sdriving using the following instructions.

```
$ git clone https://github.com/fidler-lab/social-driving.git sdriving
$ cd sdriving
$ python setup.py develop
```

If you need to train an agent, some additional configuration is needed. You can skip these steps if you are only interested in generating rollouts / don't care about logging on the server (using `WANDB_MODE=dryrun`).

1. Create a Wandb account using `wandb login` and following the steps given there.
2. A project named `Social Driving` will be automatically created. If anything goes wrong here please open an issue.

## Agents Module

### Training an Agent

Three variants of PPO are currently implemented:

| Method                             | Python Module   | Information                                       | Action Space                     | Observation Space | Compatible Environments |
|------------------------------------|-----------------|---------------------------------------------------|----------------------------------|-------------------|-------------------------|
| PPO Distributed Centralized Critic | ppo_distributed | Centralized Training with Decentralized Execution | Box / Discrete                   | Tuple             | 1, 2, 3, 4              |
| PPO OneStep                        | ppo_one_step    | Optimized Implementation for Single Step RL       | Box / Discrete                   | Box               | 5                       |
| PPO Alternating Optimization       | ppo_altopt      | PPO with Bi-Level Optimization                    | (Box / Discrete, Box / Discrete) | (Box, Tuple)      | 6                       |


To get the configurable parameters for the trainers use the following command:

```
$ python -m sdriving.agents.<module>.train --help
```

Example usage:

```
$ mpirun --np 16 python -m sdriving.agents.ppo_distributed.train.py -s /checkpoint/ --env MultiAgentRoadIntersectionFixedTrackDiscreteEnvironment --eid ckpt -se 16000 -e 10000 --pi-lr 1e-3 --vf-lr 1e-3 --seed 4567 --entropy-coeff 0.01 --target-kl 0.2 -ti 20 -wid 908515 --ac-kwargs "{\"hidden_sizes\": [256, 256], \"history_len\": 5, \"permutation_invariant\": true}" --env-kwargs "{\"horizon\": 250, \"nagents\": 12, \"mode\": 2, \"lidar_noise\": 0.0, \"history_len\": 5, \"balance_cars\": true, \"timesteps\": 10, \"npoints\": 100, \"turns\": false, \"default_color\": true, \"balance_cars\": false}"
```

**NOTE**: Even though it might be possible to run the training scripts without `mpirun` / `horovodrun`, I haven't tested them exhaustively. So just use `mpirun --np 1` if you need a single task.

### Trainer Description

The trainers make some assumptions about the environment which must be satisfied over and above the action & observation space restrictions.

1. `PPO Distributed Centralized Critic`: The environment `step` function takes an action whose batch dim is of size `N` (number of agents). Currently it doesn't support variable `N` overtime but it is very simple to implement (so open an issue if needed). It needs to return the observation for next timestep, a BoolTensor of size `N x 1` specifying if simulation for that agent is completed, Reward Tensor of size `N x 1`, and `info` similar to OpenAI Gym Environments.

2. `PPO OneStep`: The `step` function must return the Reward Tensor of size `N x 1`. By design it will assume the horizon is of size 1. This model will most likely **never** converge for any other horizon size.

3. `PPO Alternating Optimization`: The `step` function takes 2 arguments. The first one represents `stage`, which when 0 is used to perform the single step RL action. The returned value when stage = 1, should be the observation of the controller. Any furthur call to `step` follows the same API as `PPO Distributed Centralized Critic`

## Scripts Module

### Generating Rollouts

To generate rollouts for any of the registered environments run

```
python -m sdriving.scripts.rollout --help
```

To test proper functioning of an environment a good check is to generate a rollout for the same without passing any pretrained model. This will simulate the environment by sampling random actions from the action space. Even though this is not an exhaustive test and the training can still fail, it ensures that the pipeline for sampling environment observations and the episode rollout strategy used in the trainer works properly.

## Environments Module

### Available Environments

1. `MultiAgentRoadIntersectionBicycleKinematicsEnvironment`
2. `MultiAgentRoadIntersectionBicycleKinematicsDiscreteEnvironment`
3. `MultiAgentRoadIntersectionFixedTrackEnvironment`
4. `MultiAgentRoadIntersectionFixedTrackDiscreteEnvironment`
5. `MultiAgentOneShotSplinePredicitonEnvironment`
6. `MultiAgentIntersectionSplineAccelerationDiscreteEnvironment`

### Environment Configuration

[TODO]

### Writing New Environments

[TODO]

## TSIM (Traffic Simulator) Module

[TODO]

## Additional Suggestions for Debugging and Performance

* The environments heavily use JIT compilation for speed up. But it might return NaN gradients in some rare situations. The training will explicitly fail in such conditions. In these situations use `PYTORCH_JIT=0`.
* A minor bottleneck might be horovod caching. Disable caching with `HOROVOD_CACHE_CAPACITY=0`.
* By default we simulate the environment on CPU, this is performant for low nagents due to the high kernel launch overhead. In case you want to use our `tsim` and `agents` modules for simulating a large number of vehicles, uncomment [this line](https://github.com/fidler-lab/social-driving/blob/b59dede27ebfed22e2c41165a79b8fce95f308da/sdriving/agents/ppo_distributed/ppo.py#L73) or the corresponding line in other trainers.
