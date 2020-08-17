# social-driving
Design multi-agent environments and simple reward functions such that social driving behavior emerges


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

## Training an Agent

Three variants of PPO are currently implemented:

| Method                             | Python Module   | Information                                       | Action Space                     | Observation Space | Compatible Environments |
|------------------------------------|-----------------|---------------------------------------------------|----------------------------------|-------------------|-------------------------|
| PPO Distributed Centralized Critic | ppo_distributed | Centralized Training with Decentralized Execution | Box / Discrete                   | Tuple             | 1, 2, 3, 4, 5           |
| PPO OneStep                        | ppo_one_step    | Optimized Implementation for Single Step RL       | Box / Discrete                   | Box               | 5                       |
| PPO Alternating Optimization       | ppo_altopt      | PPO with Bi-Level Optimization                    | (Box / Discrete, Box / Discrete) | (Box, Tuple)      | 6                       |


To get the configurable parameters for the trainers use the following command

```
$ python -m sdriving.agents.<module>.train --help
```

Example usage:

```
$ mpirun --np 16 python -m sdriving.agents.ppo_distributed.train.py -s /checkpoint/ --env MultiAgentRoadIntersectionFixedTrackDiscreteEnvironment --eid ckpt -se 16000 -e 10000 --pi-lr 1e-3 --vf-lr 1e-3 --seed 4567 --entropy-coeff 0.01 --target-kl 0.2 -ti 20 -wid 908515 --ac-kwargs "{\"hidden_sizes\": [256, 256], \"history_len\": 5, \"permutation_invariant\": true}" --env-kwargs "{\"horizon\": 250, \"nagents\": 12, \"mode\": 2, \"lidar_noise\": 0.0, \"history_len\": 5, \"balance_cars\": true, \"timesteps\": 10, \"npoints\": 100, \"turns\": false, \"default_color\": true, \"balance_cars\": false}"
```

## Available Environments

1. `MultiAgentRoadIntersectionBicycleKinematicsEnvironment`
2. `MultiAgentRoadIntersectionBicycleKinematicsDiscreteEnvironment`
3. `MultiAgentRoadIntersectionFixedTrackEnvironment`
4. `MultiAgentRoadIntersectionFixedTrackDiscreteEnvironment`
5. `MultiAgentOneShotSplinePredicitonEnvironment`
6. `MultiAgentIntersectionSplineAccelerationDiscreteEnvironment`
