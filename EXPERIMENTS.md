# Experiment Configurations

This document outlines the training details and hyperparameter settings that are needed to reproduce the results outlined in the paper.

There are some common assumptions made about the directory structures in these code samples. Make sure to change these if you want to run the code:
1. All checkpoints are saved at `/checkpoint/avikpal/<expt id>` directory. This is to allow easy restart of code in servers with preemption.
2. Again scripts that finetune models attempt to load a model from `/checkpoint/avikpal/*` directories.
3. Training is performed across 40 nodes for the examples. This option is configured through `mpirun -np <number of processes>`. You need to make sure that every 10 processes must have access to atleast 1 GPU with 12GB of memory.
4. There is some bug in Pytorch 1.6 which breaks JIT compilation for the code. So I have the environment variable `PYTORCH_JIT=0` set globally. You can avoid using it while generating rollouts though.

## Learning to Follow Traffic Signals

All the experiments here use the `MultiAgentRoadIntersectionFixedTrackDiscreteEnvironment`. This environment uses discrete actions. To use the continuous time variant of the same environment use `MultiAgentRoadIntersectionFixedTrackEnvironment`. In the first section we shall use no perception noise and simply increase the number of agents. In the 2nd part, we will gradually increase the perception noise and keep the number of agents fixed to 4.

### Analyzing the Effect of more agents during training

1. First we shall train with only 1 agent. This is not essential for convergence of environments with more agents, however, it shows that the actions are not at all correlated with the traffic signals (just as they shouldn't be).

```bash
mpirun -np 40 python -m sdriving.agents.ppo_distributed.train -s /checkpoint/avikpal/962782 --env MultiAgentRoadIntersectionFixedTrackDiscreteEnvironment --eid ckpt -se 32000 -e 10000 --pi-lr 1e-3 --vf-lr 1e-3 --seed 30860 --entropy-coeff 0.1 --target-kl 0.2 -ti 20 -wid 962782 --ac-kwargs "{\"hidden_sizes\": [256, 256], \"history_len\": 5, \"permutation_invariant\": true}" --env-kwargs "{\"horizon\": 200, \"nagents\": 1, \"lidar_noise\": 0.0, \"history_len\": 5, \"timesteps\": 10, \"npoints\": 100, \"turns\": false, \"learn_right_of_way\": false, \"default_color\": true, \"balance_cars\": true}"
```

2. Now we continue the training with 4 agents. For a little bit faster convergence we will finetune the model trained in the last experiment.

3. Again increase the agent count by 4 and in order to make it a bit more challenging we will break the symmetry in the road pockets by changing `balanced_cars` to `False`.
   
4. Finally train with 12 agents. We don't train any furthur simply because the environment becomes too cluttered (when the agents reach their goals) making learning very difficult.

### Analyzing the Effect of Increased Perception Noise


## Bi-Level Optimization for Lane Emergence

### Analysing the Effect of Increased Number of Agents while training


## Emergence of Fast Lane in a Highway


## Learning to Slow Down at Crosswalk

[TODO] Coding is still left with the new Simulator


## Learning Right of Way

* Can be observed in the Nuscenes Intersection but a simpler one using our 4 way map would be good


## Emergence of Minimum Safe Distance


## Learning to Communicate via Turn Signals

* Most likely will have to settle for the toy env


## Driving on Real World Maps