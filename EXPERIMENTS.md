# Experiment Configurations

This document outlines the training details and hyperparameter settings that are needed to reproduce the results outlined in the paper.

There are some common assumptions made about the directory structures and environment variables in these code samples. Make sure to change/apply these if you want to run the code:

1. All checkpoints are saved at `/checkpoint/avikpal/<expt id>` directory. This is to allow easy restart of code in servers with preemption.
2. Again scripts that finetune models attempt to load a model from `/checkpoint/avikpal/*` directories.
3. Training is performed in a distributed manner using horovod. This option is configured through `mpirun -np <number of processes>`. You need to make sure that every 10 processes must have access to atleast 1 GPU with 12GB of memory.
4. There is some bug in Pytorch 1.6 which breaks JIT compilation for the code. So I have the environment variable `PYTORCH_JIT=0` set globally. You can avoid using it while generating rollouts though.
5. Also set `HOROVOD_CACHE_CAPACITY=0` for some minor speedup while interprocess communication.

## Learning to Follow Traffic Signals

All the experiments here use the `MultiAgentRoadIntersectionFixedTrackDiscreteEnvironment`. This environment uses discrete actions. To use the continuous time variant of the same environment use `MultiAgentRoadIntersectionFixedTrackEnvironment`. In the first section we shall use no perception noise and simply increase the number of agents. In the 2nd part, we will gradually increase the perception noise and keep the number of agents fixed to 4.

### Analyzing the Effect of more agents during training

* First we shall train with only 1 agent. This is not essential for convergence of environments with more agents, however, it shows that the actions are not at all correlated with the traffic signals (just as they shouldn't be).

```bash
mpirun -np 40 python -m sdriving.agents.ppo_distributed.train -s /checkpoint/avikpal/962782 --env MultiAgentRoadIntersectionFixedTrackDiscreteEnvironment --eid ckpt -se 32000 -e 60 --pi-lr 1e-3 --vf-lr 1e-3 --seed 30860 --entropy-coeff 0.001 --target-kl 0.2 -ti 20 -wid 962782 --ac-kwargs "{\"hidden_sizes\": [256, 256], \"history_len\": 5, \"permutation_invariant\": true}" --env-kwargs "{\"horizon\": 200, \"nagents\": 1, \"lidar_noise\": 0.0, \"history_len\": 5, \"timesteps\": 10, \"npoints\": 100, \"turns\": true, \"learn_right_of_way\": true, \"default_color\": true, \"balance_cars\": true}"
```

* Now we continue the training with 4 agents. For a little bit faster convergence we will finetune the model trained in the last experiment.

```bash
mpirun -np 20 python -m sdriving.agents.ppo_distributed.train -s /checkpoint/avikpal/994752 --env MultiAgentRoadIntersectionFixedTrackDiscreteEnvironment --eid ckpt -se 32000 -e 45 --pi-lr 1e-3 --vf-lr 1e-3 --seed 5688 --entropy-coeff 0.0001 --target-kl 0.2 -ti 20 -wid 994752 --ac-kwargs "{\"hidden_sizes\": [256, 256], \"history_len\": 5, \"permutation_invariant\": true}" --env-kwargs "{\"horizon\": 250, \"nagents\": 4, \"lidar_noise\": 0.0, \"history_len\": 5, \"timesteps\": 10, \"npoints\": 100, \"turns\": false, \"learn_right_of_way\": false, \"default_color\": true, \"balance_cars\": true}"
```

* Again increase the agent count by 4 and in order to make it a bit more challenging we will break the symmetry in the road pockets by changing `balanced_cars` to `False`.

```bash
mpirun -np 20 python -m sdriving.agents.ppo_distributed.train -s /checkpoint/avikpal/997782 --env MultiAgentRoadIntersectionFixedTrackDiscreteEnvironment --eid ckpt -se 32000 -e 10000 --pi-lr 1e-3 --vf-lr 1e-3 --seed 18443 --entropy-coeff 0.0001 --target-kl 0.2 -ti 20 -wid 997782 --resume --model-checkpoint /checkpoint/avikpal/994752/ckpt/checkpoints/ckpt_latest.pth --ac-kwargs "{\"hidden_sizes\": [256, 256], \"history_len\": 5, \"permutation_invariant\": true}" --env-kwargs "{\"horizon\": 250, \"nagents\": 8, \"lidar_noise\": 0.0, \"history_len\": 5, \"timesteps\": 10, \"npoints\": 100, \"turns\": false, \"learn_right_of_way\": false, \"default_color\": true, \"balance_cars\": false}"
```

* Finally train with 12 agents. We don't train any furthur simply because the environment becomes too cluttered (when the agents reach their goals) making learning very difficult.

```bash
```

### Analyzing the Effect of Increased Perception Noise

* For the 4 agent model with 0 perception noise we use the model trained in the [previous section](#analysing-the-effect-of-more-agents-while-training).

* Training an agent with 25% perception noise.

```bash
mpirun -np 20 python -m sdriving.agents.ppo_distributed.train -s /checkpoint/avikpal/986404 --env MultiAgentRoadIntersectionFixedTrackDiscreteEnvironment --eid ckpt -se 32000 -e 75 --pi-lr 1e-3 --vf-lr 1e-3 --seed 17927 --entropy-coeff 0.0001 --target-kl 0.2 -ti 20 -wid 986404 --ac-kwargs "{\"hidden_sizes\": [256, 256], \"history_len\": 5, \"permutation_invariant\": true}" --env-kwargs "{\"horizon\": 250, \"nagents\": 4, \"lidar_noise\": 0.25, \"history_len\": 5, \"timesteps\": 10, \"npoints\": 100, \"turns\": false, \"learn_right_of_way\": false, \"default_color\": true, \"balance_cars\": true}"
```
  
* Training an agent with 50% perception noise.

```bash
mpirun -np 20 python -m sdriving.agents.ppo_distributed.train -s /checkpoint/avikpal/986405 --env MultiAgentRoadIntersectionFixedTrackDiscreteEnvironment --eid ckpt -se 32000 -e 75 --pi-lr 1e-3 --vf-lr 1e-3 --seed 11407 --entropy-coeff 0.0001 --target-kl 0.2 -ti 20 -wid 986405 --ac-kwargs "{\"hidden_sizes\": [256, 256], \"history_len\": 5, \"permutation_invariant\": true}" --env-kwargs "{\"horizon\": 250, \"nagents\": 4, \"lidar_noise\": 0.50, \"history_len\": 5, \"timesteps\": 10, \"npoints\": 100, \"turns\": false, \"learn_right_of_way\": false, \"default_color\": true, \"balance_cars\": true}"
```
  
* Training an agent with 75% perception noise.

```bash
mpirun -np 20 python -m sdriving.agents.ppo_distributed.train -s /checkpoint/avikpal/986406 --env MultiAgentRoadIntersectionFixedTrackDiscreteEnvironment --eid ckpt -se 32000 -e 75 --pi-lr 1e-3 --vf-lr 1e-3 --seed 31085 --entropy-coeff 0.0001 --target-kl 0.2 -ti 20 -wid 986406 --ac-kwargs "{\"hidden_sizes\": [256, 256], \"history_len\": 5, \"permutation_invariant\": true}" --env-kwargs "{\"horizon\": 250, \"nagents\": 4, \"lidar_noise\": 0.75, \"history_len\": 5, \"timesteps\": 10, \"npoints\": 100, \"turns\": false, \"learn_right_of_way\": false, \"default_color\": true, \"balance_cars\": true}"
```
  
* Training an agent with 100% perception noise (aka. a blind agent receiving no lidar feedback).

```bash
mpirun -np 20 python -m sdriving.agents.ppo_distributed.train -s /checkpoint/avikpal/986409 --env MultiAgentRoadIntersectionFixedTrackDiscreteEnvironment --eid ckpt -se 32000 -e 50 --pi-lr 1e-3 --vf-lr 1e-3 --seed 15657 --entropy-coeff 0.0001 --target-kl 0.2 -ti 20 -wid 986409 --ac-kwargs "{\"hidden_sizes\": [256, 256], \"history_len\": 5, \"permutation_invariant\": true}" --env-kwargs "{\"horizon\": 250, \"nagents\": 4, \"lidar_noise\": 1.0, \"history_len\": 5, \"timesteps\": 10, \"npoints\": 100, \"turns\": false, \"learn_right_of_way\": false, \"default_color\": true, \"balance_cars\": true}"
```

## Bi-Level Optimization for Lane Emergence

### Analysing the Effect of Increased Number of Agents while training

## Emergence of Fast Lane in a Highway

```bash
mpirun -np 20 python -m sdriving.agents.ppo_altopt.train -s /checkpoint/avikpal/933722 --env MultiAgentHighwaySplineAccelerationDiscreteModel --eid ckpt -se1 160 -se2 8000 -e 10000 --pi-lr 1e-3 --vf-lr 1e-3 --spline-lr 1e-3 --seed 18442 --entropy-coeff 0.001 --target-kl 0.3 -ti 20 -wid 933722 --ac-kwargs "{\"hidden_sizes\": [256, 256], \"history_len\": 5, \"permutation_invariant\": true}" --actor-kwargs "{\"hidden_sizes\": [32, 32]}" --env-kwargs "{\"horizon\": 250, \"nagents\": 10, \"lidar_noise\": 0.0, \"history_len\": 5, \"timesteps\": 10, \"npoints\": 100}" -wid 933722
```

## Learning to Slow Down at Crosswalk

```bash
mpirun -np 20 python -m sdriving.agents.ppo_distributed.train -s /checkpoint/avikpal/983690 --env MultiAgentHighwayPedestriansFixedTrackDiscreteModel --eid ckpt -se 32000 -e 10000 --pi-lr 1e-3 --vf-lr 1e-3 --seed 21053 --entropy-coeff 0.0001 --target-kl 0.3 -ti 20 -wid 983690 --ac-kwargs "{\"hidden_sizes\": [64, 64], \"history_len\": 5, \"permutation_invariant\": true}" --env-kwargs "{\"horizon\": 300, \"nagents\": 4, \"lidar_noise\": 0.0, \"history_len\": 5, \"timesteps\": 10, \"npoints\": 100, \"lateral_noise_variance\": 0.0}" -wid 983690
```

## Learning Right of Way

## Emergence of Minimum Safe Distance

This experiment doesn't require training any new model. It emerges in any of the environments using ~ 8 agents. We simple evaluate using any of the trained models from [1](#emergence-of-fast-lane-in-a-highway), [2](#learning-right-of-way), or [3](#learning-to-communicate-via-turn-signals).

## Learning to Communicate via Turn Signals

## Driving on Nuscenes
