# Experiment Configurations

This document outlines the training details and hyperparameter settings that are needed to reproduce the results outlined in the paper

## Learning to Follow Traffic Signals

All the experiments here use the `MultiAgentRoadIntersectionFixedTrackDiscreteEnvironment`. This environment uses discrete actions. To use the continuous time variant of the same environment use `MultiAgentRoadIntersectionFixedTrackEnvironment`. In the first section we shall use no perception noise and simply increase the number of agents. In the 2nd part, we will gradually increase the perception noise and keep the number of agents fixed to 4.

### Analyzing the Effect of more agents during training

1. First we shall train with only 1 agent. This is not essential for convergence of environments with more agents, however, it shows that the actions are not at all correlated with the traffic signals (just as they shouldn't be).
   
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