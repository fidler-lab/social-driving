from itertools import product
from typing import List, Tuple, Union

import torch
from gym.spaces import Box, Discrete
from gym.spaces import Tuple as GSTuple
from torch import nn

from sdriving.agents.models.ppo_actor import *
from sdriving.agents.models.ppo_critic import *


class PPOActorCritic(nn.Module):
    def __init__(self, actor, critic, centralized):
        super().__init__()
        self.pi = actor
        self.v = critic
        self.centralized = centralized

    def _step_centralized(self, obs):
        """
        obs: Should be of a valid format which can be given as input
             to both Actor and Critic. The standard input involves
             every tensor to be of shape (N x B x O) or (N x O)
             where N is the number of agents and B is the batch size
        """
        with torch.no_grad():
            _, actions, log_probs = self.pi(obs)
            v = self.v(obs)
        return actions, v, log_probs

    def step(self, obs):
        return self._step_centralized(obs)

    def act(self, obs, deterministic: bool = True):
        if deterministic:
            return self.pi._deterministic(obs)
        return self.pi.sample(self.pi._distribution(obs))


class PPOWaypointActorCritic(PPOActorCritic):
    def __init__(
        self,
        observation_space: GSTuple,
        action_space: Union[Discrete, Box],
        hidden_sizes: Union[List[int], Tuple[int]] = (64, 64),
        activation: torch.nn.Module = nn.Tanh,
        nagents: int = 1,
        centralized: bool = False,
        permutation_invariant: bool = False,
    ):

        obs_dim = observation_space.shape[0]

        if isinstance(action_space, Box):
            pi = PPOWaypointGaussianActor(
                obs_dim,
                action_space,
                hidden_sizes,
                activation,
            )
        elif isinstance(action_space, Discrete):
            pi = PPOWaypointCategoricalActor(
                obs_dim,
                action_space,
                hidden_sizes,
                activation,
            )
        else:
            raise Exception(
                "Only Box and Discrete Action Spaces are supported"
            )

        if centralized:
            if permutation_invariant:
                v = PPOWaypointPermutationInvariantCentralizedCritic(
                    obs_dim, hidden_sizes, activation
                )
            else:
                v = PPOWaypointCentralizedCritic(
                    obs_dim, hidden_sizes, activation, nagents
                )
        else:
            raise Exception("Decentralized Training not available")

        super().__init__(pi, v, centralized)


class PPOLidarActorCritic(PPOActorCritic):
    def __init__(
        self,
        observation_space: GSTuple,
        action_space: Union[Discrete, Box],
        hidden_sizes: Union[List[int], Tuple[int]] = (64, 64),
        activation: torch.nn.Module = nn.Tanh,
        history_len: int = 1,
        feature_dim: int = 25,
        nagents: int = 1,
        centralized: bool = False,
        permutation_invariant: bool = False,
    ):

        obs_dim = observation_space[0].shape[0]

        if isinstance(action_space, Box):
            pi = PPOLidarGaussianActor(
                obs_dim,
                action_space,
                hidden_sizes,
                activation,
                history_len,
                feature_dim,
            )
        elif isinstance(action_space, Discrete):
            pi = PPOLidarCategoricalActor(
                obs_dim,
                action_space,
                hidden_sizes,
                activation,
                history_len,
                feature_dim,
            )
        else:
            raise Exception(
                "Only Box and Discrete Action Spaces are supported"
            )

        if centralized:
            if permutation_invariant:
                v = PPOLidarPermutationInvariantCentralizedCritic(
                    obs_dim, hidden_sizes, activation, history_len, feature_dim
                )
            else:
                v = PPOLidarCentralizedCritic(
                    obs_dim,
                    hidden_sizes,
                    activation,
                    history_len,
                    nagents,
                    feature_dim,
                )
        else:
            raise Exception("Decentralized Training not available")

        super().__init__(pi, v, centralized)
