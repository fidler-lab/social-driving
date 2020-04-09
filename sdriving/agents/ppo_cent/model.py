from typing import List, Optional, Tuple, Union

import torch
from gym.spaces import Discrete
from gym.spaces import Tuple as GSTuple
from sdriving.agents.ppo_cent.utils import mlp
from torch import nn
from torch.distributions.categorical import Categorical


class PPOCategoricalActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Union[List[int], Tuple[int]],
        activation: torch.nn.Module,
        history_len: int,
    ):
        super().__init__()
        self.logits_net = mlp(
            [obs_dim] + list(hidden_sizes) + [act_dim], activation
        )
        self.lidar_features = nn.Sequential(
            nn.Conv1d(history_len, 1, 4, 2, 2, padding_mode="circular"),
            nn.Conv1d(1, 1, 4, 2, 2, padding_mode="circular"),
            nn.AdaptiveAvgPool1d(25),
        )
        self.history_len = history_len

    def _distribution(
        self, obs: Union[Tuple[torch.nn.Module], List[torch.nn.Module]]
    ):
        # Required for compatibility with older trained models
        if not hasattr(self, "history_len"):
            self.history_len = 1

        # Extract features from the lidar data
        bsize = obs[1].size(0) if obs[1].ndim > 1 else 1
        features = self.lidar_features(
            obs[1].view(bsize, self.history_len, -1)
        ).view(bsize, -1)
        if obs[1].ndim == 1:
            features = features.view(-1)

        logits = self.logits_net(torch.cat([obs[0], features], dim=-1))
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

    def _deterministic(
        self, obs: Union[Tuple[torch.nn.Module], List[torch.nn.Module]]
    ):
        # Required for compatibility with older trained models
        if not hasattr(self, "history_len"):
            self.history_len = 1

        # Extract features from the lidar data
        bsize = obs[1].size(0) if obs[1].ndim > 1 else 1
        features = self.lidar_features(
            obs[1].view(bsize, self.history_len, -1)
        ).view(bsize, -1)
        if obs[1].ndim == 1:
            features = features.view(-1)

        probs = self.logits_net(torch.cat([obs[0], features], dim=-1))
        return torch.argmax(probs, dim=-1)

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class PPOCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Union[List[int], Tuple[int]],
        activation: torch.nn.Module,
        history_len: int,
        nagents: int,
    ):
        super().__init__()
        self.v_net = mlp(
            [obs_dim * nagents] + list(hidden_sizes) + [1], activation
        )
        self.lidar_features = nn.Sequential(
            nn.Conv1d(history_len, 1, 4, 2, 2, padding_mode="circular"),
            nn.Conv1d(1, 1, 4, 2, 2, padding_mode="circular"),
            nn.AdaptiveAvgPool1d(25),
        )
        self.nagents = nagents
        self.history_len = history_len

    def forward(
        self,
        obs_list: List[Union[Tuple[torch.nn.Module], List[torch.nn.Module]]],
    ):
        assert len(obs_list) == self.nagents

        # For loading previously trained models
        if not hasattr(self, "history_len"):
            self.history_len = 1

        f_vecs = []
        state_vec = torch.cat([o for o, _ in obs_list], dim=-1)

        for obs in obs_list:
            bsize = obs[1].size(0) if obs[1].ndim > 1 else 1
            features = self.lidar_features(
                obs[1].view(bsize, self.history_len, -1)
            ).view(bsize, -1)
            if obs[1].ndim == 1:
                features = features.view(-1)
            f_vecs.append(features)
        f_vecs = torch.cat(f_vecs, dim=-1)

        return torch.squeeze(
            self.v_net(torch.cat([state_vec, f_vecs], dim=-1)), -1
        )  # Critical to ensure v has right shape.


class ActorCritic(nn.Module):
    def __init__(
        self,
        observation_space: GSTuple,
        action_space: Discrete,
        hidden_sizes: Union[List[int], Tuple[int]] = (64, 64),
        activation: torch.nn.Module = nn.Tanh,
        conv_layers: Optional[str] = None,
        history_len: int = 1,
        nagents: int = 1,
    ):
        super().__init__()

        obs_dim = 25 + observation_space[0].shape[0]

        self.nagents = nagents

        # policy builder depends on action space
        self.pi = PPOCategoricalActor(
            obs_dim, action_space.n, hidden_sizes, activation, history_len
        )

        # build value function
        self.v = PPOCritic(
            obs_dim, hidden_sizes, activation, history_len, nagents
        )

        if conv_layers is not None:
            ac = torch.load(conv_layers, map_location="cpu")
            self.v.lidar_features = ac.v.lidar_features
            for param in self.v.lidar_features.parameters():
                param.requires_grad = False
            self.pi.lidar_features = ac.pi.lidar_features
            for param in self.pi.lidar_features.parameters():
                param.requires_grad = False
            del ac

    def step(
        self, obs: List[Union[Tuple[torch.nn.Module], List[torch.nn.Module]]]
    ):
        actions = []
        log_probs = []
        with torch.no_grad():
            for o in obs:
                pi = self.pi._distribution(o)
                a = pi.sample()
                logp_a = self.pi._log_prob_from_distribution(pi, a)

                actions.append(a)
                log_probs.append(logp_a)
            v = self.v(obs)
        return actions, v, log_probs

    def act(
        self,
        obs: Union[Tuple[torch.nn.Module], List[torch.nn.Module]],
        deterministic: bool = False,
    ):
        if deterministic:
            return self.pi._deterministic(obs)
        return self.pi._distribution(obs).sample()
