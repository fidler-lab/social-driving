from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from sdriving.agents.utils import mlp
from torch import nn


class PPOWaypointCentralizedCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Union[List[int], Tuple[int]],
        activation: torch.nn.Module,
        nagents: int,
    ):
        super().__init__()
        self.v_net = mlp(
            [obs_dim * nagents] + list(hidden_sizes) + [1], activation,
        )
        self.nagents = nagents

    def forward(self, obs_list: List[torch.Tensor]):
        assert len(obs_list) == self.nagents
        obs = torch.cat(obs_list, dim=-1)
        return self.v_net(obs).squeeze(-1)


class PPOLidarCentralizedCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Union[List[int], Tuple[int]],
        activation: torch.nn.Module,
        history_len: int,
        nagents: int,
        feature_dim: int = 25,
    ):
        super().__init__()
        self.v_net = mlp(
            [(obs_dim + feature_dim) * nagents] + list(hidden_sizes) + [1],
            activation,
        )
        self.lidar_features = nn.Sequential(
            nn.Conv1d(history_len, 1, 4, 2, 2, padding_mode="circular"),
            nn.Conv1d(1, 1, 4, 2, 2, padding_mode="circular"),
            nn.AdaptiveAvgPool1d(feature_dim),
        )
        self.history_len = history_len
        self.nagents = nagents

    def forward(
        self, obs_list: List[Union[Tuple[torch.Tensor], List[torch.Tensor]]]
    ):
        assert len(obs_list) == self.nagents

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
        )


class PPOWaypointPermutationInvariantCentralizedCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Union[List[int], Tuple[int]],
        activation: torch.nn.Module,
    ):
        super().__init__()
        self.f_net = mlp([obs_dim, hidden_sizes[0]], activation,)
        self.v_net = mlp(list(hidden_sizes[1:]) + [1], activation)

    def forward(self, obs_list: List[torch.Tensor]):
        f_vecs = []
        for obs in obs_list:
            f_vecs.append(self.f_net(obs))
        state_vec = sum(f_vecs) / len(f_vecs)
        return self.v_net(state_vec).squeeze(-1)


class PPOLidarPermutationInvariantCentralizedCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Union[List[int], Tuple[int]],
        activation: torch.nn.Module,
        history_len: int,
        feature_dim: int = 25,
    ):
        super().__init__()
        self.feature_net = mlp(
            [obs_dim + feature_dim] + [hidden_sizes[0]], activation,
        )
        self.lidar_features = nn.Sequential(
            nn.Conv1d(history_len, 1, 4, 2, 2, padding_mode="circular"),
            nn.Conv1d(1, 1, 4, 2, 2, padding_mode="circular"),
            nn.AdaptiveAvgPool1d(feature_dim),
        )
        self.v_net = mlp(list(hidden_sizes) + [1], activation,)
        self.history_len = history_len

    def forward(
        self, obs_list: List[Union[Tuple[torch.Tensor], List[torch.Tensor]]]
    ):
        f_vecs = []

        for obs in obs_list:
            bsize = obs[1].size(0) if obs[1].ndim > 1 else 1
            features = self.lidar_features(
                obs[1].view(bsize, self.history_len, -1)
            ).view(bsize, -1)
            f_vecs.append(
                self.feature_net(
                    torch.cat([obs[0].view(bsize, -1), features], dim=-1)
                )
            )
        state_vec = sum(f_vecs) / len(f_vecs)

        return torch.squeeze(self.v_net(state_vec), -1)


class PPOLidarDecentralizedCritic(PPOLidarCentralizedCritic):
    def __init__(self, *args, **kwargs):
        if len(args) >= 5:
            args = list(args)
            args[4] = 1
        else:
            kwargs["nagents"] = 1
        super().__init__(*args, **kwargs)

    def forward(self, obs: Union[Tuple[torch.Tensor], List[torch.Tensor]]):
        bsize = obs[1].size(0) if obs[1].ndim > 1 else 1
        features = self.lidar_features(
            obs[1].view(bsize, self.history_len, -1)
        ).view(bsize, -1)
        if obs[1].ndim == 1:
            features = features.view(-1)

        return torch.squeeze(
            self.v_net(torch.cat([obs[0], features], dim=-1)), -1
        )
