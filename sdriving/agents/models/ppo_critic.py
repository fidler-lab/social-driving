from typing import List, Optional, Tuple, Union

import torch
from gym.spaces import Box, Discrete
from torch import nn

from sdriving.agents.utils import mlp


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0.0)


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
            [obs_dim * nagents] + list(hidden_sizes) + [1],
            activation,
        )
        self.nagents = nagents
        self.apply(init_weights)

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
        self.apply(init_weights)

    def forward(self, obs: Tuple[torch.Tensor]):
        state_vec, lidar_vec = obs
        state_vec = state_vec.view(-1, state_vec.size(-1))
        bsize, no_batch = (
            (1, True) if lidar_vec.ndim == 2 else (lidar_vec.size(1), False)
        )
        nagents = lidar_vec.size(0)

        lidar_vec = lidar_vec.view(bsize * nagents, -1).view(
            bsize * nagents, self.history_len, -1
        )

        feature_vec = self.lidar_features(lidar_vec).squeeze(1)

        val_est = (
            torch.cat([state_vec, feature_vec], dim=-1)
            .view(nagents, bsize, -1)
            .permute(1, 0, 2)
        )
        val_est = torch.squeeze(self.v_net(val_est), -1)

        return val_est.repeat(nagents)


class PPOWaypointPermutationInvariantCentralizedCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Union[List[int], Tuple[int]],
        activation: torch.nn.Module,
    ):
        super().__init__()
        self.f_net = mlp(
            [obs_dim, hidden_sizes[0]],
            activation,
        )
        self.v_net = mlp(list(hidden_sizes[1:]) + [1], activation)
        self.apply(init_weights)

    def forward(self, obs_list: List[torch.Tensor]):
        x = obs_list[0]
        bsize, no_batch = (1, True) if x.ndim == 1 else (x.size(0), False)
        f_vecs = []
        for obs in obs_list:
            if no_batch:
                obs = obs.unsqueeze(0)
            f_vecs.append(obs)
        state_vec = (
            self.f_net(torch.cat(f_vecs, dim=0))
            .view(len(obs_list), bsize, -1)
            .mean(0)
        )
        val_est = self.v_net(state_vec).squeeze(-1)
        return val_est.squeeze(0) if no_batch else val_est


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
            [obs_dim + feature_dim] + [hidden_sizes[0]],
            activation,
        )
        self.lidar_features = nn.Sequential(
            nn.Conv1d(history_len, 1, 4, 2, 2, padding_mode="circular"),
            nn.Conv1d(1, 1, 4, 2, 2, padding_mode="circular"),
            nn.AdaptiveAvgPool1d(feature_dim),
        )
        self.v_net = mlp(
            list(hidden_sizes) + [1],
            activation,
        )
        self.history_len = history_len
        self.apply(init_weights)

    def forward(
        self, obs: Tuple[torch.Tensor], mask: Optional[torch.Tensor] = None
    ):
        state_vec, lidar_vec = obs
        state_vec = state_vec.view(-1, state_vec.size(-1))
        bsize, no_batch = (
            (1, True) if lidar_vec.ndim == 2 else (lidar_vec.size(1), False)
        )
        nagents = lidar_vec.size(0)

        lidar_vec = lidar_vec.view(bsize * nagents, -1).view(
            bsize * nagents, self.history_len, -1
        )
        feature_vec = self.lidar_features(lidar_vec).squeeze(1)

        val_est = self.feature_net(torch.cat([state_vec, feature_vec], dim=-1))
        val_est = val_est.view(nagents, bsize, val_est.size(-1))
        div_factor = val_est.size(0)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            val_est = val_est * mask
            div_factor = mask.sum(0)
        val_est = val_est.sum(0) / div_factor
        val_est = torch.squeeze(self.v_net(val_est), -1)

        return val_est.repeat(nagents)
