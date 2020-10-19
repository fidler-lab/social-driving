from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from torch import nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from sdriving.agents.utils import mlp


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0.0)


class PPOActor(nn.Module):
    def sample(self, pi):
        raise NotImplementedError

    def _distribution(self, obs):
        raise NotImplementedError

    def _deterministic(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

    def forward(self, obs, act=None):
        x = obs if isinstance(obs, torch.Tensor) else obs[0]
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            if x.ndim == 3:
                prev_shape = act.shape[:2]
                act = (
                    act.view(-1)
                    if act.ndim == 2
                    else act.view(-1, act.size(-1))
                )
            logp_a = self._log_prob_from_distribution(pi, act)
            if x.ndim == 3:
                logp_a = logp_a.view(*prev_shape)
        else:
            act = self.sample(pi)
            logp_a = self._log_prob_from_distribution(pi, act)
            if x.ndim == 3:
                act = act.view(x.size(0), x.size(1), -1)
                logp_a = logp_a.view(*x.shape[:2])
        return pi, act, logp_a

    def act(self, obs, deterministic: bool = True):
        if deterministic:
            ret_val = self._deterministic(obs)
        else:
            ret_val = self.sample(self._distribution(obs))
        x = obs if isinstance(obs, torch.Tensor) else obs[0]
        # {N or N x A} or {N x B x (1 or A)} when batched
        return (
            ret_val if x.ndim == 2 else ret_val.view(x.size(0), x.size(1), -1)
        )


class PPOCategoricalActor(PPOActor):
    def sample(self, pi):
        return pi.sample()

    def _get_logits(self, obs):
        raise NotImplementedError

    def _deterministic(self, obs):
        return torch.argmax(self._get_logits(obs), dim=-1)

    def _distribution(self, obs):
        return Categorical(logits=self._get_logits(obs))


class PPOWaypointCategoricalActor(PPOCategoricalActor):
    def __init__(
        self,
        obs_dim: int,
        act_space: Discrete,
        hidden_sizes: Union[List[int], Tuple[int]] = [256, 256],
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.deviation_net = mlp(
            [obs_dim] + list(hidden_sizes) + [act_space.n],
            activation,
        )
        self.apply(init_weights)

    def _get_logits(self, obs: torch.Tensor):
        return self.deviation_net(
            obs if obs.ndim == 2 else obs.view(-1, obs.size(-1))
        )


class PPOLidarCategoricalActor(PPOCategoricalActor):
    def __init__(
        self,
        obs_dim: int,
        act_space: Discrete,
        hidden_sizes: Union[List[int], Tuple[int]],
        activation: torch.nn.Module,
        history_len: int,
        feature_dim: int = 25,
    ):
        super().__init__()
        self.logits_net = mlp(
            [obs_dim + feature_dim] + list(hidden_sizes) + [act_space.n],
            activation,
        )
        self.lidar_features = nn.Sequential(
            nn.Conv1d(history_len, 1, 4, 2, 2, padding_mode="circular"),
            nn.Conv1d(1, 1, 4, 2, 2, padding_mode="circular"),
            nn.AdaptiveAvgPool1d(feature_dim),
        )
        self.history_len = history_len
        self.apply(init_weights)

    def _get_logits(self, obs: Tuple[torch.Tensor]):
        state_vec, lidar_vec = obs
        bsize = 1 if lidar_vec.ndim == 2 else lidar_vec.size(1)
        nagents = lidar_vec.size(0)
        state_vec = state_vec.view(-1, state_vec.size(-1))
        lidar_vec = lidar_vec.view(-1, lidar_vec.size(-1))
        features = self.lidar_features(
            lidar_vec.view(bsize * nagents, self.history_len, -1)
        ).squeeze(1)

        return self.logits_net(
            torch.cat([state_vec, features], dim=-1)
        )  # (N x B) x L


class PPOGaussianActor(PPOActor):
    def sample(self, pi):
        return self.act_scale(torch.tanh(pi.rsample()))

    def _get_mu_std(self, obs, std):
        raise NotImplementedError

    def _distribution(self, obs):
        return Normal(*self._get_mu_std(obs, True))

    def _deterministic(self, obs):
        return self.act_scale(torch.tanh(self._get_mu_std(obs, False)))

    def act_scale(self, act):
        if not act.device == self.act_high.device:
            self.act_high = self.act_high.to(act.device)
            self.act_low = self.act_low.to(act.device)
        return (act + 1) * 0.5 * (self.act_high - self.act_low) + self.act_low

    def act_rescale(self, act):
        return self.atanh(
            2 * (act - self.act_low) / (self.act_high - self.act_low) - 1.0
        )

    @staticmethod
    def atanh(x):
        return 0.5 * torch.log(torch.abs((1 + x + 1e-7) / (1 - x + 1e-7)))

    def _log_prob_from_distribution(self, pi, act):
        act = self.act_rescale(act)
        if act.ndim == 1:
            act = act.unsqueeze(0)
        logp = pi.log_prob(act).sum(axis=-1)
        logp = logp - (2 * (np.log(2) - act - F.softplus(-2 * act))).sum(
            axis=1
        )
        return logp.view(-1)


class PPOWaypointGaussianActor(PPOGaussianActor):
    def __init__(
        self,
        obs_dim: int,
        act_space: Box,
        hidden_sizes: Union[List[int], Tuple[int]] = [256, 256],
        activation: torch.nn.Module = torch.nn.ReLU,
    ):
        super().__init__()
        act_dim = act_space.shape[0]
        self.act_high = torch.as_tensor(act_space.high)
        self.act_low = torch.as_tensor(act_space.low)
        self.net = mlp(
            [obs_dim] + list(hidden_sizes),
            activation,
        )
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))
        self.apply(init_weights)

    def _get_mu_std(self, obs: torch.Tensor, std: bool = True):
        out = self.net(obs.view(-1, obs.size(-1)))

        if std:
            return (
                self.mu_layer(out),
                torch.exp(torch.clamp(self.log_std, -20.0, 2.0)),
            )
            return mu, std
        else:
            return self.mu_layer(out)


class PPOLidarGaussianActor(PPOGaussianActor):
    def __init__(
        self,
        obs_dim: int,
        act_space: Box,
        hidden_sizes: Union[List[int], Tuple[int]],
        activation: torch.nn.Module,
        history_len: int,
        feature_dim: int = 25,
    ):
        super().__init__()
        act_dim = act_space.shape[0]
        self.act_high = torch.as_tensor(act_space.high)
        self.act_low = torch.as_tensor(act_space.low)
        self.net = mlp(
            [obs_dim + feature_dim] + list(hidden_sizes),
            activation,
            activation,
        )
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.lidar_features = nn.Sequential(
            nn.Conv1d(history_len, 1, 4, 2, 2, padding_mode="circular"),
            nn.Conv1d(1, 1, 4, 2, 2, padding_mode="circular"),
            nn.AdaptiveAvgPool1d(feature_dim),
        )
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim)).unsqueeze(0)
        self.history_len = history_len
        self.apply(init_weights)

    def _get_mu_std(self, obs: Tuple[torch.Tensor], std: bool = True):
        state_vec, lidar_vec = obs
        bsize = 1 if lidar_vec.ndim == 2 else lidar_vec.size(1)
        nagents = lidar_vec.size(0)
        state_vec = state_vec.view(-1, state_vec.size(-1))
        lidar_vec = lidar_vec.view(-1, lidar_vec.size(-1))
        features = self.lidar_features(
            lidar_vec.view(bsize * nagents, self.history_len, -1)
        ).squeeze(1)

        vec = self.net(torch.cat([state_vec, features], dim=-1))
        if std:
            return (
                self.mu_layer(vec),
                torch.exp(torch.clamp(self.log_std, -2.0, 20.0))
                .repeat(vec.size(0), 1)
                .type_as(vec),
            )
        else:
            return self.mu_layer(vec)