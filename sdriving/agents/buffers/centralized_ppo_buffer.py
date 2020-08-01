from collections import namedtuple
from typing import Optional, Union

import numpy as np
import torch
from sdriving.agents.utils import (
    combined_shape,
    discount_cumsum,
    hvd_statistics_scalar,
)


BufferReturn = namedtuple(
    "BufferReturn", ["obs", "lidar", "act", "ret", "adv", "logp", "vest"]
)


def allocate_zeros_tensor(size, device):
    return torch.zeros(size, dtype=torch.float32, device=device).pin_memory()


class CentralizedPPOBuffer:
    def __init__(
        self,
        state_dim: int,
        lidar_dim: int,
        act_dim: int,
        size: int,
        gamma: float = 0.99,
        lam: float = 0.95,
        nagents: int = 1,
        device=torch.device("cpu"),
    ):
        self.agent_list = [f"agent_{i}" for i in range(nagents)]
        self.agent_list_to_batch = {f"agent_{i}": i for i in range(nagents)}

        func = lambda: allocate_zeros_tensor(
            combined_shape(size, state_dim, batch=nagents), device
        )

        self.state_buf = func()
        self.lidar_buf = func()
        self.act_buf = func()

        func = lambda: allocate_zeros_tensor(
            combined_shape(size, batch=nagents), device
        )

        self.vest_buf = func()
        self.adv_buf = func()
        self.rew_buf = func()
        self.ret_buf = func()
        self.val_buf = func()
        self.logp_buf = func()

        self.gamma, self.lam = gamma, lam
        self.max_size = size
        self.nagents = nagents
        self.ptr = [0 for _ in range(nagents)]
        self.path_start_idx = [0 for _ in range(nagents)]

    def store(self, a_id: str, obs, lidar, act, rew, val, logp):
        """Append one timestep of agent-environment interaction to the
        buffer."""
        b = self.agent_list_to_batch[a_id]
        idx = self.ptr[b]
        assert idx < self.max_size

        self.state_buf[b, idx] = obs
        if lidar is not None:
            self.lidar_buf[b, idx] = lidar

        self.act_buf[b, idx] = act
        self.rew_buf[b, idx] = rew
        self.val_buf[b, idx] = val
        self.logp_buf[b, idx] = logp
        self.ptr[a_id] = idx + 1

    def finish_path(
        self, last_val: Optional[Union[dict, int, torch.Tensor]] = None
    ):
        if last_val is None:
            last_val = {a_id: 0 for a_id in self.agent_list}
        if isinstance(last_val, (int, torch.Tensor)):
            last_val = [last_val] ** len(self.agent_list)

        for a_id in self.agent_list:
            b = self.agent_list_to_batch[a_id]
            path_slice = slice(self.path_start_idx[b], self.ptr[b])
            rews = torch.cat([self.rew_buf[b, path_slice], last_val[a_id]])
            vals = torch.cat([self.val_buf[b, path_slice], last_val[a_id]])

            # the next two lines implement GAE-Lambda advantage calculation
            deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
            self.adv_buf[b, path_slice] = discount_cumsum(
                deltas, self.gamma * self.lam
            )

            # the next line computes rewards-to-go,
            # to be targets for the value function
            self.ret_buf[b, path_slice] = discount_cumsum(rews, self.gamma)[
                :-1
            ]

            self.path_start_idx[b] = self.ptr[b]

    def get(self):
        """Call this at the end of an epoch to get all of the data from the
        buffer, with advantages appropriately normalized (shifted to have mean
        zero and std one).
        Also, resets some pointers in the buffer.
        """
        for a_id in self.agent_list:
            b = self.agent_list_to_batch[a_id]
            self.ptr[b], self.path_start_idx[b] = 0, 0

            # the next two lines implement the advantage normalization trick
            adv_mean, adv_std = hvd_statistics_scalar(self.adv_buf[b])
            self.adv_buf[b] = (self.adv_buf[b] - adv_mean) / (adv_std + 1e-7)
        return BufferReturn(
            obs=self.state_buf[a_id],
            lidar=self.lidar_buf[a_id],
            act=self.act_buf[a_id],
            ret=self.ret_buf[a_id],
            adv=self.adv_buf[a_id],
            logp=self.logp_buf[a_id],
            vest=self.val_buf[a_id],
        )
