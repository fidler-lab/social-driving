from typing import Optional, Union

import numpy as np
import torch
from sdriving.agents.utils import (
    combined_shape,
    discount_cumsum,
    mpi_statistics_scalar,
)


class DecentralizedPPOBuffer:
    def __init__(
        self,
        state_dim: int,
        lidar_dim: int,
        act_dim: int,
        size: int,
        gamma: float = 0.99,
        lam: float = 0.95,
        max_agents: float = 1,
    ):
        size = size * max_agents
        self.state_buf = torch.zeros(
            combined_shape(size, state_dim), dtype=torch.float32
        )
        self.lidar_buf = torch.zeros(
            combined_shape(size, lidar_dim), dtype=torch.float32
        )
        self.act_buf = torch.zeros(
            combined_shape(size, act_dim), dtype=torch.float32
        )
        self.adv_buf = torch.zeros(size, dtype=torch.float32)
        self.rew_buf = torch.zeros(size, dtype=torch.float32)
        self.ret_buf = torch.zeros(size, dtype=torch.float32)
        self.val_buf = torch.zeros(size, dtype=torch.float32)
        self.logp_buf = torch.zeros(size, dtype=torch.float32)

        self.agent_id_idxs = {}

        self.gamma, self.lam = gamma, lam
        self.max_size = size
        self.ptr = 0

    def store(self, a_id, obs, lidar, act, rew, val, logp):
        """Append one timestep of agent-environment interaction to the
        buffer."""
        assert (
            self.ptr < self.max_size
        )  # buffer has to have room so you can store
        if a_id not in self.agent_id_idxs:
            self.agent_id_idxs[a_id] = [self.ptr]
        else:
            self.agent_id_idxs[a_id].append(self.ptr)
        self.state_buf[self.ptr] = obs
        self.lidar_buf[self.ptr] = lidar
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr = self.ptr + 1

    def finish_path(
        self,
        a_id: Optional[str],
        last_val: Optional[Union[int, torch.Tensor]] = None,
    ):
        last_val = 0 if last_val is None else last_val

        if a_id is None:
            a_ids = list(self.agent_id_idxs.keys())
        else:
            a_ids = [a_id]

        for a_id in a_ids:
            if a_id not in self.agent_id_idxs:
                continue
            path_slice = self.agent_id_idxs[a_id]
            rews = np.append(
                self.rew_buf[path_slice].detach().numpy(), last_val
            )
            vals = np.append(
                self.val_buf[path_slice].detach().numpy(), last_val
            )

            # the next two lines implement GAE-Lambda advantage calculation
            deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
            self.adv_buf[path_slice] = discount_cumsum(
                deltas, self.gamma * self.lam
            ).float()

            # the next line computes rewards-to-go,
            # to be targets for the value function
            self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[
                :-1
            ].float()

            del self.agent_id_idxs[a_id]

    def get(self):
        ptr_copy = self.ptr
        self.ptr = 0
        self.agent_id_idxs = {}

        # the next two lines implement the advantage normalization trick
        # adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf.numpy())
        # self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-7)
        return dict(
            obs=self.state_buf[:ptr_copy],
            lidar=self.lidar_buf[:ptr_copy],
            act=self.act_buf[:ptr_copy],
            ret=self.ret_buf[:ptr_copy],
            adv=self.adv_buf[:ptr_copy],
            logp=self.logp_buf[:ptr_copy],
            vest=self.val_buf[:ptr_copy],
        )