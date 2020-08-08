from typing import Optional, Union

import numpy as np
import torch

# from sdriving.agents.utils import (
#     combined_shape,
#     discount_cumsum,
#     mpi_statistics_scalar,
# )


class CentralizedPPOBufferVariableNagents:
    def __init__(
        self,
        state_dim: int,
        lidar_dim: int,
        act_dim: int,
        size: int,
        gamma: float = 0.99,
        lam: float = 0.95,
    ):
        self.agent_list = {}
        self.state_buf = {}
        self.lidar_buf = {}
        self.act_buf = {}
        self.adv_buf = {}
        self.vest_buf = {}
        self.val_buf = {}
        self.ret_buf = {}
        self.logp_buf = {}
        self.rew_buf = {}

        self.lidar_dim, self.state_dim, self.act_dim = (
            lidar_dim,
            state_dim,
            act_dim,
        )
        self.gamma, self.lam = gamma, lam
        self.max_size = size
        self.ptr = {a_id: 0 for a_id in self.agent_list}
        self.path_start_idx = {a_id: 0 for a_id in self.agent_list}

    def _add_new_agent(self, a_id: str):
        self.agent_list[a_id] = None
        self.state_buf[a_id] = torch.zeros(
            combined_shape(self.max_size, self.state_dim), dtype=torch.float32
        )
        self.lidar_buf[a_id] = torch.zeros(
            combined_shape(self.max_size, self.lidar_dim), dtype=torch.float32
        )
        self.act_buf[a_id] = torch.zeros(
            combined_shape(self.max_size, self.act_dim), dtype=torch.float32
        )
        self.vest_buf[a_id] = torch.zeros(self.max_size, dtype=torch.float32)
        self.adv_buf[a_id] = torch.zeros(self.max_size, dtype=torch.float32)
        self.rew_buf[a_id] = torch.zeros(self.max_size, dtype=torch.float32)
        self.ret_buf[a_id] = torch.zeros(self.max_size, dtype=torch.float32)
        self.val_buf[a_id] = torch.zeros(self.max_size, dtype=torch.float32)
        self.logp_buf[a_id] = torch.zeros(self.max_size, dtype=torch.float32)
        self.ptr[a_id] = 0
        self.path_start_idx[a_id] = 0

    def _get_agent_num(self, a_id: str):
        return int(a_id.split("_")[1])

    def add_new_agents(self, a_id: str):
        if a_id in self.agent_list:
            return
        else:
            if a_id in self.state_buf:
                self.ptr[a_id] = 0
                self.path_start_idx[a_id] = 0
            else:
                self._add_new_agent(a_id)

    def store(self, a_id: str, obs, lidar, act, rew, val, logp):
        """Append one timestep of agent-environment interaction to the
        buffer."""
        self.add_new_agents(a_id)
        assert (
            self.ptr[a_id] < self.max_size
        )  # buffer has to have room so you can store
        self.state_buf[a_id][self.ptr[a_id]] = obs
        self.lidar_buf[a_id][self.ptr[a_id]] = lidar
        self.act_buf[a_id][self.ptr[a_id]] = act
        self.rew_buf[a_id][self.ptr[a_id]] = rew
        self.val_buf[a_id][self.ptr[a_id]] = val
        self.logp_buf[a_id][self.ptr[a_id]] = logp
        self.ptr[a_id] = self.ptr[a_id] + 1

    def finish_path(
        self, last_val: Optional[Union[dict, int, torch.Tensor]] = None
    ):
        if last_val is None:
            last_val = {a_id: 0 for a_id in self.agent_list}
        if isinstance(last_val, (int, torch.Tensor)):
            last_val = {a_id: last_val for a_id in self.agent_list}

        for a_id in self.agent_list.keys():
            path_slice = slice(self.path_start_idx[a_id], self.ptr[a_id])
            rews = np.append(
                self.rew_buf[a_id][path_slice].detach().numpy(), last_val[a_id]
            )
            vals = np.append(
                self.val_buf[a_id][path_slice].detach().numpy(), last_val[a_id]
            )

            # the next two lines implement GAE-Lambda advantage calculation
            deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
            self.adv_buf[a_id][path_slice] = discount_cumsum(
                deltas, self.gamma * self.lam
            )

            # the next line computes rewards-to-go,
            # to be targets for the value function
            self.ret_buf[a_id][path_slice] = discount_cumsum(rews, self.gamma)[
                :-1
            ]

            self.path_start_idx[a_id] = self.ptr[a_id]

    def get(self):
        # NOTE: We do not normalize the entire batch. Rather this needs to be
        # done at the minibatch level while training
        a_ids = list(self.agent_list.keys())
        self.agent_list = {}
        slices = {}
        for a_id in a_ids:
            slices[a_id] = slice(0, self.ptr[a_id])
            self.ptr[a_id], self.path_start_idx[a_id] = 0, 0
        return {
            a_id: dict(
                obs=self.state_buf[a_id][slices[a_id]],
                lidar=self.lidar_buf[a_id][slices[a_id]],
                act=self.act_buf[a_id][slices[a_id]],
                ret=self.ret_buf[a_id][slices[a_id]],
                adv=self.adv_buf[a_id][slices[a_id]],
                logp=self.logp_buf[a_id][slices[a_id]],
                vest=self.val_buf[a_id][slices[a_id]],
            )
            for a_id in a_ids
        }
