from typing import Optional, Union

import numpy as np
import torch
from sdriving.agents.utils import (
    combined_shape,
    discount_cumsum,
    mpi_statistics_scalar,
)


class OneStepPPOBuffer:
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        size: int,
        max_agents: float = 1,
    ):
        size = size * max_agents
        self.state_buf = torch.zeros(
            combined_shape(size, state_dim), dtype=torch.float32
        )
        self.act_buf = torch.zeros(
            combined_shape(size, act_dim), dtype=torch.float32
        )
        self.rew_buf = torch.zeros(size, dtype=torch.float32)
        self.logp_buf = torch.zeros(size, dtype=torch.float32)

        self.max_size = size
        self.ptr = 0
        
    def store(self, obs, act, rew, logp):
        """Append one timestep of agent-environment interaction to the
        buffer."""
        assert (
            self.ptr < self.max_size
        )  # buffer has to have room so you can store
        self.state_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.logp_buf[self.ptr] = logp
        self.ptr = self.ptr + 1
        
    def get(self):
        ptr_copy = self.ptr
        self.ptr = 0

        return dict(
            obs=self.state_buf[:ptr_copy],
            act=self.act_buf[:ptr_copy],
            rew=self.rew_buf[:ptr_copy],
            logp=self.logp_buf[:ptr_copy],
        )