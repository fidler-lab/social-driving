from typing import Optional, Union

import numpy as np
import torch
from sdriving.agents.utils import (
    combined_shape,
    discount_cumsum,
    mpi_statistics_scalar,
)


class ReinforceBuffer:
    """A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-
    Lambda) for calculating the advantages of state-action pairs."""

    def __init__(
        self, state_dim: int, act_dim: int, size: int, nagents: float = 1,
    ):
        size = size * nagents
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
        """Call this at the end of an epoch to get all of the data from the
        buffer, with advantages appropriately normalized (shifted to have mean
        zero and std one).
        Also, resets some pointers in the buffer.
        """
        ptr_copy = self.ptr
        self.ptr = 0
        self.agent_id_idxs = {}

        # the next two lines implement the advantage normalization trick
        # adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf.numpy())
        # self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-7)
        return dict(
            observation=self.state_buf[:ptr_copy],
            action=self.act_buf[:ptr_copy],
            reward=self.rew_buf[:ptr_copy],
            logp=self.logp_buf[:ptr_copy],
        )