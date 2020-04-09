from typing import Optional, Union

import numpy as np
import scipy.signal

import torch
import torch.nn as nn
from spinup.utils.mpi_tools import mpi_avg, mpi_statistics_scalar, num_procs


def trainable_parameters(net):
    return list(filter(lambda x: x.requires_grad, net.parameters()))


def mpi_avg_grads(module):
    """Average contents of gradient buffers across MPI processes.
    """
    if num_procs() == 1:
        return
    for p in trainable_parameters(module):
        p_grad_numpy = p.grad.numpy()  # numpy view of tensor data
        avg_p_grad = mpi_avg(p.grad)
        p_grad_numpy[:] = avg_p_grad[:]


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def initialize_weights_orthogonal(layer):
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv1d):
        nn.init.orthogonal_(layer.weight.data)
        nn.init.normal_(layer.bias.data)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers).apply(initialize_weights_orthogonal)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    val = scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[
        ::-1
    ]
    return torch.as_tensor(val.copy())


class PPOBuffer:
    """A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-
    Lambda) for calculating the advantages of state-action pairs."""

    def __init__(
        self,
        state_dim: int,
        lidar_dim: int,
        act_dim: int,
        size: int,
        gamma: float = 0.99,
        lam: float = 0.95,
        nagents: int = 1,
    ):
        self.agent_list = [f"agent_{i}" for i in range(nagents)]

        self.state_buf = {
            a_id: torch.zeros(
                combined_shape(size, state_dim), dtype=torch.float32
            )
            for a_id in self.agent_list
        }
        self.lidar_buf = {
            a_id: torch.zeros(
                combined_shape(size, lidar_dim), dtype=torch.float32
            )
            for a_id in self.agent_list
        }

        self.act_buf = {
            a_id: torch.zeros(
                combined_shape(size, act_dim), dtype=torch.float32
            )
            for a_id in self.agent_list
        }
        self.adv_buf = {
            a_id: torch.zeros(size, dtype=torch.float32)
            for a_id in self.agent_list
        }
        self.rew_buf = {
            a_id: torch.zeros(size, dtype=torch.float32)
            for a_id in self.agent_list
        }
        self.ret_buf = {
            a_id: torch.zeros(size, dtype=torch.float32)
            for a_id in self.agent_list
        }
        self.val_buf = {
            a_id: torch.zeros(size, dtype=torch.float32)
            for a_id in self.agent_list
        }
        self.logp_buf = {
            a_id: torch.zeros(size, dtype=torch.float32)
            for a_id in self.agent_list
        }

        self.gamma, self.lam = gamma, lam
        self.max_size = size
        self.ptr = {a_id: 0 for a_id in self.agent_list}
        self.path_start_idx = {a_id: 0 for a_id in self.agent_list}

    def store(self, a_id: str, obs, lidar, act, rew, val, logp):
        """Append one timestep of agent-environment interaction to the
        buffer."""
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

        for a_id in self.agent_list:
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
        """Call this at the end of an epoch to get all of the data from the
        buffer, with advantages appropriately normalized (shifted to have mean
        zero and std one).

        Also, resets some pointers in the buffer.
        """
        for a_id in self.agent_list:
            assert self.ptr[a_id] == self.max_size

        for a_id in self.agent_list:
            self.ptr[a_id], self.path_start_idx[a_id] = 0, 0

            # the next two lines implement the advantage normalization trick
            adv_mean, adv_std = mpi_statistics_scalar(
                self.adv_buf[a_id].numpy()
            )
            self.adv_buf[a_id] = (self.adv_buf[a_id] - adv_mean) / adv_std
        return {
            a_id: dict(
                obs=self.state_buf[a_id],
                lidar=self.lidar_buf[a_id],
                act=self.act_buf[a_id],
                ret=self.ret_buf[a_id],
                adv=self.adv_buf[a_id],
                logp=self.logp_buf[a_id],
            )
            for a_id in self.agent_list
        }
