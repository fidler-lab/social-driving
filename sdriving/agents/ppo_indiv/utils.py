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

        self.gamma, self.lam = gamma, lam
        self.max_size = size
        self.ptr = 0
        self.path_start_idx = 0

    def store(self, obs, lidar, act, rew, val, logp):
        """Append one timestep of agent-environment interaction to the
        buffer."""
        assert (
            self.ptr < self.max_size
        )  # buffer has to have room so you can store
        self.state_buf[self.ptr] = obs
        self.lidar_buf[self.ptr] = lidar
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr = self.ptr + 1

    def finish_path(self, last_val: Optional[Union[int, torch.Tensor]] = None):
        last_val = 0 if last_val is None else last_val

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice].detach().numpy(), last_val)
        vals = np.append(self.val_buf[path_slice].detach().numpy(), last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(
            deltas, self.gamma * self.lam
        )

        # the next line computes rewards-to-go,
        # to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """Call this at the end of an epoch to get all of the data from the
        buffer, with advantages appropriately normalized (shifted to have mean
        zero and std one).

        Also, resets some pointers in the buffer.
        """
        self.ptr, self.path_start_idx = 0, 0

        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf.numpy())
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return dict(
            obs=self.state_buf,
            lidar=self.lidar_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
            logp=self.logp_buf,
        )
