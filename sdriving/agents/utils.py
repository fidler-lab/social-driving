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
