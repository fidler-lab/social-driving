import random

import horovod.torch as hvd
import numpy as np
import scipy.signal
import torch
import torch.nn as nn


def find_free_port():
    import socket

    s = socket.socket()
    s.bind(("", 0))  # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def trainable_parameters(net):
    return list(filter(lambda x: x.requires_grad, net.parameters()))


def mpi_avg_grads(module):
    """Average contents of gradient buffers across MPI processes."""
    if num_procs() == 1:
        return
    for p in trainable_parameters(module):
        p_grad_numpy = p.grad.numpy()  # numpy view of tensor data
        avg_p_grad = mpi_avg(p.grad)
        p_grad_numpy[:] = avg_p_grad[:]


def combined_shape(length, shape=None, batch=None):
    if batch is None:
        if shape is None:
            return (length,)
        return (length, shape) if np.isscalar(shape) else (length, *shape)
    else:
        if shape is None:
            return (
                batch,
                length,
            )
        return (
            (batch, length, shape)
            if np.isscalar(shape)
            else (batch, length, *shape)
        )


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


def discount_cumsum(x: torch.Tensor, discount: float):
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
    val = scipy.signal.lfilter(
        [1], [1, float(-discount)], x.detach().cpu().numpy()[::-1], axis=0
    )[::-1]
    return torch.from_numpy(val.copy()).to(x.device)


def hvd_scalar_statistics(x: torch.Tensor):
    # Horovod doesn't compile on GPU on my machine
    dev = x.device
    N = x.nelement() * hvd.size()
    mean = hvd.allreduce(x.mean().cpu(), op=hvd.Average)
    std = torch.sqrt(
        hvd.allreduce((x - mean).pow(2).sum().cpu(), op=hvd.Sum) / N
    )
    return mean.to(dev), std.to(dev)


def hvd_scalar_statistics_with_min_max(x: torch.Tensor):
    dev = x.device
    minimum = torch.min(
        hvd.allgather(torch.min(x, dim=0, keepdim=True)[0].cpu())
    )
    maximum = torch.max(
        hvd.allgather(torch.max(x, dim=0, keepdim=True)[0].cpu())
    )
    return [*hvd_scalar_statistics(x), minimum.to(dev), maximum.to(dev)]


def hvd_average_grad(x: torch.nn.Module, dev: torch.device):
    for p in x.parameters():
        if p.grad is not None:
            # For Gradient Averaging we need to scale the learning rate
            p.grad = hvd.allreduce(p.grad.cpu(), op=hvd.Sum).to(dev)
