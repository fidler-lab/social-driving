from typing import List, Union

import torch
from torch import nn

from sdriving.trafficsim.utils import angle_normalize


class BicycleKinematicsModel(nn.Module):
    """
    Kinematic Bicycle Model from `"Kinematic and Dynamic Vehicle Models for
    Autonomous Driving Control Design" by Kong et. al.
    <https://borrelli.me.berkeley.edu/pdfpub/IV_KinematicMPC_jason.pdf>`_.
    """

    def __init__(
        self,
        dt: float = 0.10,
        dim: Union[float, List[float]] = 4.48,
        v_lim: Union[float, List[float]] = 5.0,
    ):
        if isinstance(dim, float):
            dim = [dim]
        if isinstance(v_lim, float):
            v_lim = [v_lim]

        assert len(dim) == len(v_lim)

        super().__init__()

        self.dt = dt
        self.dim = torch.as_tensor(dim).unsqueeze(1)
        self.v_lim = torch.as_tensor(v_lim).unsqueeze(1)

        self.device = torch.device("cpu")
        self.nbatch = len(dim)

    def to(self, device):
        if device == self.device:
            return
        self.dim = self.dim.to(device)
        self.v_lim = self.v_lim.to(device)
        self.device = device

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """
        Args:
            state: N x 4 Dimensional Tensor, where N is the batch size, and
                   the second dimension represents
                   {x coordinate, y coordinate, velocity, orientation}
            action: N x 2 Dimensional Tensor, where N is the batch size, and
                    the second dimension represents
                    {steering angle, acceleration}
        """
        dt = self.dt
        x = state[:, 0:1]
        y = state[:, 1:2]
        v = state[:, 2:3]
        theta = state[:, 3:4]
        steering = action[:, 0:1]
        acceleration = action[:, 1:2]

        beta = torch.atan(torch.tan(steering) / 2)
        self.cache_beta = beta

        x = x + v * torch.cos(theta + beta) * dt
        y = y + v * torch.sin(theta + beta) * dt

        if state.size(0) == self.nbatch:
            v_lim = self.v_lim
            dim = self.dim
        elif state.size(0) > self.nbatch:
            v_lim = (
                self.v_lim.unsqueeze(0)
                .repeat(state.size(0) // self.nbatch, 1, 1)
                .view(-1, 1)
            )
            dim = (
                self.dim.unsqueeze(0)
                .repeat(state.size(0) // self.nbatch, 1, 1)
                .view(-1, 1)
            )

        v = torch.min(torch.max(v + acceleration * dt, -v_lim), v_lim)

        theta = theta + v * torch.sin(beta) * 2 / dim
        theta = angle_normalize(theta)

        return torch.cat([x, y, v, theta], dim=1)

    def grad_input(self, state: torch.Tensor, action: torch.Tensor):
        nbatch = state.size(0)
        dt = self.dt
        v = state[:, 2:3]
        theta = state[:, 3:4]
        steering = action[:, 0:1]
        acceleration = action[:, 1:2]
        v2 = v + acceleration * dt  # N x 1

        if state.size(0) == self.nbatch:
            v_lim = self.v_lim  # N x 1
            dim = self.dims  # N x 1
        elif state.size(0) > self.nbatch:
            v_lim = (
                self.v_lim.unsqueeze(0)
                .repeat(state.size(0) // self.nbatch, 1, 1)
                .view(-1, 1)
            )  # N x 1
            dim = (
                self.dim.unsqueeze(0)
                .repeat(state.size(0) // self.nbatch, 1, 1)
                .view(-1, 1)
            )  # N x 1

        beta = torch.atan(torch.tan(steering) / 2)  # N x 1
        c = (2 * torch.sin(beta) / dim)[:, :, None]  # N x 1
        stb = torch.sin(theta + beta)  # N x 1
        ctb = torch.cos(theta + beta)  # N x 1

        dbeta_dsteer = (
            1 / (1 + 3 * (torch.cos(steering) ** 2))[:, :, None]
        )  # N x 1 x 1

        do_dx = torch.as_tensor([[[1.0, 0.0, 0.0, 0.0]]]).repeat(
            nbatch, 1, 1
        )  # N x 1 x 4
        do_dy = torch.as_tensor([[[0.0, 1.0, 0.0, 0.0]]]).repeat(
            nbatch, 1, 1
        )  # N x 1 x 4

        dx_dv = (ctb * dt)[:, :, None]  # N x 1 x 1
        dy_dv = (stb * dt)[:, :, None]  # N x 1 x 1
        dv_dv = torch.where(
            (v2 > v_lim) + (v2 < -v_lim),
            torch.zeros(nbatch, 1),
            torch.ones(nbatch, 1),
        )[
            :, :, None
        ]  # N x 1 x 1
        dt_dv = c * dv_dv  # N x 1 x 1
        do_dv = torch.cat([dx_dv, dy_dv, dv_dv, dt_dv], dim=-1)  # N x 1 x 4

        dx_dt = (-stb * v * dt)[:, :, None]  # N x 1 x 1
        dy_dt = (ctb * v * dt)[:, :, None]  # N x 1 x 1
        do_dt = torch.cat(
            [
                dx_dt,
                dy_dt,
                torch.zeros(nbatch, 1, 1),
                torch.ones(nbatch, 1, 1),
            ],
            dim=-1,
        )  # N x 1 x 4

        do_ds = torch.cat([do_dx, do_dy, do_dv, do_dt], dim=1).permute(
            0, 2, 1
        )  # N x 4 x 4

        dv_da = dv_dv * dt  # N x 1 x 1
        dt_da = c * dv_da  # N x 1 x 1
        do_da = torch.cat(
            [torch.zeros(nbatch, 1, 2), dv_da, dt_da], dim=-1
        )  # N x 1 x 4

        dx_dst = dx_dt
        dy_dst = dy_dt
        dt_dst = (
            2 * torch.min(torch.max(v2, -v_lim), v_lim) * torch.cos(beta)
        ) / dim  # N x 1
        dt_dst = dt_dst[:, :, None]
        do_dst = (
            torch.cat(
                [dx_dst, dy_dst, torch.zeros(nbatch, 1, 1), dt_dst], dim=-1
            )
            * dbeta_dsteer
        )  # N x 1 x 4

        do_du = torch.cat([do_dst, do_da], dim=1).permute(0, 2, 1)  # N x 4 x 2

        return do_ds, do_du
