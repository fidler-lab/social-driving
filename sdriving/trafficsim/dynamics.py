import math
from typing import Dict, List, Tuple, Union

import torch
from torch import nn

from sdriving.trafficsim.clothoid import ClothoidMotion
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


class FixedTrackAccelerationModel(nn.Module):
    """
    A fixed track is provided for the agent during instantiation. The agent
    only controls the acceleration and the path is controlled by the track.
    """

    def __init__(
        self,
        track,
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
        # track stores a map from x-range, y-range to a circle center and
        # radius. If the agent is in that part of the map it needs to
        # adhere to that track, which essentially changes its orientation.
        self.track = track

        self.dt = dt
        self.dim = torch.as_tensor(dim).unsqueeze(1)
        self.v_lim = torch.as_tensor(v_lim).unsqueeze(1)

        self.device = torch.device("cpu")
        self.nbatch = len(dim)

    def _get_track(self, x, y):
        for key in self.track.keys():
            x_range, y_range = key
            if (x >= x_range[0] and x <= x_range[1]) and (
                y >= y_range[0] and y <= y_range[1]
            ):
                return self.track[key]
        return None

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
            action: N x 1 Dimensional Tensor, where N is the batch size, and
                    the second dimension represents
                    {acceleration}
        """
        assert state.size(0) == 1

        dt = self.dt
        x = state[:, 0:1]
        y = state[:, 1:2]
        v = state[:, 2:3]
        theta = state[:, 3:4]
        acceleration = action[:, 0:1]

        # FIXME: This wont work for batched data
        track = self._get_track(x.item(), y.item())
        if track is not None and not track["turn"]:
            theta = torch.as_tensor([[track["theta"]]]).to(self.device)
        x = x + v * torch.cos(theta) * dt
        y = y + v * torch.sin(theta) * dt

        if state.size(0) == self.nbatch:
            v_lim = self.v_lim
        elif state.size(0) > self.nbatch:
            v_lim = (
                self.v_lim.unsqueeze(0)
                .repeat(state.size(0) // self.nbatch, 1, 1)
                .view(-1, 1)
            )

        v = torch.min(torch.max(v + acceleration * dt, -v_lim), v_lim)

        if track is not None and track["turn"]:
            # Project the updated position to the nearest point on the circle
            # For now the curve can only be a circle
            x_c, y_c = track["center"]
            x_ref = x - x_c
            y_ref = y - y_c
            phi = torch.atan2(y_ref, x_ref)
            r = track["radius"]
            x = r * torch.cos(phi) + x_c
            y = r * torch.sin(phi) + y_c
            if track["clockwise"]:
                theta = angle_normalize(phi - math.pi / 2)
            else:
                theta = angle_normalize(math.pi / 2 + phi)

        return torch.cat([x, y, v, theta], dim=1)


class SplineAccelerationModel(nn.Module):
    # NOTE: This is not an accurate model following the kinematics model.
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

        self.points = torch.zeros(1, 2)
        self.orientations = None
        self.cur_position = 0

    def to(self, device):
        if device == self.device:
            return
        self.dim = self.dim.to(device)
        self.v_lim = self.v_lim.to(device)
        self.device = device
        self.points = self.points.to(device)
        self.orientations = self.orientations.to(device)

    def register(self, points: torch.Tensor):
        # points --> N x 2
        self.points = points.to(self.device)
        pdiff = self.points[1:, :] - self.points[:-1, :]
        self.orientations = torch.atan(pdiff[:, 1] / (pdiff[:, 0] + 1e-7))
        self.orientations = torch.cat(
            [self.orientations, self.orientations[-1].unsqueeze(0)]
        )
        self.orientations.unsqueeze_(1)
        self.cur_position = 0

    def get_next_target(self):
        return self.points[
            min(self.cur_position + 1, self.points.size(0) - 1), :
        ]

    def _nearest_point_on_spline(self, pt: torch.Tensor):
        cpos = self.cur_position
        min_pos = max(0, cpos - 5)
        max_pos = min(self.points.size(0) - 1, cpos + 5)
        npos = min_pos + torch.argmin(
            ((pt.unsqueeze(0) - self.points[min_pos:max_pos, :]) ** 2).sum()
        )
        self.cur_position = npos
        return npos

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """
        Args:
            state: N x 4 Dimensional Tensor, where N is the batch size, and
                   the second dimension represents
                   {x coordinate, y coordinate, velocity, orientation}
            action: N x 1 Dimensional Tensor, where N is the batch size, and
                    the second dimension represents
                    {acceleration}
        """
        assert state.size(0) == 1

        dt = self.dt
        x = state[:, 0:1]
        y = state[:, 1:2]
        v = state[:, 2:3]
        theta = state[:, 3:4]
        acceleration = action[:, 0:1]

        x = x + v * torch.cos(theta) * dt
        y = y + v * torch.sin(theta) * dt
        pos = self._nearest_point_on_spline(torch.as_tensor([x, y]))
        x, y = self.points[pos, :]
        x = x[None, None]
        y = y[None, None]
        theta = self.orientations[pos, :].unsqueeze(0)

        v_lim = self.v_lim
        v = torch.min(torch.max(v + acceleration * dt, -v_lim), v_lim)

        return torch.cat([x, y, v, theta], dim=1)


class ClothoidBicycleKinematicsModel(BicycleKinematicsModel):
    def __init__(self, track, *args, **kwargs):
        # track is a B x k x 3 tensor
        super().__init__(*args, **kwargs)
        self.motion = ClothoidMotion()
        self.distances = torch.zeros(self.nbatch)
        self.distance_checks = torch.zeros(self.nbatch).bool()
        self.track_num = torch.zeros(self.nbatch).int()
        self.position = torch.zeros(self.nbatch, 2)
        self.theta = torch.zeros(self.nbatch, 1)
        self.registered = torch.zeros(self.nbatch).bool()

        assert self.nbatch == track.size(0)
        self.track = track

    def reset(self):
        self.position = torch.zeros(self.nbatch, 2)
        self.theta = torch.zeros(self.nbatch, 1)
        self.registered = torch.zeros(self.nbatch).bool()
        self.distances = torch.zeros(self.nbatch)
        self.distance_checks = torch.zeros(self.nbatch).bool()
        self.track_num = torch.zeros(self.nbatch).int()

    def _get_track(self, state):
        a = []
        dir = []
        for i, tnum in enumerate(self.track_num):
            if not self.registered[i]:
                self.registered[i] = True
                self.position[i, :] = state[i, 0:2]
                self.theta[i, :] = state[i, 2:3]
            a.append(self.track[i : i + 1, tnum, 0:1])
            dir.append(self.track[i : i + 1, tnum, 2:3])
            if (
                not self.distance_checks[i]
                and self.track[i, tnum, 1] < self.distances[i]
            ):
                print(tnum, self.position[i, :])
                self.track_num[i] = self.track_num[i] + 1
                if self.track_num[i] > self.track.size(1) - 1:
                    self.track_num[i] = self.track.size(1) - 1
                    self.distance_checks[i] = True
                else:
                    self.distances[i] = 0.0
                    self.position[i, :] = state[i, 0:2]
                    self.theta[i, :] = state[i, 2:3]
                print(self.track_num[i], self.position[i, :])
            elif self.distances[i] < 0.0:
                self.distance_checks[i] = False
                self.track_num[i] = max(self.track_num[i] - 1, 0.0)
                # TODO: Special case for 0, Will have to modify position
                # accordingly
                self.distances[i] = (
                    self.track[i, self.track_num[i], 1] + self.distances[i]
                )
        return torch.cat(a, dim=0), torch.cat(dir, dim=0)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """
        Args:
            state: N x 4 Dimensional Tensor
            action: N x 1 Dimensional Tensor
        """
        dt = self.dt
        x = state[:, 0:1]
        y = state[:, 1:2]
        v = state[:, 2:3]
        theta = state[:, 3:4]
        acceleration = action[:, 0:1]

        self.distances = self.distances + v * dt

        a, dir = self._get_track(state[:, 0:3])

        print(self.distances)
        s_t, theta = self.motion(
            self.position, a, self.distances, self.theta, dir
        )

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

        return torch.cat([s_t, v, theta], dim=1)
