import math
from typing import Dict, List, Tuple, Union

import torch
from sdriving.trafficsim.parametric_curves import (
    CatmullRomSplineMotion,
    ClothoidMotion,
    LinearSplineMotion,
)
from sdriving.trafficsim.utils import angle_normalize
from torch import nn

from spinup.utils.mpi_tools import proc_id

EPS = 1e-7


def softclamp(x, lim):
    return (x / lim).tanh() * lim


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
        v_lim: Union[float, List[float]] = 8.0,
        accln_lim: Union[float, List[float]] = 1.5,
    ):
        if isinstance(dim, float):
            dim = [dim]
        if isinstance(v_lim, float):
            v_lim = [v_lim]
        if isinstance(accln_lim, float):
            accln_lim = [accln_lim]

        assert len(dim) == len(v_lim)

        super().__init__()

        self.dt = dt
        self.dim = torch.as_tensor(dim).unsqueeze(1)
        self.v_lim = torch.as_tensor(v_lim).unsqueeze(1)
        self.accln_lim = torch.as_tensor(accln_lim).unsqueeze(1)

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
            a_lim = self.accln_lim
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
            a_lim = (
                self.accln_lim.unsqueeze(0)
                .repeat(state.size(0) // self.nbatch, 1, 1)
                .view(-1, 1)
            )

        v = torch.min(torch.max(v + acceleration * dt, -v_lim), v_lim)

        theta = theta + v * torch.sin(beta) * 2 / dim
        theta = angle_normalize(theta)

        return torch.cat([x, y, v, theta], dim=1)


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


class CatmullRomSplineAccelerationModel(BicycleKinematicsModel):
    def register_track(self, track: torch.Tensor, dummy_point: bool = True, **kwargs):
        assert self.nbatch == 1
        self.motion = CatmullRomSplineMotion(track, **kwargs)
        diff = self.motion.diff
        ratio = diff[:, 1] / (diff[:, 0] + EPS)  # (k - 1)
        self.arc_lengths = self.motion.arc_lengths
        self.theta = angle_normalize(
            torch.where(
                diff[:, 0] > 0,
                torch.atan(ratio),
                math.pi + torch.atan(ratio),
            )
        )
        if dummy_point:
            self.distance = self.motion.arc_lengths[self.motion.p_num]
            self.track_num = torch.ones(1).long() * self.motion.p_num
        else:
            self.distance = torch.zeros(1)
            self.track_num = torch.zeros(1).long()

    def _get_track(self, state):
        # To ensure the end is < self.motion.curve_length
        s = (
            self.distance % self.motion.curve_length
        ) % self.motion.curve_length
        sg = self.track_num
        while s < 0:
            s = s + self.arc_lengths[0, -1]
        while (
            s > self.arc_lengths[(sg + 1) % self.motion.npoints]
            or s < self.arc_lengths[sg]
        ):
            sg = (sg + 1) % self.motion.npoints

        self.track_num = sg

        return (sg.unsqueeze(0), self.theta[sg])

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

        self.distance = self.distance + v * dt

        sg, theta = self._get_track(state)
        ts = self.motion.map_s_to_t(
            self.distance % self.motion.curve_length, sg
        )
        s_t = self.motion(ts)
        theta = theta.unsqueeze(-1)

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


class ParametricBicycleKinematicsModel(BicycleKinematicsModel):
    def __init__(
        self, track, model=ClothoidMotion, *args, model_kwargs={}, **kwargs
    ):
        # track is a B x k x 3 tensor
        super().__init__(*args, **kwargs)

        # TODO: Work on a batched implementation that is differentiable
        assert self.nbatch == 1

        self.motion = model()
        self.model = model
        self.distances = torch.zeros(self.nbatch)
        self.distance_checks = torch.zeros(self.nbatch).bool()
        self.track_num = torch.zeros(self.nbatch).int()
        self.position = torch.zeros(self.nbatch, 2)
        self.theta = torch.zeros(self.nbatch, 1)
        self.registered = torch.zeros(self.nbatch).bool()
        self.arc_lengths = None
        self.model_kwargs = model_kwargs

        #         assert self.nbatch == track.size(0)
        self.track_unmodified = track.clone()
        self.track = self._setup_track(track)

    def _setup_track(self, track):
        if isinstance(self.motion, ClothoidMotion):
            assert track.shape[0] == self.nbatch
            assert track.shape[2] == 3
            track[:, :, 1] = torch.cumsum(track[:, :, 1], dim=1)
            return track
        elif isinstance(self.motion, LinearSplineMotion):
            assert track.shape[0] == self.nbatch
            assert track.shape[2] == 2
            diff = track[:, :-1, :] - track[:, 1:, :]
            ratio = diff[:, :, 1] / diff[:, :, 0]  # B x (k - 1)
            self.arc_lengths = diff.pow(2).sum(-1).sqrt()
            self.theta = angle_normalize(
                torch.where(
                    diff[:, :, 0] < 0,
                    torch.atan(ratio),
                    math.pi + torch.atan(ratio),
                )
            )
            self.theta.unsqueeze_(-1)
            return track
        elif isinstance(self.motion, CatmullRomSplineMotion):
            self.motion = self.model(track, **self.model_kwargs)
            diff = self.motion.diff
            ratio = diff[:, 1] / (diff[:, 0] + EPS)  # (k - 1)
            self.arc_lengths = self.motion.arc_lengths
            self.theta = angle_normalize(
                torch.where(
                    diff[:, 0] > 0,
                    torch.atan(ratio),
                    math.pi + torch.atan(ratio),
                )
            )
            self.theta.unsqueeze_(-1)
            return track

    def reset(self):
        self.position = torch.zeros(self.nbatch, 2)
        self.theta = torch.zeros(self.nbatch, 1)
        self.registered = torch.zeros(self.nbatch).bool()
        self.distances = torch.zeros(self.nbatch)
        self.distance_checks = torch.zeros(self.nbatch).bool()
        self.track_num = torch.zeros(self.nbatch).int()
        self.track = self._setup_track(self.track_unmodified)

    def _get_track(self, state):
        if isinstance(self.motion, ClothoidMotion):
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
                    self.track_num[i] = self.track_num[i] + 1
                    if self.track_num[i] > self.track.size(1) - 1:
                        self.track_num[i] = self.track.size(1) - 1
                        self.distance_checks[i] = True
                    else:
                        self.distances[i] = 0.0
                        self.position[i, :] = state[i, 0:2]
                        self.theta[i, :] = state[i, 2:3]
                elif self.distances[i] < 0.0:
                    self.distance_checks[i] = False
                    self.track_num[i] = max(self.track_num[i] - 1, 0.0)
                    # TODO: Special case for 0, Will have to modify position
                    # accordingly
                    self.distances[i] = (
                        self.track[i, self.track_num[i], 1] + self.distances[i]
                    )
            return torch.cat(a, dim=0), torch.cat(dir, dim=0)
        elif isinstance(self.motion, LinearSplineMotion):
            # For now the batch size is 1
            for i, tnum in enumerate(self.track_num.clone()):
                if self.distances[i] > self.arc_lengths[i, tnum]:
                    self.track_num[i] = min(
                        self.track_num[i] + 1, self.track.size(1) - 2
                    )
                    if self.track_num[i] == tnum + 1:
                        self.distances[i] = 0.0
                elif self.distances[i] < 0.0:
                    self.track_num[i] = max(self.track_num[i] - 1, 0)
                    if self.track_num[i] == tnum - 1:
                        self.distances[i] = (
                            self.arc_lengths[i, tnum - 1] - self.distances[i]
                        )
                return (
                    self.track[i, tnum, :],
                    self.track[i, tnum + 1, :],
                    self.arc_lengths[i, tnum],
                    self.theta[i, tnum],
                )
        elif isinstance(self.motion, CatmullRomSplineMotion):
            # To ensure the end is < self.motion.curve_length
            s = (
                self.distances[0] % self.motion.curve_length
            ) % self.motion.curve_length
            sg = self.track_num[0]
            while s < 0:
                s = s + self.arc_lengths[0, -1]

            while (
                s > self.arc_lengths[(sg + 1) % self.motion.npoints]
                or s < self.arc_lengths[sg]
            ):
                sg = (sg + 1) % self.motion.npoints

            self.track_num[0] = sg

            return (sg.unsqueeze(0), self.theta[sg])

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

        if isinstance(self.motion, ClothoidMotion):
            a, dir = self._get_track(state)
            s_t, theta = self.motion(
                self.position, a, self.distances, self.theta, dir
            )
        elif isinstance(self.motion, LinearSplineMotion):
            pt1, pt2, arc_length, theta = self._get_track(state)
            s_t = self.motion(pt1, pt2, self.distances / arc_length)
            theta = theta.unsqueeze(-1)
        elif isinstance(self.motion, CatmullRomSplineMotion):
            sg, theta = self._get_track(state)
            ts = self.motion.map_s_to_t(
                self.distances[0:1] % self.motion.curve_length, sg
            )
            s_t = self.motion(ts)
            theta = theta.unsqueeze(-1)

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
