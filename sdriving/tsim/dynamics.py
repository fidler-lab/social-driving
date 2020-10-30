import math

import torch
from torch import nn

from sdriving.tsim.parametric_curves import CatmullRomSpline
from sdriving.tsim.utils import angle_normalize, remove_batch_element

EPS = 1e-7


class _BicycleKinematicsModel(nn.Module):
    """
    Kinematic Bicycle Model from `"Kinematic and Dynamic Vehicle Models for
    Autonomous Driving Control Design" by Kong et. al.
    <https://borrelli.me.berkeley.edu/pdfpub/IV_KinematicMPC_jason.pdf>`_.
    """

    def __init__(
        self,
        dt: float = 0.10,
        dim: torch.Tensor = torch.ones(1) * 4.48,
        v_lim: torch.Tensor = torch.ones(1) * 8.0,
    ):
        super().__init__()
        self.dt = dt
        self.dim = dim.unsqueeze(1)
        self.v_lim = v_lim.unsqueeze(1)
        self.v_lim_neg = -self.v_lim

        self.device = torch.device("cpu")
        self.to(self.dim.device)
        self.nbatch = dim.size(0)

    def to(self, device):
        if device == self.device:
            return
        for k, t in filter(
            lambda x: torch.is_tensor(x[1]), self.__dict__.items()
        ):
            setattr(self, k, t.to(device))
        self.device = device

    @torch.jit.export
    def remove(self, idx: int):
        self.nbatch -= 1
        self.v_lim = remove_batch_element(self.v_lim, idx)
        self.v_lim_neg = remove_batch_element(self.v_lim_neg, idx)
        self.dim = remove_batch_element(self.dim, idx)

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
        x, y, v, theta = [state[:, i : (i + 1)] for i in range(4)]
        steering, acceleration = [action[:, i : (i + 1)] for i in range(2)]

        beta = torch.atan(torch.tan(steering) / 2)

        tb = theta + beta
        vdt = v * dt

        x = x + vdt * torch.cos(tb)
        y = y + vdt * torch.sin(tb)

        v = torch.min(
            torch.max(v + acceleration * dt, self.v_lim_neg), self.v_lim
        )

        theta = theta + v * torch.sin(beta) * 2 / self.dim
        theta = angle_normalize(theta)

        return torch.cat([x, y, v, theta], dim=1)


def BicycleKinematicsModel(*args, **kwargs):
    return torch.jit.script(_BicycleKinematicsModel(*args, **kwargs))


class _FixedTrackAccelerationModel(nn.Module):
    """
    A fixed track is provided for the agent during instantiation. The agent
    only controls the acceleration and the path is controlled by the track.
    """

    def __init__(
        self,
        theta1: torch.Tensor,
        theta2: torch.Tensor,
        radius: torch.Tensor,
        center: torch.Tensor,
        distance1: torch.Tensor,
        dt: float = 0.10,
        v_lim: torch.Tensor = torch.ones(1) * 8.0,
    ):
        super().__init__()
        self.theta1 = theta1.unsqueeze(1)
        self.sign = torch.sign(
            angle_normalize(theta2.unsqueeze(1) - self.theta1)
        )
        # Pass radius as 0 (actually inf) to produce a straight road
        self.radius = radius.unsqueeze(1)
        self.distance1 = distance1.unsqueeze(1)
        self.distances = torch.zeros_like(self.distance1)
        self.center = center
        self.circ_arc = self.radius * math.pi / 2

        self.turns = self.radius != 0

        self.dt = dt
        self.v_lim = v_lim.unsqueeze(1)
        self.v_lim_neg = -self.v_lim

        self.device = torch.device("cpu")
        self.to(self.v_lim.device)
        self.nbatch = self.v_lim.size(0)

    def reset(self):
        # Only useful if the environment remains static. That is
        # unlikely to happen
        self.distances.fill_(0.0)

    def to(self, device):
        if device == self.device:
            return
        for k, t in filter(
            lambda x: torch.is_tensor(x[1]), self.__dict__.items()
        ):
            setattr(self, k, t.to(device))
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

        dt = self.dt
        x, y, v, theta = [state[:, i : (i + 1)] for i in range(4)]
        acceleration = action
        vdt = v * dt

        self.distances = self.distances + vdt
        arc = self.distances - self.distance1

        stheta, ctheta = torch.sin(theta), torch.cos(theta)

        in_arc = (arc > 0) * (arc < self.circ_arc) * self.turns
        nin_arc = ~in_arc
        x = nin_arc * (x + vdt * ctheta) + in_arc * (
            self.center[:, 0:1] + self.sign * self.radius * stheta
        )
        y = nin_arc * (y + vdt * stheta) + in_arc * (
            self.center[:, 1:2] - self.sign * self.radius * ctheta
        )
        v = torch.min(
            torch.max(v + acceleration * dt, self.v_lim_neg), self.v_lim
        )

        arc_part = torch.clamp(
            arc / (0.5 * math.pi * self.radius + 1e-7), 0, math.pi / 2
        )
        theta = self.theta1 + self.sign * arc_part

        return torch.cat([x, y, v, theta], dim=1)


def FixedTrackAccelerationModel(*args, **kwargs):
    return torch.jit.script(_FixedTrackAccelerationModel(*args, **kwargs))


class _SplineModel(nn.Module):
    """
    IMPORTANT NOTE: Much like the FixedTrackAccelerationModel
    this Model is stateful. Even though the state passed to it
    must be of the shape N x 4, the x, y, and theta will be
    simply ignored. This allows for a consistent API and
    makes environment design simpler
    """

    def __init__(
        self,
        cps: torch.Tensor,
        p_num: int = 5,
        alpha: int = 0.5,
        dt: float = 0.10,
        v_lim: torch.Tensor = torch.ones(1) * 8.0,
    ):
        super().__init__()
        self.dt = dt
        self.v_lim = v_lim.unsqueeze(1)
        self.v_lim_neg = -self.v_lim

        self.device = cps.device
        self.to(self.v_lim.device)
        self.nbatch = v_lim.size(0)

        self.motion = CatmullRomSpline(cps, p_num, alpha)
        self.distances = torch.zeros(self.nbatch, 1, device=self.device)
        diff = self.motion.diff
        ratio = diff[:, :, 1] / (diff[:, :, 0] + EPS)
        self.arc_lengths = self.motion.arc_lengths
        self.curve_lengths = self.motion.curve_length.unsqueeze(1) - 1e-3
        # Assume that last 2 points are not part of the spline.
        self.distance_proxy = (
            (cps[:, :-3, :] - cps[:, 1:-2, :])
            .pow(2)
            .sum(-1)
            .sqrt()
            .sum(-1, keepdim=True)
        )
        self.theta = angle_normalize(
            torch.where(
                diff[:, :, 0] > 0,
                torch.atan(ratio),
                math.pi + torch.atan(ratio),
            )
        ).unsqueeze(-1)

    @torch.jit.export
    def reset(self):
        self.distances = torch.zeros(self.nbatch, 1, device=self.device)

    @torch.jit.export
    def remove(self, idx: int):
        self.motion.remove(idx)
        self.nbatch -= 1
        self.v_lim = remove_batch_element(self.v_lim, idx)
        self.v_lim_neg = remove_batch_element(self.v_lim_neg, idx)
        self.distances = remove_batch_element(self.distances, idx)
        self.distance_proxy = remove_batch_element(self.distance_proxy, idx)
        self.theta = remove_batch_element(self.theta, idx)
        self.curve_lengths = remove_batch_element(self.curve_lengths, idx)
        self.arc_lengths = remove_batch_element(self.arc_lengths, idx)

    def to(self, device):
        if device == self.device:
            return
        for k, t in filter(
            lambda x: torch.is_tensor(x[1]), self.__dict__.items()
        ):
            setattr(self, k, t.to(device))
        self.device = device

    def forward(self, state: torch.Tensor, action: torch.Tensor):  # N x 1
        dt = self.dt
        v = state[:, 2:3]
        acceleration = action[:, 0:1]

        self.distances = self.distances + v * dt
        distances = self.distances % self.curve_lengths  # N x 1

        c1 = self.arc_lengths[:, :-1]  # N x (P - 1)
        c2 = self.arc_lengths[:, 1:]  # N x (P - 1)
        sgs = torch.where(
            (c1 <= distances) * (distances < c2)
        )  # (N x 1, N x 1)

        ts = self.motion(distances, sgs)  # N x 1
        pts = self.motion.sample_points(ts).reshape(-1, 2)  # N x 2

        theta = self.theta[sgs[0], sgs[1]].reshape(pts.size(0), 1)
        v = torch.min(
            torch.max(v + acceleration * dt, self.v_lim_neg), self.v_lim
        )
        return torch.cat([pts, v, theta], dim=1)


def SplineModel(*args, **kwargs):
    return torch.jit.script(_SplineModel(*args, **kwargs))
