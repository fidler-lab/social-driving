import math
from typing import Dict, List, Tuple, Union

import torch
from sdriving.tsim.parametric_curves import (
    CatmullRomSplineMotion,
)
from sdriving.tsim.utils import angle_normalize
from torch import nn

EPS = 1e-7


class BicycleKinematics(nn.Module):
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
        x, y, v, theta = [state[:, i:(i + 1)] for i in range(4)]
        steering, acceleration = [action[:, i:(i + 1)] for i in range(2)]

        beta = torch.atan(torch.tan(steering) / 2)

        tb = theta + beta
        vdt = v * dt

        x = x + vdt * torch.cos(tb)
        y = y + vdt * torch.sin(tb)

        v = torch.min(
            torch.max(v + acceleration * dt, self.v_lim_neg),
            self.v_lim
        )

        theta = theta + v * torch.sin(beta) * 2 / self.dim
        theta = angle_normalize(theta)

        return torch.cat([x, y, v, theta], dim=1)

    def __call__(self, state: torch.Tensor, action: torch.Tensor):
        return self.forward(state, action)


def BicycleKinematicsModel(*args, **kwargs):
    return torch.jit.script(BicycleKinematics(*args, **kwargs))


class FixedTrackAcceleration(nn.Module):
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
        x, y, v, theta = [state[:, i:(i + 1)] for i in range(4)]
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
            torch.max(v + acceleration * dt, self.v_lim_neg),
            self.v_lim
        )

        arc_part = torch.clamp(arc / (math.pi * self.radius / 2), 0, math.pi / 2)
        theta = self.theta1 + self.sign * arc_part

        return torch.cat([x, y, v, theta], dim=1)


def FixedTrackAccelerationModel(*args, **kwargs):
    return torch.jit.script(FixedTrackAcceleration(*args, **kwargs))