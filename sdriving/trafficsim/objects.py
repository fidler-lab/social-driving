from typing import List

import torch
import numpy as np
import matplotlib.pyplot as plt
from sdriving.trafficsim.utils import (
    angle_normalize,
    transform_2d_coordinates,
)


class StaticObject:
    def __init__(
        self,
        name: str,
        loc: torch.Tensor,
        dims: torch.Tensor,
        orientation: torch.Tensor,
    ):
        self.name = name
        # For simplicity we represent all objects as a rectangle
        if dims.size() in [(1,), ()]:
            # If single element treat it as a square
            dims = torch.ones(2) * dims
        self.position = loc
        self.orientation = orientation
        self.dimensions = dims

        self.base_coordinates = torch.as_tensor(
            [
                [self.dimensions[0] / 2, self.dimensions[1] / 2],
                [-self.dimensions[0] / 2, self.dimensions[1] / 2],
                [-self.dimensions[0] / 2, -self.dimensions[1] / 2],
                [self.dimensions[0] / 2, -self.dimensions[1] / 2],
            ]
        )

        self.cached_coordinates = False
        self.coordinates = self.get_coordinates()

    def get_coordinates(self):
        if self.cached_coordinates:
            return self.coordinates
        else:
            self.coordinates = transform_2d_coordinates(
                self.base_coordinates, self.orientation, self.position
            )
            self.cached_coordinates = True
            return self.coordinates

    def get_edges(self):
        coordinates = self.get_coordinates()
        pt1 = coordinates
        pt2 = torch.cat([coordinates[1:], coordinates[0:1]])
        return pt1, pt2

    def step(self, tstep: int = 1):
        # Static Objects don't change configuration
        return

    @staticmethod
    def to_numpy(x: torch.Tensor):
        return x.detach().cpu().numpy()

    def render(self, ax, color: str = "r"):
        pos = self.to_numpy(self.position)
        h = self.to_numpy(self.orientation)
        dim = self.dimensions[0].item()
        box = self.to_numpy(self.get_coordinates())

        arrow = np.array(
            [pos, pos + dim / 2.0 * np.array([np.cos(h), np.sin(h)])]
        )

        # Draw the vehicle and the heading
        plt.fill(box[:, 0], box[:, 1], color, edgecolor="k", alpha=0.5)
        plt.plot(arrow[:, 0], arrow[:, 1], "g")


class Pedestrian(StaticObject):
    def __init__(
        self,
        name: str,
        loc: torch.Tensor,
        dims: torch.Tensor = torch.ones(1) * 0.8,
        orientation: torch.Tensor = torch.as_tensor(0.0),
        velocity: torch.Tensor = torch.ones(1),
        dt: float = 0.1,
    ):
        # Orientation of the Pedestrian remains same
        self.velocity = velocity * torch.cat(
            [
                torch.cos(orientation).unsqueeze(0),
                torch.sin(orientation).unsqueeze(0),
            ]
        )
        self.dt = dt
        super(Pedestrian, self).__init__(name, loc, dims, orientation)

    def step(self, tstep):
        t = tstep * self.dt
        dxdy = self.velocity * t
        self.position = self.position + dxdy
        self.cached_coordinates = False
