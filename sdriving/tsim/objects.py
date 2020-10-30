from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from sdriving.tsim.utils import (
    angle_normalize,
    get_2d_rotation_matrix,
    transform_2d_coordinates,
    transform_2d_coordinates_rotation_matrix,
)


class _Pedestrians(torch.nn.Module):
    def __init__(
        self,
        position: torch.Tensor,  # N x 2
        dims: torch.Tensor,  # N x 2
        orientation: torch.Tensor,  # N x 1
        velocity: torch.Tensor,  # N x 1
        dt: float = 0.1,
        name: str = "pedestrian",
    ):
        super().__init__()

        self.name = name

        self.position = position
        self.orientation = angle_normalize(orientation)
        self.dimensions = dims
        self.dt = dt

        self.nbatch = self.position.size(0)

        self.speed = velocity
        self.velocity = velocity * torch.cat(
            [
                torch.cos(orientation),
                torch.sin(orientation),
            ],
            dim=-1,
        )

        mul_factor = (
            torch.as_tensor([[1, 1], [1, -1], [-1, -1], [-1, 1]])
            .unsqueeze(0)
            .type_as(self.dimensions)
        )

        self.base_coordinates = mul_factor * self.dimensions.unsqueeze(1) / 2
        self.mul_factor = mul_factor
        self.device = torch.device("cpu")

        self.to(self.position.device)

        self.cached_coordinates = False
        self.coordinates = self._get_coordinates()

    def to(self, device: torch.device):
        if device == self.device:
            return
        for k, t in self.__dict__.items():
            if torch.is_tensor(t):
                setattr(self, k, t.to(device))
        self.device = device

    @torch.jit.export
    def _get_coordinates(self):
        self.cached_coordinates = True
        rot_mat = get_2d_rotation_matrix(self.orientation[:, 0])
        if rot_mat.ndim == 2:
            rot_mat.unsqueeze_(0)
        self.coordinates = transform_2d_coordinates_rotation_matrix(
            self.base_coordinates,
            rot_mat,
            self.position[:, None, :],
        )
        return self.coordinates

    @torch.jit.export
    def get_coordinates(self):
        return (
            self.coordinates
            if self.cached_coordinates
            else self._get_coordinates()
        )

    @torch.jit.export
    def get_edges(self):
        coordinates = self.get_coordinates()
        pt1 = coordinates
        pt2 = torch.cat([coordinates[:, 1:, :], coordinates[:, 0:1, :]], dim=1)
        return pt1.reshape(-1, 2), pt2.reshape(-1, 2)

    @torch.jit.export
    def step(self, tstep: int):
        t = tstep * self.dt
        dxdy = self.velocity * t
        self.position = self.position + dxdy
        self.cached_coordinates = False


def Pedestrians(*args, **kwargs):
    return torch.jit.script(_Pedestrians(*args, **kwargs))


def render_object(
    obj,
    ax,
    color: Union[str, List[Union[tuple, str]]] = "g",
):
    if isinstance(color, str):
        color = [color] * obj.nbatch
    for b in range(obj.nbatch):
        pos = obj.position[b, :].detach().cpu().numpy()
        h = obj.orientation[b, :].detach().cpu().numpy()
        dim = obj.dimensions[b, 0].item()
        box = obj.get_coordinates()[b, :, :].detach().cpu().numpy()
        # Draw the vehicle and the heading
        ax.fill(
            box[:, 0],
            box[:, 1],
            facecolor=color[b],
            edgecolor="black",
            alpha=0.5,
        )
        ax.plot(
            [pos[0], pos[0] + 0.5 * dim * np.cos(h)],
            [pos[1], pos[1] + 0.5 * dim * np.sin(h)],
            "c",
        )
