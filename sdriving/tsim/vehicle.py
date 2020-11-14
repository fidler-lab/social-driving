import math
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from sdriving.tsim.utils import (
    angle_normalize,
    check_intersection_lines,
    circle_area_overlap,
    get_2d_rotation_matrix,
    remove_batch_element,
    transform_2d_coordinates_rotation_matrix,
)


@torch.jit.script
def generate_bool_buffer(n: int, device: torch.device):
    bool_buffer = torch.ones(n * 4, n * 4, dtype=torch.bool, device=device)
    for i in range(0, n * 4, 4):
        bool_buffer[i : (i + 4), i : (i + 4)] = 0
    return bool_buffer


class _BatchedVehicle(torch.nn.Module):
    """
    A fleet of vehicles. A single vehicle is a batched vehicle with
    1 as the batch size
    """

    def __init__(
        self,
        position: torch.Tensor,  # N x 2
        orientation: torch.Tensor,  # N x 1
        destination: torch.Tensor,  # N x 2
        dest_orientation: torch.Tensor,  # N x 1
        dimensions: torch.Tensor = torch.as_tensor([[4.48, 2.2]]),  # N x 2
        initial_speed: torch.Tensor = torch.zeros(1, 1),  # N x 1
        name: str = "car",
        min_lidar_range: float = 5.0,
        max_lidar_range: float = 50.0,
        vision_range: float = 50.0,
    ):
        super().__init__()
        self.name = name

        self.position = position
        self.orientation = angle_normalize(orientation)
        self.destination = destination
        self.dest_orientation = angle_normalize(dest_orientation)
        self.dimensions = dimensions

        self.nbatch = self.position.size(0)
        self.bool_buffer = torch.zeros(1).bool()

        self.speed = initial_speed
        self.safety_circle = (
            1.3
            * torch.sqrt(
                ((self.dimensions / 2) ** 2).sum(1, keepdim=True)
            ).detach()
        )
        self.area = math.pi * self.safety_circle ** 2

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

        self.max_lidar_range = max_lidar_range
        self.min_lidar_range = min_lidar_range
        self.vision_range = vision_range

    @torch.jit.export
    def remove(self, idx: int):
        self.nbatch -= 1

        self.speed = remove_batch_element(self.speed, idx)
        self.safety_circle = remove_batch_element(self.safety_circle, idx)
        self.area = remove_batch_element(self.area, idx)
        self.base_coordinates = remove_batch_element(
            self.base_coordinates, idx
        )
        self.position = remove_batch_element(self.position, idx)
        self.destination = remove_batch_element(self.destination, idx)
        self.orientation = remove_batch_element(self.orientation, idx)
        self.dest_orientation = remove_batch_element(
            self.dest_orientation, idx
        )
        self.dimensions = remove_batch_element(self.dimensions, idx)

        self.coordinates = self._get_coordinates()
        self.bool_buffer = generate_bool_buffer(self.nbatch, self.device)

    def to(self, device: torch.device):
        if device == self.device:
            return
        for k, t in self.__dict__.items():
            if torch.is_tensor(t):
                setattr(self, k, t.to(device))
        self.device = device

    @torch.jit.export
    def add_bool_buffer(self, bool_buffer: torch.Tensor):
        self.bool_buffer = bool_buffer

    @torch.jit.export
    def add_vehicle(
        self,
        position: torch.Tensor,  # 1 x 2
        orientation: torch.Tensor,  # 1 x 1
        destination: torch.Tensor,  # 1 x 2
        dest_orientation: torch.Tensor,  # 1 x 1
        dimensions: torch.Tensor = torch.as_tensor([[4.48, 2.2]]),  # 1 x 2
        initial_speed: torch.Tensor = torch.zeros(1, 1),  # 1 x 1
    ) -> bool:
        position = position.to(self.device)
        orientation = angle_normalize(orientation.to(self.device))
        dimensions = dimensions.to(self.device)
        base_coordinates = self.mul_factor * dimensions.unsqueeze(1) / 2
        rot_mat = get_2d_rotation_matrix(orientation[:, 0])
        coordinates = torch.matmul(base_coordinates[0], rot_mat) + position
        check = self.collision_check_with_rectangle(
            coordinates, torch.cat([coordinates[1:, :], coordinates[0:1, :]])
        )

        if check.any():
            return False

        self.position = torch.cat([self.position, position])
        self.orientation = torch.cat([self.orientation, orientation])
        self.destination = torch.cat(
            [self.destination, destination.to(self.device)]
        )
        self.dest_orientation = torch.cat(
            [
                self.dest_orientation,
                angle_normalize(dest_orientation.to(self.device)),
            ]
        )
        self.dimensions = torch.cat([self.dimensions, dimensions])
        self.speed = torch.cat([self.speed, initial_speed.to(self.device)])
        self.base_coordinates = torch.cat(
            [self.base_coordinates, base_coordinates]
        )
        self.coordinates = torch.cat(
            [self.coordinates, coordinates[None, :, :]]
        )
        self.nbatch += 1

        return True

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
        return pt1, pt2  # N x 4 x 2, N x 4 x 2

    @torch.jit.export
    def update_state(self, state: torch.Tensor):
        """
        Args:
            state: {x coordinate, y coordinate, speed, orientation}
        """
        self.position = state[:, :2]
        self.speed = state[:, 2:3]
        self.orientation = state[:, 3:4]
        self.cached_coordinates = False

    @torch.jit.export
    def get_state(self):
        return torch.cat([self.position, self.speed, self.orientation], dim=-1)

    @torch.jit.export
    def distance_from_point(self, point: torch.Tensor):
        return (self.position - point).pow(2).sum(1, keepdim=True).sqrt()

    @torch.jit.export
    def distance_from_points(self, points: torch.Tensor):
        return (
            (self.position.unsqueeze(1) - points)
            .pow(2)
            .sum(2, keepdim=True)
            .sqrt()
        )

    @torch.jit.export
    def distance_from_destination(self):
        return self.distance_from_point(self.destination)

    @torch.jit.export
    def optimal_heading_to_point(self, point: torch.Tensor):
        vec = point - self.position
        vec = vec / (torch.norm(vec, dim=1, keepdim=True) + 1e-7)  # N x 2
        phi = torch.atan2(vec[:, 1:], vec[:, :1])
        theta = torch.where(
            self.orientation >= 0,
            self.orientation,
            self.orientation + 2 * math.pi,
        )
        return angle_normalize(phi - theta)

    @torch.jit.export
    def optimal_heading_to_points(self, points: torch.Tensor):  # N x B x 2
        vec = points - self.position.unsqueeze(1)
        vec = vec / (torch.norm(vec, dim=2, keepdim=True) + 1e-7)  # N X B x 2
        phi = torch.atan2(vec[..., 1:], vec[..., :1])  # N x B x 1
        theta = torch.where(
            self.orientation >= 0,
            self.orientation,
            self.orientation + 2 * math.pi,
        ).unsqueeze(
            1
        )  # N x 1 x 1
        diff = phi - theta
        return angle_normalize(diff.view(-1, 1)).view(points.shape[:2])

    @torch.jit.export
    def optimal_heading(self):
        return self.optimal_heading_to_point(self.destination)

    @torch.jit.export
    def collision_check(self):
        p1, p2 = self.get_edges()
        p1, p2 = p1.view(-1, 2), p2.view(-1, 2)

        c = check_intersection_lines(p1, p2, p1, p2)
        c = c * self.bool_buffer
        return c.view(self.nbatch, 4, -1).any(1).any(1)

    @torch.jit.export
    def collision_check_with_rectangle(
        self,
        point1: torch.Tensor,
        point2: torch.Tensor,  # 4 x 2  # 4 x 2
    ):
        p1, p2 = self.get_edges()
        p1, p2 = p1.view(-1, 2), p2.view(-1, 2)

        c = check_intersection_lines(p1, p2, point1, point2)
        return c.any()


def BatchedVehicle(*args, **kwargs):
    return torch.jit.script(_BatchedVehicle(*args, **kwargs))


class _Vehicle(_BatchedVehicle):
    def __init__(
        self,
        position: torch.Tensor,  # 2
        orientation: torch.Tensor,  # 1
        destination: torch.Tensor,  # 2
        dest_orientation: torch.Tensor,  # 1
        dimensions: torch.Tensor = torch.as_tensor([4.48, 2.2]),  # 2
        initial_speed: torch.Tensor = torch.zeros(1),  # 1
        name: str = "car",
        min_lidar_range: float = 1.0,
        max_lidar_range: float = 50.0,
        vision_range: float = 50.0,
    ):
        super().__init__(
            position.unsqueeze(0),
            orientation.unsqueeze(0),
            destination.unsqueeze(0),
            dest_orientation.unsqueeze(0),
            dimensions.unsqueeze(0),
            initial_speed.unsqueeze(0),
            name,
            min_lidar_range,
            max_lidar_range,
            vision_range,
        )


def Vehicle(*args, **kwargs):
    return torch.jit.script(_Vehicle(*args, **kwargs))


def render_vehicle(
    obj: Union[_BatchedVehicle, _Vehicle],
    ax,
    color: Union[str, List[Union[tuple, str]]] = "g",
    draw_lidar_range: bool = False,
):
    if isinstance(color, str):
        color = [color] * obj.nbatch
    for b in range(obj.nbatch):
        pos = obj.position[b, :].detach().cpu().numpy()
        h = obj.orientation[b, :].detach().cpu().numpy()
        dim = obj.dimensions[b, 0].item()
        box = obj.get_coordinates()[b, :, :].detach().cpu().numpy()
        lr = obj.max_lidar_range

        # Draw the vehicle and the heading
        ax.fill(
            box[:, 0],
            box[:, 1],
            facecolor=color[b],
            edgecolor="black",
            alpha=1.0,
        )
        ax.plot(
            [pos[0], pos[0] + 0.5 * dim * np.cos(h)],
            [pos[1], pos[1] + 0.5 * dim * np.sin(h)],
            "k",
        )

        # Draw the destination if available
        dest = obj.destination[b, :].detach().cpu().numpy()
        ax.plot([dest[0]], [dest[1]], color=color[b], marker="x", markersize=5)

        # Draw the lidar sensor range
        if draw_lidar_range:
            ax.add_artist(
                plt.Circle(pos, lr, color=color[b], fill=False, linestyle="--", lw=0.5)
            )


@torch.jit.export
def safety_circle_overlap(obj1: _BatchedVehicle, obj2: _BatchedVehicle):
    center1 = obj1.position.repeat(obj2.nbatch, 1)
    center2 = obj2.position.unsqueeze(0).repeat(obj1.nbatch, 1, 1).view(-1, 2)
    radius1 = obj1.safety_circle.repeat(obj2.nbatch, 1)
    radius2 = (
        obj2.safety_circle.unsqueeze(0).repeat(obj1.nbatch, 1, 1).view(-1, 1)
    )
    return circle_area_overlap(center1, center2, radius1, radius2).view(
        obj1.nbatch, obj2.nbatch
    )


@torch.jit.export
def intervehicle_collision_check(obj1: _BatchedVehicle, obj2: _BatchedVehicle):
    p1, p2 = obj1.get_edges()
    p1, p2 = p1.view(-1, 2), p2.view(-1, 2)
    p3, p4 = obj2.get_edges()
    p3, p4 = p3.view(-1, 2), p4.view(-1, 2)

    c = check_intersection_lines(p1, p2, p3, p4)
    return c.view(obj1.nbatch, 4, -1).any(1).any(1)
