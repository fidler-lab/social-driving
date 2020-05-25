import math
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from sdriving.trafficsim.utils import (
    angle_normalize,
    circle_area_overlap,
    transform_2d_coordinates,
)


class Vehicle:
    def __init__(
        self,
        position: Union[List[float], torch.Tensor] = [0.0] * 2,
        orientation: Union[float, torch.Tensor] = 0.0,
        dimensions: Union[List[float], torch.Tensor] = [4.48, 2.2],
        destination: Optional[Union[List[float], torch.Tensor]] = None,
        dest_orientation: Optional[Union[float, torch.Tensor]] = None,
        initial_speed: Union[float, torch.Tensor] = 0.0,
        name: str = "car",
        min_lidar_range: float = 1.0,
        max_lidar_range: float = 50.0,
        vision_range: float = 30.0,  # For seeing things - traffic signal
    ):
        self.name = name

        self.position = torch.as_tensor(position)
        self.orientation = angle_normalize(torch.as_tensor(orientation))
        self.dimensions = torch.as_tensor(dimensions)

        self.speed = torch.as_tensor(initial_speed)
        self.safety_circle = (
            1.3 * torch.sqrt(((self.dimensions / 2) ** 2).sum()).item()
        )
        self.area = math.pi * self.safety_circle ** 2

        if destination is not None:
            self.destination = torch.as_tensor(destination)
            self.dest_orientation = angle_normalize(
                torch.as_tensor(dest_orientation)
            )
        else:
            self.destination = None
            self.dest_orientation = None

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

        self.max_lidar_range = max_lidar_range
        self.min_lidar_range = min_lidar_range
        self.vision_range = vision_range
        self.device = torch.device("cpu")

        self.render_utils = None

    def to(self, device):
        if device == self.device:
            return
        self.position = self.position.to(device)
        self.orientation = self.orientation.to(device)
        self.speed = self.speed.to(device)
        self.dimensions = self.dimensions.to(device)
        self.safety_circle = self.safety_circle.to(device)
        if self.destination is not None:
            self.destination = self.destination.to(device)
            self.dest_orientation = self.dest_orientation.to(device)
        self.base_coordinates = self.base_coordinates.to(device)
        if self.cached_coordinates:
            self.coordinates = self.coordinates.to(device)
        self.device = device

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
        pt1 = torch.cat(
            [
                coordinates[0:1],
                coordinates[1:2],
                coordinates[2:3],
                coordinates[3:],
            ]
        )
        pt2 = torch.cat(
            [
                coordinates[1:2],
                coordinates[2:3],
                coordinates[3:],
                coordinates[0:1],
            ]
        )
        return pt1, pt2

    def update_state(self, state: torch.Tensor):
        """
        Args:
            state: {x coordinate, y coordinate, speed, orientation}
        """
        self.position = state[:2]
        self.speed = state[2]
        self.orientation = state[3]
        self.cached_coordinates = False

    def distance_from_point(self, point: torch.Tensor):
        vec = self.position - point
        return torch.sqrt((vec.dot(vec)).sum())

    def distance_from_destination(self):
        if self.destination is not None:
            return self.distance_from_point(self.destination)
        raise Exception("Destination is not defined for the Vehicle")

    def optimal_heading_to_point(self, point: torch.Tensor):
        vec = point - self.position
        vec /= torch.norm(vec) + 1e-10
        cur_vec = torch.as_tensor(
            [
                torch.cos(self.orientation).item(),
                torch.sin(self.orientation).item(),
            ]
        )
        opt_angle = angle_normalize(
            torch.acos(vec.dot(cur_vec).clamp_(-1.0, 1.0))
        )
        return opt_angle

    def optimal_heading(self):
        if self.destination is not None:
            return self.optimal_heading_to_point(self.destination)
        raise Exception("Destination is not defined for the Vehicle")

    @staticmethod
    def to_numpy(x: torch.Tensor):
        return x.detach().cpu().numpy()

    def safety_circle_overlap(self, obj):
        if isinstance(obj, Vehicle):
            return circle_area_overlap(
                self.position,
                obj.position,
                self.safety_circle,
                obj.safety_circle,
            )
        raise NotImplementedError

    """
    def render_pyglet(self, viewer, zoom_factor=10.0):
        if self.render_utils is None:
            # The bottom left corner is (0, 0)
            trans_factor = torch.as_tensor(
                [viewer.width / 2, viewer.height / 2]
            )

            coord = self.base_coordinates * zoom_factor + trans_factor
            box = rendering.FilledPolygon(coord.numpy())
            box.set_color(0.0, 1.0, 0.0)
            transform = rendering.Transform()
            box.add_attr(transform)
            viewer.add_geom(box)

            head_coord = torch.as_tensor(
                [[0.0, 0.0], [self.dimensions[0], 0.0]]
            )
            head_coord = head_coord * zoom_factor + trans_factor
            heading = rendering.PolyLine(head_coord, close=True)
            heading.set_color(0.0, 0.0, 1.0)
            heading.add_attr(transform)
            viewer.add_geom(heading)

            # TODO: Add the lidar range circles
            self.render_utils = {
                "transform": transform,
                "zoom_factor": zoom_factor,
                "trans_factor": trans_factor,
            }

        transform = self.render_utils["transform"]
        zoom_factor = self.render_utils["zoom_factor"]
        trans_factor = self.render_utils["trans_factor"]

        translation = (self.position * zoom_factor + trans_factor).numpy()
        print(translation)
        transform.set_translation(*translation)
        transform.set_rotation(self.orientation.numpy())
    """

    def render(
        self,
        ax,
        color: str = "g",
        draw_safety_circle: bool = False,
        draw_lidar_range: bool = True,
    ):
        pos = self.to_numpy(self.position)
        h = self.to_numpy(self.orientation)
        dim = self.dimensions[0].item()
        box = self.to_numpy(self.get_coordinates())

        arrow = np.array(
            [pos, pos + dim / 2.0 * np.array([np.cos(h), np.sin(h)])]
        )

        # Draw the vehicle and the heading
        plt.fill(box[:, 0], box[:, 1], color, edgecolor="k", alpha=0.5)
        plt.plot(arrow[:, 0], arrow[:, 1], "b")

        # Draw the destination if available
        if self.destination is not None:
            dest = self.to_numpy(self.destination)
            plt.plot(dest[0], dest[1], color, marker="x", markersize=5)

        # Draw the safety circle
        if draw_safety_circle:
            ax.add_artist(
                plt.Circle(
                    pos,
                    self.to_numpy(self.safety_circle),
                    color="r",
                    fill=False,
                )
            )

        # Draw the lidar sensor range
        if draw_lidar_range:
            ax.add_artist(
                plt.Circle(
                    pos,
                    self.min_lidar_range,
                    color="b",
                    fill=False,
                    linestyle="--",
                )
            )
            ax.add_artist(
                plt.Circle(
                    pos,
                    self.max_lidar_range,
                    color="b",
                    fill=False,
                    linestyle="--",
                )
            )

    def drawing_info(self):
        info = {
            obj: self.to_numpy(getattr(self, obj))
            for obj in [
                "position",
                "get_coordinates",
                "orientation",
                "min_lidar_range",
                "max_lidar_range",
                "safety_circle",
            ]
        }
        if self.destination is not None:
            info["destination"] = self.to_numpy(self.destination)
        info["dimension"] = self.dimensions[0].item()
        return info


class VehicleEntity:
    def __init__(self, vehicle: Vehicle, road, grayarea: bool):
        self.vehicle = vehicle
        self.road = road
        self.grayarea = grayarea

    def __repr__(self):
        return f"Vehicle: {self.vehicle.name} | Road: {self.road}"
