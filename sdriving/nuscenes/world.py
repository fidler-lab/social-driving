import random
import logging as lg
import math
import os
from itertools import product
from collections import OrderedDict, deque
from copy import copy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from sdriving.tsim import (
    TrafficSignal,
    World,
    check_intersection_lines,
    generate_lidar_data,
    angle_normalize,
)
from sdriving.nuscenes.utils import lies_in_drivable_area


class NuscenesWorld(World):
    def __init__(
        self,
        map_path: str,
        disable_collision_check: bool = False,
        figsize: Tuple[int] = (10, 10),
        no_signal_val: float = 0.75,
    ):
        self.vehicles = OrderedDict()
        self.traffic_signals_in_path = OrderedDict()
        self.traffic_signals = OrderedDict()
        self.objects = OrderedDict()
        self.current_positions = OrderedDict()

        # This can be set to True for speedup especially if you are using the predefined
        # splines and know that such collisions are not possible
        self.disable_collision_check = disable_collision_check

        self.no_signal_val = no_signal_val

        self.figsize = figsize
        self.fig = None
        self.ax = None
        self.cam = None

        self.device = torch.device("cpu")

        self.map_path = map_path
        self.parse_map_data()

    def remove(self, aname: str, idx: int):
        del self.traffic_signals_in_path[aname]

    def parse_map_data(self):
        data = torch.load(self.map_path)
        self.map_data = data

        self.pt1, self.pt2 = data["edges"]

        for k, v in data.items():
            if k in ["edges", "splines"]:
                continue
            setattr(self, k, v)

        # splines -> Which Pocket -> Which Path -> Which Point ->
        # (spos, epos, orient, eorient, cps)
        self.splines = data["splines"]
        sampling_indices = []
        for k, v in self.splines.items():
            sampling_indices.extend(
                list(product([k], list(v.keys()), range(5)))
            )
        self.sampling_indices = sampling_indices

        self.sampling_indices_list = copy(self.sampling_indices)

        val = [
            torch.as_tensor([0.0, 0.5, 1.0, 0.5]),
            torch.as_tensor([1.0, 0.5, 0.0, 0.5]),
        ]
        colors = [["r", "y", "g", "y"], ["g", "y", "r", "y"]]
        times = torch.as_tensor([100, 20, 100, 20])

        for i in range(data["signal_locations"].size(0)):
            col_map = data["color_mapping"][i]
            self.traffic_signals[i] = (
                TrafficSignal(
                    val[col_map], 0, times, f"signal_{i}", colors[col_map]
                ),
                data["signal_locations"][i],
            )

    def reset(self):
        self.sampling_indices_list = copy(self.sampling_indices)
        super().reset()

    def sample_new_vehicle_position(self):
        # Returns a tuple containing (spos, epos, orient, eorient, cps)
        idx = self.sampling_indices_list.pop(
            random.randrange(len(self.sampling_indices_list))
        )
        return idx, self.splines[idx[0]][idx[1]][idx[2]]

    def _render_background(self, ax):
        ax.scatter(
            self.plotting_utils[1],
            self.plotting_utils[2],
            c=self.plotting_utils[3],
            alpha=0.1,
        )
        self._render_traffic_signal(ax)

    def shortest_path_trajectory(self, *args, **kwargs):
        # This function is not needed and hence shouldn't be called
        raise NotImplementedError

    def get_road_edges(self):
        return self.pt1, self.pt2

    def check_collision(self, vname: str):
        vehicle = self.vehicles[vname]
        # Since we are using predefined splines the agents must lie inside
        # the drivable area
        if self.disable_collision_check:
            return torch.zeros(vehicle.nbatch, device=self.device).bool()
        else:
            return ~lies_in_drivable_area(
                vehicle.position, self.center, self.bx, self.dx, self.road_img
            )

    def add_traffic_signal(
        self,
        r1name: str,
        r2name: str,
        val: torch.Tensor,
        start_signal: int,
        times: torch.Tensor,
        colors: List[str],
        add_reverse: bool = False,
        location=None,
        location_rev=None,
    ):
        # NOTE: This function should not be called. Traffic Signals should
        #       be placed in the preprocessed `pth` files. Else use our
        #       interactive map generator to do the same.
        raise NotImplementedError

    def add_vehicle(self, vehicle, spline_idx):
        self.vehicles[vehicle.name] = vehicle
        nbatch = vehicle.position.size(0)
        for b in range(nbatch):
            name = vehicle.name + f"_{b}"
            self.traffic_signals_in_path[name] = deque()
            self.traffic_signals_in_path[name].append(
                (
                    self.traffic_signals[
                        self.map_data["starts_to_signal"][spline_idx[b][0]]
                    ],
                )
            )

    def update_state(
        self, vname: str, new_state: torch.Tensor, wait: bool = False
    ):
        vehicle = self.vehicles[vname]
        vehicle.update_state(new_state)

        if wait:
            return

        ts = self.traffic_signals_in_path

        p = vehicle.position
        locations = torch.cat(
            [
                v[0][0][1].unsqueeze(0).to(self.device)
                if len(v) > 0
                else torch.ones(1, 2).type_as(p) * 1e12
                for n, v in ts.items()
            ]
        )

        head = vehicle.optimal_heading_to_point(locations)

        crossed = torch.abs(head) > math.pi / 2

        for b, v in enumerate(ts.values()):
            if crossed[b] and len(v) != 0:
                v.popleft()
