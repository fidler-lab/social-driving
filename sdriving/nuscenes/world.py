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
        figsize: Tuple[int] = (10, 10),
        no_signal_val: float = 0.75,
    ):
        self.vehicles = OrderedDict()
        self.traffic_signals_in_path = OrderedDict()
        self.traffic_signals = OrderedDict()
        self.objects = OrderedDict()
        self.current_positions = OrderedDict()

        self.no_signal_val = no_signal_val

        self.figsize = figsize
        self.fig = None
        self.ax = None
        self.cam = None

        self.device = torch.device("cpu")
        
        self.map_path = map_path
        self.parse_map_data()

    def parse_map_data(self):
        data = torch.load(self.map_path)
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
                list(product([k], range(len(v.keys())), range(5)))
            )
        self.sampling_indices = sampling_indices
        
        self.sampling_indices_list = copy(self.sampling_indices)

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
            alpha=0.1
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
        # return torch.zeros(vehicle.nbatch, device=self.device).bool()
        return ~lies_in_drivable_area(
            vehicle.position,
            self.center,
            self.bx,
            self.dx,
            self.road_img
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
        # TODO
        raise NotImplementedError
        
    def add_vehicle(self, vehicle, spline_idx):
        self.vehicles[vehicle.name] = vehicle
        # TODO: Add to the list of traffic signals
        nbatch = vehicle.position.size(0)
        for b in range(nbatch):
            name = vehicle.name + str(b)
            self.traffic_signals_in_path[name] = deque()
    
    def update_state(self, vname: str, new_state: torch.Tensor, wait: bool = False):
        # wait is unnecessary here, but base_env calls with it
        vehicle = self.vehicles[vname]
        vehicle.update_state(new_state)
        
        # TODO: Update the next traffic signal in path