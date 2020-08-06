import logging as lg
import math
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from celluloid import Camera

from sdriving.tsim.road import RoadNetwork
from sdriving.tsim.traffic_signal import TrafficSignal
from sdriving.tsim.utils import (
    check_intersection_lines,
    generate_lidar_data,
    angle_normalize,
)

matplotlib.use("Agg")


class World:
    def __init__(
        self,
        road_network: RoadNetwork,
        figsize: Tuple[int] = (10, 10),
        no_signal_val: float = 0.75,
    ):
        self.vehicles = OrderedDict()
        self.trajectory_nodes = OrderedDict()
        self.trajectory_points = OrderedDict()
        self.traffic_signals_in_path = OrderedDict()
        self.traffic_signals = OrderedDict()
        self.road_network = road_network
        self.objects = OrderedDict()

        self.no_signal_val = no_signal_val

        self.figsize = figsize
        self.fig = None
        self.ax = None
        self.cam = None

        self.axis_cache = OrderedDict()
        self.device = torch.device("cpu")

    def to(self, device: torch.device):
        if device == self.device:
            return
        self.transfer_dict(self.__dict__, device)
        self.device = device

    def transfer_dict(self, d: Union[dict, OrderedDict], device: torch.device):
        for k, t in d.items():
            if torch.is_tensor(t):
                d[k] = t.to(device)
            elif hasattr(t, "to"):
                t.to(device)
            elif isinstance(t, (dict, OrderedDict)):
                self.transfer_dict(t, device)

    def shortest_path_trajectory(
        self,
        start_pt: torch.Tensor,  # N x 2
        end_pt: torch.Tensor,  # N x 2
        orientation: torch.Tensor,  # N x 1
    ):
        return self.road_network.shortest_path_trajectory(
            start_pt, end_pt, orientation,
        )  # N x B  Assume all shortest paths are of equal length

    def check_collision(self, vname: str):
        vehicle = self.vehicles[vname]
        p1, p2 = self.road_network.get_edges()  # N x 2, N x 2
        p3, p4 = vehicle.get_edges()  # B x 4 x 2, B x 4 x 2

        p1s, p2s = [p1], [p2]
        for obj in self.objects.values():
            edges = obj.get_edges()
            p1s.append(edges[0])
            p2s.append(edges[1])
        p1 = torch.cat(p1s, dim=0)
        p2 = torch.cat(p2s, dim=0)

        p3, p4 = p3.view(-1, 2), p4.view(-1, 2)
        return (
            check_intersection_lines(p1, p2, p3, p4)
            .any(-1)
            .view(-1, 4)
            .any(-1)
        )

    def get_lidar_data(self, vname: str, npoints: int):
        return self.get_lidar_data_from_state(
            self.vehicles[vname].get_state(), vname, npoints,
        )

    def get_lidar_data_from_state(
        self, state: torch.Tensor, vname: str, npoints: int
    ):
        p1, p2 = [], []
        e1, e2 = self.road_network.get_edges()
        p1.append(e1)
        p2.append(e2)
        for n, v in self.vehicles.items():
            if n is vname:
                continue
            e1, e2 = v.get_edges()
            p1.append(e1)
            p2.append(e2)
        vehicle = self.vehicles[vname]
        return generate_lidar_data(
            state[:, :2],
            state[:, 2:3],
            torch.cat(p1),
            torch.cat(p2),
            npoints,
            vehicle.min_lidar_range,
            vehicle.max_lidar_range,
        )

    def update_world_state(self, tstep=1):
        for obj in self.objects.values():
            obj.step(tstep)
        for ts, _ in self.traffic_signals.values():
            ts.update_lights(tstep)

    def to(self, device: torch.device):
        if device == self.device:
            return
        self.transfer_dict(self.__dict__, device)
        self.device = device

    def transfer_dict(self, d: Union[dict, OrderedDict], device: torch.device):
        for k, t in d.items():
            if torch.is_tensor(t):
                d[k] = t.to(device)
            elif hasattr(t, "to"):
                t.to(device)
            elif isinstance(t, (dict, OrderedDict)):
                self.transfer_dict(t, device)

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
        r1end = -1
        r2end = -1
        rd1 = self.road_network.roads[r1name]
        rd2 = self.road_network.roads[r2name]
        for i, rds in rd1.road_connections.items():
            if r2name in rds:
                r1end = i
                break
        for i, rds in rd2.road_connections.items():
            if r1name in rds:
                r2end = i
                break
        node1 = self.road_network.point_to_node(rd1.end_coordinates[r1end])
        node2 = self.road_network.point_to_node(rd2.end_coordinates[r2end])
        signal = TrafficSignal(
            val=val,
            start_signal=start_signal,
            times=times,
            name=f"{node1} --> {node2}",
            colors=colors,
        )
        self.traffic_signals[(node1, node2,)] = (
            signal,
            rd1.end_coordinates[r1end]
            if location is None
            else torch.as_tensor(location),
        )
        if add_reverse:
            signal = TrafficSignal(
                val=val,
                start_signal=start_signal,
                times=times,
                name=f"{node2} --> {node1}",
                colors=colors,
            )
            self.traffic_signals[(node2, node1,)] = (
                signal,
                rd2.end_coordinates[r2end]
                if location_rev is None
                else torch.as_tensor(location_rev),
            )

    def add_object(self, obj):
        self.objects[obj.name] = obj

    def add_vehicle(self, vehicle):
        self.vehicles[vehicle.name] = vehicle
        traj_points, traj_nodes = self.shortest_path_trajectory(
            vehicle.position, vehicle.destination, vehicle.orientation
        )
        if isinstance(traj_nodes, list):
            traj_nodes = [
                torch.cat([t, -torch.ones(1).long()]) for t in traj_nodes
            ]
            traj_points = [
                torch.cat([p, vehicle.destination[i]])
                for i, p in enumerate(traj_points)
            ]
        else:
            traj_nodes = torch.cat(
                [traj_nodes, -torch.ones(traj_nodes.size(0), 1)], dim=-1
            )
            traj_points = torch.cat(
                [traj_points, vehicle.destination.unsqueeze(1)], dim=1
            )
        self.trajectory_nodes[vehicle.name] = traj_nodes
        self.trajectory_points[vehicle.name] = traj_points

        nbatch = vehicle.position.size(0)
        self.current_positions[vehicle.name] = torch.zeros(nbatch).long()
        for b in range(nbatch):
            name = vehicle.name + str(b)
            self.traffic_signals_in_path[name] = deque()
            for i1, i2 in zip(traj_nodes[:-1], traj_nodes[1:]):
                if (i1, i2) in self.traffic_signals:
                    self.traffic_signals_in_path[name].append(
                        (self.traffic_signals[(i1, i2)], i1)
                    )

    def update_state(self, vname: str, new_state: torch.Tensor):
        vehicle = self.vehicles[vname]
        vehicle.update_state(new_state)

        pos = vehicle.position
        tar = self.trajectory_points[vname][
            :, self.current_positions[vname], :
        ]
        nodes = self.trajectory_nodes[vname][:, self.current_positions[vname]]
        head = vehicle.optimal_heading_from_point(tar)

        crossed = torch.abs(head) > math.pi / 2

        self.current_positions[vname] += crossed

        for b in range(new_state.size(0)):
            name = vname + str(b)
            ts = self.traffic_signals_in_path[name]
            if (
                crossed[b]
                and ts.count() != 0
                and (ts[0][-1] == nodes[b]).all()
            ):
                ts.popleft()

    def get_traffic_signal(self, vname: str):
        vehicle = self.vehicles[vname]
        ts = self.traffic_signals_in_path
        names = [vname + str(b) for b in vehicle.nbatch]
        p = vehicle.position

        locations = torch.cat(
            [
                ts[n][0][0].unsqueeze(0)
                if ts[n].count() > 0
                else torch.ones(1, 2).type_as(p) * 1e12
                for n in names
            ]
        )

        distances = (locations - vehicle.position).pow(2).sum(-1).sqrt()

        visible = distances < vehicle.vision_range
        signals = []

        for n in names:
            if visible[n] and ts[n].count() > 0:
                signals.append(tn[n][0][1].get_value())
            else:
                signals.append(self.no_signal_val)

        return torch.as_tensor(signals).type_as(p)
