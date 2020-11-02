import logging as lg
import math
import os
from collections import OrderedDict, deque
from typing import List, Tuple, Union

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
from celluloid import Camera

from sdriving.tsim.objects import render_object
from sdriving.tsim.road import RoadNetwork
from sdriving.tsim.traffic_signal import TrafficSignal
from sdriving.tsim.utils import (
    angle_normalize,
    check_intersection_lines,
    generate_lidar_data,
    remove_batch_element,
)
from sdriving.tsim.vehicle import render_vehicle

matplotlib.use("Agg")

X = [-10, 10]
Y = [-10, 10]


class World:
    def __init__(
        self,
        road_network: RoadNetwork,
        figsize: Tuple[int] = (10, 10),
        no_signal_val: float = 0.75,
        xlims: tuple = (-100, 100),
        ylims: tuple = (-100, 100),
    ):
        self.vehicles = OrderedDict()
        self.trajectory_nodes = OrderedDict()
        self.trajectory_points = OrderedDict()
        self.traffic_signals_in_path = OrderedDict()
        self.traffic_signals = OrderedDict()
        self.road_network = road_network
        self.objects = OrderedDict()
        self.current_positions = OrderedDict()

        self.no_signal_val = no_signal_val

        self.figsize = figsize
        self.fig = None
        self.ax = None
        self.cam = None

        self.device = torch.device("cpu")

        self.xlims = xlims
        self.ylims = ylims

    def initialize_communication_channel(self, num: int, width: int):
        self.comm_channel = [
            torch.zeros(
                (num, width), dtype=torch.float, device=self.device
            ),  # The communication channel which agents populate
            torch.zeros(num, 2, device=self.device),
            # Location from where it was broadcasted
        ]

    def broadcast_data(self, data: torch.Tensor, location: torch.Tensor):
        self.comm_channel[0] = data
        self.comm_channel[1] = location

    def get_broadcast_data_all_agents(self):
        vehicle = self.vehicles["agent"]

        # Agents can only see things that are in front of them and within their
        # range of vision
        data, broadcast_locations = self.comm_channel  # (N x W, N x 2)
        if data.size(0) == 0:
            return torch.rand((0, data.size(1))).type_as(data)

        broadcast_locations = broadcast_locations.view(1, -1, 2).repeat(
            vehicle.nbatch, 1, 1
        )  # N x N x 2

        head = vehicle.optimal_heading_to_points(broadcast_locations)  # N x N
        dist = vehicle.distance_from_points(broadcast_locations)[
            ..., 0
        ]  # N x N

        dist_to_visible = (
            (dist > vehicle.vision_range) + (head.abs() > math.pi / 6)
        ) * 1e12 + dist  # N x N

        _, idxs = dist_to_visible.min(1)

        value = []
        for idx in idxs:
            value.append(data[idx : (idx + 1), :])
        return torch.cat(value)

    def remove(self, aname: str, idx: int):
        del self.traffic_signals_in_path[aname]

        self.trajectory_nodes["agent"] = remove_batch_element(
            self.trajectory_nodes["agent"], idx
        )
        self.trajectory_points["agent"] = remove_batch_element(
            self.trajectory_points["agent"], idx
        )
        self.current_positions["agent"] = remove_batch_element(
            self.current_positions["agent"], idx
        )

        if hasattr(self, "comm_channel"):
            self.comm_channel = [
                remove_batch_element(self.comm_channel[0], idx),
                remove_batch_element(self.comm_channel[1], idx),
            ]

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
        dest_orientation: torch.Tensor,  # N x 1
    ):
        return self.road_network.shortest_path_trajectory(
            start_pt, end_pt, orientation, dest_orientation
        )  # N x B  Assume all shortest paths are of equal length

    def get_road_edges(self):
        return self.road_network.get_edges()

    def check_collision(self, vname: str):
        vehicle = self.vehicles[vname]
        p1, p2 = self.get_road_edges()  # N x 2, N x 2
        p3, p4 = vehicle.get_edges()  # B x 4 x 2, B x 4 x 2

        p1s, p2s = [p1], [p2]
        for obj in self.objects.values():
            edges = obj.get_edges()
            p1s.append(edges[0])
            p2s.append(edges[1])
        p1 = torch.cat(p1s, dim=0)  # N x 2
        p2 = torch.cat(p2s, dim=0)  # N x 2

        p3, p4 = p3.view(-1, 2), p4.view(-1, 2)
        return (
            check_intersection_lines(p1, p2, p3, p4)  # (B x 4) x N
            .any(-1)
            .view(-1, 4)
            .any(-1)
        )

    def get_all_vehicle_state(self):
        states = [v.get_state() for v in self.vehicles.values()]
        return torch.cat(states)

    def get_vehicle_state(self, vname: str):
        return self.vehicles[vname].get_state()

    def get_lidar_data_all_vehicles(self, npoints: int, **kwargs):
        return torch.cat(
            [self.get_lidar_data(v, npoints, **kwargs) for v in self.vehicles]
        )

    def get_lidar_data(self, vname: str, npoints: int, **kwargs):
        return self.get_lidar_data_from_state(
            self.get_vehicle_state(vname), vname, npoints, **kwargs
        )

    def get_lidar_data_from_state(
        self,
        state: torch.Tensor,
        vname: str,
        npoints: int,
        ignore_vehicles: bool = False,
        ignore_road_edges: bool = False,
        ignore_objects: bool = False,
    ):
        assert not (ignore_road_edges and ignore_vehicles), AssertionError(
            "All objects cannot be ignored"
        )
        p1, p2 = [], []
        if not ignore_road_edges:
            e1, e2 = self.get_road_edges()
            p1.append(e1)
            p2.append(e2)
        if not ignore_objects:
            for obj in self.objects.values():
                edges = obj.get_edges()
                p1.append(edges[0])
                p2.append(edges[1])
        if not ignore_vehicles:
            for n, v in self.vehicles.items():
                e1, e2 = v.get_edges()
                p1.append(e1.view(-1, 2))
                p2.append(e2.view(-1, 2))
        vehicle = self.vehicles[vname]
        return generate_lidar_data(
            state[:, :2],  # B x 2
            state[:, 3:4],  # B x 1
            torch.cat(p1),  # N x 2
            torch.cat(p2),  # N x 2
            npoints,
            vehicle.min_lidar_range,
            vehicle.max_lidar_range,
        )

    def update_world_state(self, tstep=1):
        for obj in self.objects.values():
            obj.step(tstep)
        for ts, _ in self.traffic_signals.values():
            ts.update_lights(tstep)

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
        node1 = self.road_network.point_to_node[rd1.end_coordinates[r1end]]
        node2 = self.road_network.point_to_node[rd2.end_coordinates[r2end]]
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

    def add_vehicle(self, vehicle, trajectory: bool = True):
        self.vehicles[vehicle.name] = vehicle
        self.trajectory = trajectory

        if not trajectory:
            return

        traj_points, traj_nodes = self.shortest_path_trajectory(
            vehicle.position,
            vehicle.destination,
            vehicle.orientation,
            vehicle.dest_orientation,
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
                [traj_nodes, -torch.ones(traj_nodes.size(0), 1).long()], dim=-1
            )
            traj_points = torch.cat(
                [traj_points, vehicle.destination.unsqueeze(1)], dim=1
            )
        self.trajectory_nodes[vehicle.name] = traj_nodes
        self.trajectory_points[vehicle.name] = traj_points

        nbatch = vehicle.position.size(0)
        self.current_positions[vehicle.name] = torch.zeros(nbatch).long()
        tn = traj_nodes.detach().cpu().numpy()
        for b in range(nbatch):
            name = vehicle.name + f"_{b}"
            self.traffic_signals_in_path[name] = deque()
            for i1, i2 in zip(tn[b, :-1], tn[b, 1:]):
                if (i1, i2) in self.traffic_signals:
                    self.traffic_signals_in_path[name].append(
                        (self.traffic_signals[(i1, i2)], i1)
                    )

    def update_state(self, vname: str, new_state: torch.Tensor, wait: bool):
        vehicle = self.vehicles[vname]
        vehicle.update_state(new_state)

        if wait or not self.trajectory:
            return

        vehicle.position
        tar = torch.cat(
            [
                self.trajectory_points[vname][
                    i : (i + 1), self.current_positions[vname][i], :
                ]
                for i in range(new_state.size(0))
            ]
        )
        head = vehicle.optimal_heading_to_point(tar)

        crossed = torch.abs(head) > math.pi / 2
        nodes = self.trajectory_nodes[vname]

        cp = self.current_positions[vname]
        for b, ts in enumerate(self.traffic_signals_in_path.values()):
            node = nodes[b, cp[b]]
            if crossed[b] and len(ts) != 0 and (ts[0][-1] == node).all():
                ts.popleft()
        self.current_positions[vname] = torch.clamp(
            cp + crossed[:, 0], max=self.trajectory_points[vname].size(1) - 1
        )

    def get_all_traffic_signal(self):
        return torch.cat([self.get_traffic_signal(n) for n in self.vehicles])

    def get_traffic_signal(self, vname: str):
        vehicle = self.vehicles[vname]
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

        distances = (locations - vehicle.position).pow(2).sum(-1).sqrt()

        visible = distances < vehicle.vision_range
        signals = []

        for i, (n, v) in enumerate(ts.items()):
            if visible[i] and len(v) > 0:
                signals.append(v[0][0][0].get_value())
            else:
                signals.append(self.no_signal_val)

        return torch.as_tensor(signals).type_as(p)

    def reset(self):
        self.current_positions = OrderedDict()
        for ts, _ in self.traffic_signals.values():
            ts.reset()
        for v in self.vehicles.values():
            del v
        self.fig = None
        self.ax = None
        self.cam = None

    def add_object(self, obj):
        self.objects[obj.name] = obj

    def _render_background(self, ax):
        self.road_network.render(ax)
        self._render_traffic_signal(ax)

    def _render_traffic_signal(self, ax):
        for signal, pt in self.traffic_signals.values():
            ax.add_artist(
                plt.Circle(
                    pt.cpu().numpy(), 1.0, color=signal.get_color(), fill=True
                )
            )

    def _render_vehicle(self, v, ax):
        if hasattr(self, "comm_channel"):
            if self.comm_channel[0].size(1) == 3:
                colors = [
                    tuple(c.detach().cpu().numpy().tolist())
                    for c in self.comm_channel[0]
                ]
            elif self.comm_channel[0].size(1) == 1:
                colors = [
                    [*c.detach().cpu().numpy().tolist(), 0, 0]
                    for c in self.comm_channel[0]
                ]
            else:
                colors = ["blue"] * v.position.size(0)
        else:
            colors = ["blue"] * v.position.size(0)
        render_vehicle(v, ax, color=colors)

    def render(
        self, pts=None, path=None, lims=None, render_lidar=False, **kwargs
    ):
        if path is not None:
            ani = self.cam.animate(blit=True, interval=80)
            path_root, path_ext = os.path.splitext(path)
            if path_ext == ".gif":
                lg.warning(
                    "Writing GIF is very slow!!! Writing to an MP4 instead"
                )
                path = path_root + ".mp4"
            ani.save(path)
            self.fig = None
            self.ax = None
            self.cam = None
            return

        if self.fig is None:
            self.fig = kwargs.get("fig", plt.figure(figsize=self.figsize))
            self.ax = kwargs.get("ax", self.fig.add_subplot(1, 1, 1))
            self.cam = Camera(self.fig)
            if hasattr(self, "xlims"):
                plt.xlim(self.xlims)
                plt.ylim(self.ylims)
            plt.grid(False)

        if lims is not None:
            self.ax.set_xlim(lims["x"])
            self.ax.set_ylim(lims["y"])

        self._render_background(self.ax)

        if pts is None:
            pts = {}
        for key, pt in pts.items():
            if isinstance(pt, list):
                for point in pt:
                    if len(point) != 2:
                        break
                    plt.plot(
                        point[0],
                        point[1],
                        color="r",
                        marker="x",
                        markersize=5,
                    )

        for v in self.vehicles.values():
            self._render_vehicle(v, self.ax)
        for obj in self.objects.values():
            render_object(obj, self.ax)

        self.cam.snap()
