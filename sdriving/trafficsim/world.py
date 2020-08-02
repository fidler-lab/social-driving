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
from sdriving.trafficsim.dynamics import (
    BicycleKinematicsModel as VehicleDynamics,
)

# from sdriving.trafficsim.mpc_controller import MPCController
from sdriving.trafficsim.road import RoadNetwork
from sdriving.trafficsim.traffic_signal import TrafficSignal
from sdriving.trafficsim.utils import (
    check_intersection_lines,
    generate_lidar_data,
    angle_normalize,
)
from sdriving.trafficsim.vehicle import Vehicle, VehicleEntity

matplotlib.use("Agg")

plt.style.use("seaborn-pastel")


# TODO: Testing GPU Support
class World:
    def __init__(
        self,
        road_network: RoadNetwork,
        figsize: Tuple[int] = (10, 10),
        no_signal_val: float = 0.75,
    ):
        # Name ---> VehicleEntity
        self.vehicles = OrderedDict()
        self.traffic_signals = OrderedDict()
        self.road_network = road_network
        self.objects = OrderedDict()

        self.no_signal_val = no_signal_val

        self.figsize = figsize
        self.fig = None
        self.ax = None
        self.cam = None

        self.dynamics_ordering = None
        self.global_dynamics = None
        self.global_mpc_controller = None

        self.vlims_map = OrderedDict()
        self.dims_map = OrderedDict()

        self.nbatch = 0

        self.axis_cache = OrderedDict()
        self.device = torch.device("cpu")

    def to(self, device):
        # NOTE: This should be called after `compile` is called
        if device == self.device:
            return
        self.device = device
        for v in self.vehicles.values():
            v.to(device)
        self.road_network.to(device)
        if self.global_dynamics is not None:
            self.global_dynamics.to(device)
        if self.global_mpc_controller is not None:
            self.global_mpc_controller.to(device)

    def add_traffic_signal(
        self,
        r1name: str,
        r2name: str,
        val: Union[int, list],
        start_signal: int,
        times: List[int],
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
        node1 = self.point_to_node(rd1.end_coordinates[r1end])
        node2 = self.point_to_node(rd2.end_coordinates[r2end])
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
        # The signal will be visible only when the agent is within a certain
        # distance from the signal
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

    def get_distance_from_road_axis(
        self, vname: str, pt: int, dest: torch.Tensor
    ):
        ventity = self.vehicles[vname]
        if pt != -1:
            point = self.road_network.node_to_point[pt]
        else:
            point = dest
        dist = self.road_network.directional_distance_from_point(
            ventity.road, point, ventity.vehicle.position,
        )
        return dist

    def get_traffic_signal(
        self,
        node1: int,
        node2: int,
        pt: torch.Tensor,
        visible_distance: float = 15.0,
    ):
        if node1 == -1 or node2 == -1:
            return torch.as_tensor(self.no_signal_val)
        signal = self.traffic_signals.get((node1, node2), None)
        if signal:
            if ((signal[1] - pt) ** 2).sum().sqrt() < visible_distance:
                return torch.as_tensor(signal[0].get_value())
        return torch.as_tensor(self.no_signal_val)

    def point_to_node(self, pt: torch.Tensor):
        return self.road_network.maps[self.road_network.point_to_node[pt]]

    def add_object(self, obj):
        self.objects[obj.name] = obj

    def add_vehicle(self, vehicle: Vehicle, road: str, v_lim: float):
        assert (
            road in self.road_network.roads.keys()
            or road in self.road_network.gareas.keys()
        )
        self.road_network.add_vehicle(road, vehicle)
        # Vehicles can't start in a Gray Area
        self.vehicles[vehicle.name] = VehicleEntity(vehicle, road, False)
        self.vlims_map[vehicle.name] = v_lim
        self.dims_map[vehicle.name] = vehicle.dimensions[0]

    def compile(self):
        """
        Once all the vehicles have been added. This compilation step creates
        the MPC Controller and the global dynamics model. It also creates the
        road network graph if it does not exist
        """
        if self.road_network.graph is None:
            self.road_network.construct_graph()

        # Create the VehicleDynamics object and clear the cache
        self.dynamics_ordering = list(self.vlims_map.keys())
        self.global_dynamics = VehicleDynamics(
            v_lim=list(self.vlims_map.values()),
            dim=list(self.dims_map.values()),
        )
        nbatch = len(self.vlims_map.keys())
        self.vlims_map = OrderedDict()
        self.dims_map = OrderedDict()

        # Create the MPC Controller
        self.global_mpc_controller = None # MPCController(nbatch=nbatch)

    def dynamic_environment(
        self, vehicles_remove: Optional[Union[str, List[str]]] = None
    ):
        # FIXME: This needs to be tested
        # Get the elements in the desired format
        if vehicles_remove is None:
            vehicles_remove = []
        elif isinstance(vehicles_remove, str):
            vehicles_remove = [vehicles_remove]

        for vrem in vehicles_remove:
            ventity = self.vehicles[vrem]
            # Remove the vehicle from the road
            rd = ventity.road
            if rd in self.road_network.roads.keys():
                # Vehicle on a road
                del self.road_network.roads[rd].vehicles[vrem]
            else:
                # Vehicle is on a Gray Area
                del self.road_network.gareas[rd].vehicles[vrem]
            # Remove the vehicle from the world
            del self.vehicles[vrem]
            del self.vlims_map[vrem]
            del self.dims_map[vrem]

        # In case construct_graph was not manually called
        if self.road_network.graph is None:
            self.road_network.construct_graph()

        # Construct the global MPC and Dynamics model
        nbatch = len(self.vlims_map.keys())
        if self.nbatch != nbatch or self.global_mpc_controller is None:
            self.nbatch = nbatch
            self.global_mpc_controller = MPCController(nbatch=nbatch)

        # Ensure that the order of the input later matches with the proper
        # ordering of the dynamics
        self.dynamics_ordering = list(self.vlims_map.keys())
        self.global_dynamics = VehicleDynamics(
            v_lim=list(self.vlims_map.values()),
            dim=list(self.dims_map.values()),
        )

    def shortest_path_trajectory(
        self, start_pt: torch.Tensor, end_pt: torch.Tensor, vname=None
    ):
        return self.road_network.shortest_path_trajectory(
            start_pt,
            end_pt,
            self.vehicles[vname] if vname is not None else None,
        )

    def check_collision(self, vname: str):
        ventity = self.vehicles[vname]
        if ventity.grayarea:
            return False
        p1, p2 = self.road_network.roads[ventity.road].get_edges()

        p1s, p2s = [p1], [p2]
        for obj in self.objects.values():
            edges = obj.get_edges()
            p1s.append(edges[0])
            p2s.append(edges[1])
        p1 = torch.cat(p1s, dim=0)
        p2 = torch.cat(p2s, dim=0)

        p21 = p2 - p1
        p3, p4 = ventity.vehicle.get_edges()
        for i in range(4):
            if check_intersection_lines(p1, p2, p3[i], p4[i], p21):
                return True
        return False

    def step(
        self,
        start_states: Dict[str, torch.Tensor],
        goal_states: Dict[str, torch.Tensor],
        timesteps: int,
    ):
        start_states = torch.cat(
            [s.unsqueeze(0) for s in start_states.values()]
        )
        goal_states = torch.cat([s.unsqueeze(0) for s in goal_states.values()])
        nominal_states, nominal_actions, _ = self.global_mpc_controller(
            start_states,
            goal_states,
            self.global_dynamics,
            timesteps=timesteps,
        )
        return {
            a_id: (nominal_states[:, i, :], nominal_actions[:, i, :])
            for (i, a_id) in enumerate(self.vehicles)
        }

    def get_lidar_data(self, vname: str, npoints: int, cars: bool = True):
        ventity = self.vehicles[vname]
        # Record npoints
        self.npoints = npoints

        return self.get_lidar_data_from_state(
            torch.cat(
                [
                    ventity.vehicle.position,
                    ventity.vehicle.speed.unsqueeze(0),
                    ventity.vehicle.orientation.unsqueeze(0),
                ]
            ),
            vname,
            npoints,
            cars,
        )

    def get_lidar_data_from_state(
        self, state: torch.Tensor, vname: str, npoints: int, cars: bool = True
    ):
        ventity = self.vehicles[vname]
        p1, p2 = self.road_network.get_neighbouring_edges(
            ventity.road, vname, "garea" if ventity.grayarea else "road", cars
        )
        p1s, p2s = [p1], [p2]
        for obj in self.objects.values():
            edges = obj.get_edges()
            p1s.append(edges[0])
            p2s.append(edges[1])
        p1 = torch.cat(p1s, dim=0)
        p2 = torch.cat(p2s, dim=0)
        return generate_lidar_data(
            state[:2],
            state[3],
            p1,
            p2,
            npoints,
            ventity.vehicle.min_lidar_range,
            ventity.vehicle.max_lidar_range,
        )

    def update_world_state(self, tstep=1):
        for obj in self.objects.values():
            obj.step(tstep)
        for ts, _ in self.traffic_signals.values():
            ts.update_lights(tstep)

    def update_state(self, vname, new_state, change_road_association=False):
        # Try to change road association after a fixed iterations.
        # No need to try to do it everytime
        self.vehicles[vname].vehicle.update_state(new_state)
        if change_road_association:
            prev_road = self.vehicles[vname].road
            new_pos = self.vehicles[vname].vehicle.position.detach()
            ga = self.vehicles[vname].grayarea
            if ga:
                prev_road = self.road_network.gareas[prev_road]
            else:
                prev_road = self.road_network.roads[prev_road]
                if prev_road.lies_in(new_pos):
                    return
                else:
                    del prev_road.vehicles[vname]
            rds, fallback = prev_road.get_nearest_end_roads(new_pos)
            found = False
            for rd in rds:
                found = self.road_network.roads[rd].add_vehicle(
                    self.vehicles[vname].vehicle
                )
                if found:
                    # print(f"Road changed to {rd}")
                    self.vehicles[vname].road = rd
                    self.vehicles[vname].grayarea = False
                    if ga:
                        del prev_road.vehicles[vname]
                    break
            if not ga and not found:
                # print(f"Road changed to GrayArea {fallback.name}")
                fallback.add_vehicle(self.vehicles[vname].vehicle)
                self.vehicles[vname].road = fallback.name
                self.vehicles[vname].grayarea = True

    def reset(self):
        self.vlims_map = OrderedDict()
        self.dims_map = OrderedDict()
        for rd in self.road_network.roads.values():
            keys = list(rd.vehicles.keys())
            for k in keys:
                del rd.vehicles[k]
        for ga in self.road_network.gareas.values():
            keys = list(ga.vehicles.keys())
            for k in keys:
                del ga.vehicles[k]
        for ts, _ in self.traffic_signals.values():
            ts.reset()

    def get_configuration_data(self, name: str, data: dict):
        config = {name: data}
        config["Traffic Signal"] = []
        for (ts, pos) in self.traffic_signals.values():
            pos = [pos[0].item(), pos[1].item()]
            config["Traffic Signal"].append({"name": ts.name, "position": pos})
        return config

    def render(self, *args, backend: str = "matplotlib", **kwargs):
        if backend == "matplotlib":
            self.render_matplotlib(*args, **kwargs)
        elif backend == "pyglet":
            raise NotImplementedError

    def _render_background(self, ax):
        self.road_network.render(ax)

        for signal, pt in self.traffic_signals.values():
            ax.add_artist(
                plt.Circle(
                    pt.cpu().numpy(), 1.0, color=signal.get_color(), fill=True
                )
            )

    def _render_vehicle(self, vname, ax, render_lidar):
        vehicle = self.vehicles[vname].vehicle
        vehicle.render(ax, color="blue")

        if render_lidar:
            npoints = self.npoints if hasattr(self, "npoints") else 360
            lidar = self.get_lidar_data(vname, npoints)
            angles = angle_normalize(
                vehicle.orientation
                + torch.linspace(
                    0.0, 2 * math.pi * (1 - 1 / npoints), npoints,
                )
            )
            lidar = torch.where(
                torch.isinf(lidar), torch.zeros_like(lidar), lidar
            )
            ca = torch.cos(angles)
            sa = torch.sin(angles)

            pt_x = (vehicle.position[0] + lidar * ca).numpy()
            pt_y = (vehicle.position[1] + lidar * sa).numpy()

            for i in range(npoints):
                ax.plot(
                    [vehicle.position[0].item(), pt_x[i].item()],
                    [vehicle.position[1].item(), pt_y[i].item()],
                    "b:",
                    markersize=0.01,
                )

    def render_matplotlib(
        self,
        pts=None,
        path=None,
        render_vehicle=None,
        lims=None,
        render_lidar=False,
    ):
        if render_vehicle is None:
            render_vehicle = self._render_vehicle

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
            self.fig = plt.figure(figsize=self.figsize)
            self.ax = self.fig.add_subplot(1, 1, 1)
            self.cam = Camera(self.fig)
            plt.xlim(-100.0, 100.0)
            plt.ylim(-100.0, 100.0)
            plt.grid(True)

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

        for key in self.vehicles:
            render_vehicle(key, self.ax, render_lidar)
        for obj in self.objects.values():
            obj.render(self.ax)

        self.cam.snap()
