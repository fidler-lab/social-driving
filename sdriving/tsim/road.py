import random
import string
from collections import OrderedDict
from itertools import combinations
from typing import List, Optional, Tuple, Union

import networkx as nx

import matplotlib.pyplot as plt
import numpy as np
import torch
from sdriving.tsim.utils import (
    get_2d_rotation_matrix,
    transform_2d_coordinates,
    transform_2d_coordinates_rotation_matrix,
    is_perpendicular
)


class GrayArea:
    def __init__(self, name: str):
        self.name = name

        # Names of the connected roads. This is the adjacency list used in the
        # RoadNetwork
        self.roads = []
        self.rends = []

        self.center = torch.zeros(0, 2)
        self.device = torch.device("cpu")

    def to(self, device):
        if device == self.device:
            return
        self.device = device
        self.center = self.center.to(device)

    def connect_road(self, road, end: int, net):
        self.roads.append(road.name)
        self.rends.append(end)

        center = torch.zeros(1, 2)
        for rname, rend in zip(self.roads, self.rends):
            center += net.roads[rname].end_coordinates[rend]
        self.center = center / len(self.roads)


class Road:
    def __init__(
        self,
        name: str,
        center: torch.Tensor, # 1 x 2
        x_len: float,
        y_len: float,
        orientation: torch.Tensor,
        can_cross: List[bool] = [True, False, True, False],
        has_endpoints: Optional[List[bool]] = None,
    ):
        self.name = name

        self.base_coordinates = torch.as_tensor([
            [-x_len / 2, -y_len / 2],
            [-x_len / 2, y_len / 2],
            [x_len / 2, y_len / 2],
            [x_len / 2, -y_len / 2],
        ])

        self.x_len = x_len
        self.y_len = y_len
        self.orientation = orientation
        self.rot_matrix = get_2d_rotation_matrix(self.orientation)
        self.inv_rot_matrix = self.rot_matrix.inverse()
        self.center = center

        self.coordinates = transform_2d_coordinates(
            self.base_coordinates - self.center, self.orientation, self.center
        )
        self.device = torch.device("cpu")

        cant_cross = [not cc for cc in can_cross]
        self.pt1 = self.coordinates[cant_cross]
        self.pt2 = torch.cat([
            self.coordinates[1:], self.coordinates[0:1],
        ])[cant_cross]

        self.end_coordinates = OrderedDict()

        # Names of the connected roads. This is the adjacency list used in the
        # RoadNetwork
        self.road_connections = OrderedDict()
        self.ga_connections = OrderedDict()
        if has_endpoints is None:
            has_endpoints = can_cross
        for i, cc in enumerate(has_endpoints[:-1], 1):
            if cc:
                self.end_coordinates[i] = torch.mean(
                    self.coordinates[(i - 1) : (i + 1)], dim=0
                )
        if has_endpoints[-1]:
            self.end_coordinates[4] = (
                self.coordinates[0] + self.coordinates[-1]
            ) / 2
        for i, cc in enumerate(can_cross, 1):
            if cc:
                self.road_connections[i] = []
                self.ga_connections[i] = None

    def to(self, device: torch.device):
        if device == self.device:
            return
        self.transfer_dict(self.__dict__, device)
        self.device = device
    
    def transfer_dict(
        self, d: Union[dict, OrderedDict], device: torch.device
    ):
        for k, t in d.items():
            if torch.is_tensor(t):
                d[k] = t.to(device)
            elif hasattr(t, "to"):
                t.to(device)
            elif isinstance(t, (dict, OrderedDict)):
                self.transfer_dict(t, device)

    def connect_road(self, road, point: int):
        assert point in self.road_connections.keys(), AssertionError(
            f"Road has noopening on {point} end"
        )
        assert road.name not in self.road_connections[point]
        self.road_connections[point].append(road.name)

    @staticmethod
    def rand_uniform(
        size: Union[List[int], Tuple[int], int], low: float, high: float
    ):
        """Uniformly sample in the range [low, high]
        """
        return torch.rand(size) * (high - low) + low

    def sample(
        self,
        size: Union[List[int], Tuple[int], int] = 1,
        x_bound: float = 0.75,
        y_bound: float = 0.75,
    ):
        bounds = torch.as_tensor([x_bound * self.x_len, y_bound * self.y_len])
        return transform_2d_coordinates_rotation_matrix(
            self.rand_uniform((size, 2), bounds * 0.5, bounds * -0.5,),
            self.rot_matrix,
            self.offset,
        ).to(self.device)

    def get_edges(self):
        return self.pt1, self.pt2


class RoadNetwork:
    def __init__(self):
        self.pt1 = torch.zeros(0, 2)
        self.pt2 = torch.zeros(0, 2)
        self.centers = torch.zeros(0, 2)
        self.graph = None

        self.roads = OrderedDict()
        self.gareas = OrderedDict()
        self.name_to_center_idx = OrderedDict()
        self.name_to_edges_idx = OrderedDict()
        self.c_components = -1
        self.e_components = -1

        self.device = torch.device("cpu")

    def get_edges(self):
        return self.pt1, self.pt2

    def to(self, device: torch.device):
        if device == self.device:
            return
        self.transfer_dict(self.__dict__, device)
        self.device = device
    
    def transfer_dict(
        self, d: Union[dict, OrderedDict], device: torch.device
    ):
        for k, t in d.items():
            if torch.is_tensor(t):
                d[k] = t.to(device)
            elif hasattr(t, "to"):
                t.to(device)
            elif isinstance(t, (dict, OrderedDict)):
                self.transfer_dict(t, device)
    
    def add_road(self, road: Road):
        assert road.name not in self.roads.keys(), AssertionError(
            f"{road.name} is already registered"
        )
        self.roads[road.name] = road
        self.e_components += 1
        self.name_to_edges_idx[road.name] = self.e_components
        pt1, pt2 = road.get_edges()
        self.pt1 = torch.cat([self.pt1, pt1])
        self.pt2 = torch.cat([self.pt2, pt2])
        self.c_components += 1
        self.name_to_center_idx[road.name] = self.c_components
        self.centers = torch.cat([self.centers, road.center])
    
    def add_garea(self):
        garea = GrayArea("".join([
            random.choice(string.ascii_letters + string.digits)
            for n in range(10)
        ]))
        self.gareas[garea.name] = garea
        self.c_components += 1
        self.name_to_center_idx[garea.name] = self.c_components
        self.centers = torch.cat([self.centers, garea.center])

    def join_roads(self, rname1: str, r1point: int, rname2: str, r2point: int):
        assert rname1 in self.roads, AssertionError(
            f"Register {rname1} using add_road()"
        )
        assert rname2 in self.roads, AssertionError(
            f"Register {rname2} using add_road()"
        )

        self.roads[rname1].connect_road(self.roads[rname2], r1point)
        self.roads[rname2].connect_road(self.roads[rname1], r2point)

        garea = None
        garea = self.roads[rname1].ga_connections[r1point]
        if garea is None:
            garea = self.roads[rname2].ga_connections[r2point]

        # All road connections should have a GrayArea in between them
        if garea is None:
            self.add_garea()
        if self.roads[rname1].ga_connections[r1point] is None:
            garea.connect_road(self.roads[rname1], r1point, self)
            self.roads[rname1].ga_connections[r1point] = garea
        if self.roads[rname2].ga_connections[r2point] is None:
            garea.connect_road(self.roads[rname2], r2point, self)
            self.roads[rname2].ga_connections[r2point] = garea
    
    def is_perpendicular(
        self,
        name: List[str],  # N
        pt1: torch.Tensor,  # N x 2
        pt2: torch.Tensor,  # N x 2
        tol: float = 1e-1,
    ) -> torch.BoolTensor:
        """Checks if the two vectors (pt1 - pt2) amd (pt1 - center)
        are perpendicular to each other.
        """
        center = self.centers[
            [self.name_to_center_idx[n] for n in name], :
        ]  # N x 2
        return is_perpendicular(pt1, pt2, center, tol)
    
    # TODO: Implement function to get distance from road axis

    def construct_graph(self):
        # This requires the networkx package
        graph = nx.Graph()
        self.point_to_node = {}
        count = 0
        # Add all the nodes to the graph. Also add all the connections between
        # end points of the same road
        for road in self.roads.values():
            for end in road.end_coordinates.values():
                self.point_to_node[end] = count
                graph.add_node(count, location=end.cpu().numpy())
                count += 1
            for p1, p2 in combinations(list(road.end_coordinates.values()), 2):
                graph.add_edge(
                    self.point_to_node[p1],
                    self.point_to_node[p2],
                    weight=((p1 - p2) ** 2).sum().sqrt(),
                )
        self.node_to_point = list(self.point_to_node.keys())
        self.all_nodes = torch.cat(
            [x.unsqueeze(0) for x in self.node_to_point]
        )
        self.maps = {i: i for i in range(self.all_nodes.size(0))}
        # Now draw edges based on inter road connections
        pt = []
        for _, ga in self.gareas.items():
            pt = []
            for r, end in zip(ga.roads, ga.rends):
                pt.append(self.roads[r].end_coordinates[end])
            for p1, p2 in combinations(pt, 2):
                graph.add_edge(
                    self.point_to_node[p1],
                    self.point_to_node[p2],
                    weight=((p1 - p2) ** 2).sum().sqrt(),
                )

        # Contract edges if the nodes are very close
        for ed in graph.edges:
            try:
                if graph.edges[ed]["weight"] < 0.5:
                    graph = nx.minors.contracted_edge(graph, ed, False)
                    self.maps[ed[1]] = ed[0]
            except KeyError:
                continue

        self.graph = graph
        return graph
    
    def nearest_graph_node(self, pt: torch.Tensor):
        return self.maps[
            torch.argmin(((pt - self.all_nodes) ** 2).sum(1)).item()
        ]

#     def shortest_path_trajectory(
#         self,
#         start_pt: torch.Tensor,
#         end_pt: torch.Tensor,
#         vehicle
#         ventity: Optional[VehicleEntity] = None,
#     ):
#         if ventity is None:
#             start_node = self.nearest_graph_node(start_pt)
#         else:
#             # TODO: What to do in case of end points being closed
#             nopt = np.inf
#             for pt in self.roads[ventity.road].end_coordinates.values():
#                 opt = torch.abs(ventity.vehicle.optimal_heading_to_point(pt))
#                 if opt < nopt:
#                     nopt = opt
#                     start_node = self.maps[self.point_to_node[pt]]
#         end_node = self.nearest_graph_node(end_pt)
#         path = nx.astar_path(self.graph, start_node, end_node)
#         if len(path) < 2:
#             return path
#         if ((self.node_to_point[path[-2]] - end_pt) ** 2).sum() < (
#             (self.node_to_point[path[-2]] - self.node_to_point[path[-1]]) ** 2
#         ).sum():
#             return path[:-1]
#         return path
    
    # Need a generic_sample function which uses triangulation to sample from
    # an arbitrary polygon
    def sample(
        self,
        size: Union[Tuple[int], List[int], int] = 1,
        x_bound: float = 0.75,
        y_bound: float = 0.75,
    ):
        roads = list(self.roads.values())
        samples = []
        for i in range(size):
            road = random.choice(roads)
            samples.append((road.name, road.sample(1, x_bound, y_bound)[0]))
        return samples
    
    def render(self, ax):
        for p1, p2 in zip(self.pt1, self.pt2):
            p1 = p1.detach().cpu().numpy()
            p2 = p2.detach().cpu().numpy()
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='r')