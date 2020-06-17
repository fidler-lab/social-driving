import random
import string
from itertools import combinations
from typing import List, Optional, Tuple, Union

import networkx as nx

import matplotlib.pyplot as plt
import numpy as np
import torch
from sdriving.trafficsim.utils import (
    get_2d_rotation_matrix,
    transform_2d_coordinates,
    transform_2d_coordinates_rotation_matrix,
)
from sdriving.trafficsim.vehicle import Vehicle, VehicleEntity


class GrayArea:
    def __init__(self, name: str):
        self.name = name

        # Mapping from Vehicle name to Vehicle Object
        self.vehicles = {}

        # Names of the connected roads. This is the adjacency list used in the
        # RoadNetwork
        self.roads = []
        self.rends = []

        self.center = None
        self.device = torch.device("cpu")

    def to(self, device):
        if device == self.device:
            return
        self.device = device
        self.center = self.center.to(device)

    def connect_road(self, road, end: int, net):
        self.roads.append(road.name)
        self.rends.append(end)

        center = torch.zeros(2)
        for rname, rend in zip(self.roads, self.rends):
            center += net.roads[rname].end_coordinates[rend]
        self.center = center / len(self.roads)

    def __repr__(self):
        ret = f"Gray Area: {self.name}\n"
        if len(self.vehicles.keys()) > 0:
            ret += f"\tVehicles: {list(self.vehicles.keys())}\n"
        if len(self.roads) > 0:
            ret += f"\tConnections: {self.roads}"
        return string

    def get_nearest_end_roads(self, point: torch.Tensor):
        return self.roads, None

    def add_vehicle(self, vehicle: Vehicle):
        self.vehicles[vehicle.name] = vehicle
        return True


class Road:
    def __init__(
        self,
        name: str,
        center: Union[Tuple[float], List[float], torch.Tensor],
        x_len: int,
        y_len: int,
        orientation: float,
        can_cross: List[bool] = [True, False, True, False],
        has_endpoints: Optional[List[bool]] = None,
    ):
        self.name = name

        self.base_coordinates = torch.as_tensor(
            [
                [-x_len / 2, -y_len / 2],
                [-x_len / 2, y_len / 2],
                [x_len / 2, y_len / 2],
                [x_len / 2, -y_len / 2],
            ]
        )

        self.x_len = x_len
        self.y_len = y_len
        self.orientation = torch.as_tensor(orientation)
        self.rot_matrix = get_2d_rotation_matrix(self.orientation)
        self.inv_rot_matrix = self.rot_matrix.inverse()
        self.offset = torch.as_tensor(center)

        self.coordinates = transform_2d_coordinates(
            self.base_coordinates, self.orientation, self.offset
        )
        self.device = torch.device("cpu")

        cant_cross = [not cc for cc in can_cross]
        self.pt1 = torch.cat(
            [
                self.coordinates[0:1],
                self.coordinates[1:2],
                self.coordinates[2:3],
                self.coordinates[3:],
            ]
        )[cant_cross]
        self.pt2 = torch.cat(
            [
                self.coordinates[1:2],
                self.coordinates[2:3],
                self.coordinates[3:],
                self.coordinates[0:1],
            ]
        )[cant_cross]

        self.end_coordinates = {}

        # Dictionary mapping vehicle name to Vehicle Object
        self.vehicles = {}
        # Names of the connected roads. This is the adjacency list used in the
        # RoadNetwork
        self.road_connections = {}
        self.ga_connections = {}
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

    def to(self, device):
        if device == self.device:
            return
        self.device = device
        self.base_coordinates.to(device)
        self.orientation.to(device)
        self.rot_matrix.to(device)
        self.inv_rot_matrix.to(device)
        self.offset.to(device)
        self.coordinates.to(device)
        self.pt1.to(device)
        self.pt2.to(device)
        for key, item in self.end_coordinates.items():
            self.end_coordinates[key] = item.to(device)

    def get_nearest_end_roads(self, point: torch.Tensor):
        min_end = 0
        min_dist = np.inf
        for i, end in self.end_coordinates.items():
            dist = ((point - end) ** 2).sum()
            if dist < min_dist:
                min_end = i
                min_dist = dist
        return self.road_connections[min_end], self.ga_connections[min_end]

    def add_vehicle(self, vehicle: Vehicle):
        # Add vehicle only if it lies inside the drivable region. Else return
        # False
        if self.lies_in(vehicle.position):
            self.vehicles[vehicle.name] = vehicle
            return True
        return False

    def lies_in(self, pt):
        pt = torch.matmul(pt - self.offset, self.inv_rot_matrix)
        if (pt[0] >= -self.x_len / 2 and pt[0] <= self.x_len / 2) and (
            pt[1] >= -self.y_len / 2 and pt[1] <= self.y_len / 2
        ):
            return True
        return False

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

    def __repr__(self):
        ret = f"Road: {self.name}\n"
        ret += f"\tEnd Coordinates: "
        end_coords = list(
            map(
                lambda x: list(x[1].cpu().numpy()),
                self.end_coordinates.items(),
            )
        )
        ret += f"{end_coords}\n"
        if len(self.vehicles.keys()) > 0:
            ret += f"\tVehicles: {list(self.vehicles.keys())}\n"
        for k, rds in self.road_connections.items():
            if len(rds) > 0:
                ret += f"\tEnd {k} Connections: {rds}\n"
        return ret

    def render(self, ax, color: str = "r"):
        for i in range(self.pt1.size(0)):
            p1x, p1y = self.pt1[i].detach().cpu().numpy()
            p2x, p2y = self.pt2[i].detach().cpu().numpy()
            plt.plot([p1x, p2x], [p1y, p2y], color=color, linewidth=2)


class RoadNetwork:
    def __init__(self):
        self.roads = {}
        self.gareas = {}

        self.device = torch.device("cpu")

    def to(self, device):
        if device == self.device:
            return
        self.device = device
        for rd in self.road.values():
            rd.to(device)
        for ga in self.gareas.values():
            ga.to(device)

    def add_road(self, road: Road):
        assert road.name not in self.roads.keys(), AssertionError(
            f"{road.name} is already registered"
        )
        self.roads[road.name] = road

    def join_roads(self, rname1: str, r1point: int, rname2: str, r2point: int):
        assert rname1 in self.roads.keys(), AssertionError(
            f"Register {rname1} using add_road()"
        )
        assert rname2 in self.roads.keys(), AssertionError(
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
            garea = GrayArea(
                "".join(
                    [
                        random.choice(string.ascii_letters + string.digits)
                        for n in range(10)
                    ]
                )
            )
            self.gareas[garea.name] = garea
        if self.roads[rname1].ga_connections[r1point] is None:
            garea.connect_road(self.roads[rname1], r1point, self)
            self.roads[rname1].ga_connections[r1point] = garea
        if self.roads[rname2].ga_connections[r2point] is None:
            garea.connect_road(self.roads[rname2], r2point, self)
            self.roads[rname2].ga_connections[r2point] = garea

    def is_perpendicular(
        self,
        name: str,
        pt1: torch.Tensor,
        pt2: torch.Tensor,
        tol: float = 1e-1,
    ) -> bool:
        """Checks if the two vectors (pt1 - pt2) amd (pt1 - center)
        are perpendicular to each other.
        """
        vec1 = pt1 - pt2
        vec1 /= torch.norm(vec1)
        try:
            center = self.roads[name].offset
        except KeyError:
            center = self.gareas[name].center
        vec2 = pt1 - center
        vec2 /= torch.norm(vec2)
        return torch.abs(vec1.dot(vec2)) < tol

    def directional_distance_from_point(
        self, name: str, pt1: torch.Tensor, pt2: torch.Tensor
    ) -> torch.Tensor:
        """Computes the distance of the point `pt2` from the axis of the
        road which is given by `pt1 - center`. To ensure correctness
        the destination of an agent on the axis of the road.
        """
        if name in self.roads:
            center = self.roads[name].offset
        else:
            center = self.gareas[name].center
        dir = pt1 - center
        l2 = torch.sqrt((dir ** 2).sum())
        dir /= l2
        diff = pt2 - center
        t = diff.dot(dir)
        projection = center + t * dir
        return torch.sign(diff[0] * dir[1] - diff[1] * dir[0]) * torch.sqrt(
            ((projection - pt2) ** 2).sum()
        )

    def get_neighbouring_edges(self, name, vname=None, type="road", cars=True):
        pt1s = []
        pt2s = []
        if type == "road":
            roads = [name]
            for _, val in self.roads[name].road_connections.items():
                roads += val
        else:
            roads = self.gareas[name].roads
            if cars:
                for _, vehicle in self.gareas[name].vehicles.items():
                    if vname is not None and vname is vehicle.name:
                        continue
                    pt1, pt2 = vehicle.get_edges()
                    pt1s.append(pt1)
                    pt2s.append(pt2)
        for rname in roads:
            road = self.roads[rname]
            pt1, pt2 = road.get_edges()
            pt1s.append(pt1)
            pt2s.append(pt2)
            if cars:
                for _, vehicle in road.vehicles.items():
                    if vname is not None and vname is vehicle.name:
                        continue
                    pt1, pt2 = vehicle.get_edges()
                    pt1s.append(pt1)
                    pt2s.append(pt2)
        return torch.cat(pt1s), torch.cat(pt2s)

    def add_vehicle(self, rname: str, vehicle: Vehicle):
        res = self.roads[rname].add_vehicle(vehicle)
        assert res

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

    def shortest_path_trajectory(
        self,
        start_pt: torch.Tensor,
        end_pt: torch.Tensor,
        ventity: Optional[VehicleEntity] = None,
    ):
        if ventity is None:
            start_node = self.nearest_graph_node(start_pt)
        else:
            # TODO: What to do in case of end points being closed
            nopt = np.inf
            for pt in self.roads[ventity.road].end_coordinates.values():
                opt = torch.abs(ventity.vehicle.optimal_heading_to_point(pt))
                if opt < nopt:
                    nopt = opt
                    start_node = self.maps[self.point_to_node[pt]]
        end_node = self.nearest_graph_node(end_pt)
        path = nx.astar_path(self.graph, start_node, end_node)
        if len(path) < 2:
            return path
        if ((self.node_to_point[path[-2]] - end_pt) ** 2).sum() < (
            (self.node_to_point[path[-2]] - self.node_to_point[path[-1]]) ** 2
        ).sum():
            return path[:-1]
        return path

    def plot(self, fname: Optional[str] = None):
        # Plots generated are incorrect for road ends != 2
        if not hasattr(self, "graph"):
            for _, ga in self.gareas.items():
                pt = []
                for r, end in zip(ga.roads, ga.rends):
                    pt.append(self.roads[r].end_coordinates[end].unsqueeze(0))
                for a, b in combinations(pt, 2):
                    x, y = torch.cat([a, b]).T.cpu()
                    plt.plot(x.numpy(), y.numpy(), lw=2.0, color="black")
            pt = []
            for _, road in self.roads.items():
                pt.extend(
                    [
                        coord.unsqueeze(0)
                        for _, coord in road.end_coordinates.items()
                    ]
                )
            x, y = torch.cat(pt).T.cpu()
            for i in range(len(self.roads.keys())):
                plt.plot(
                    x[i * 2 : (i + 1) * 2].numpy(),
                    y[i * 2 : (i + 1) * 2].numpy(),
                    lw=4.0,
                    marker="o",
                )
            plt.plot()
        else:
            pos = {}
            for node in self.graph.nodes:
                pos[node] = self.graph.nodes[node]["location"]
            nx.draw_networkx_nodes(self.graph, pos)
            nx.draw_networkx_edges(self.graph, pos)
            nx.draw_networkx_labels(self.graph, pos)
        ax = plt.gca()
        ax.set_xlabel("X Coordinates")
        ax.set_ylabel("Y Coordinates")
        ax.set_title("Road Network")
        if fname:
            ax.get_figure().savefig(fname)

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

    def __repr__(self):
        val = f"Road Network:\n"
        for k in self.roads.keys():
            val += f"\tRoad: {k}\n"
        for k in self.gareas.keys():
            val += f"\tGrayArea: {k}\n"
        return val

    def render(self, ax):
        for _, road in self.roads.items():
            road.render(ax)
