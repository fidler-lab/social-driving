import math
from itertools import combinations
from typing import List, Tuple

import torch
from sdriving.tsim.road import Road, RoadNetwork
from sdriving.tsim.traffic_signal import TrafficSignal
from sdriving.tsim.utils import transform_2d_coordinates
from sdriving.tsim.world import World


def generate_nway_intersection_block(
    n: int,
    closed: List[bool] = [False] * 4,
    length: float = 40.0,
    road_width: float = 20.0,
    name: str = "intersection",
    center: torch.Tensor = torch.zeros(1, 2),
    orientation: torch.Tensor = torch.zeros(1),
    has_endpoints: List[bool] = [True] * 4,
) -> RoadNetwork:
    road_centers = []
    road_orientations = []

    base_angle = 2 * math.pi / n
    dist = (length + road_width / math.tan(base_angle / 2)) / 2
    
    orient = orientation + torch.arange(0, n).float().unsqueeze(1) * base_angle  # N x 1
    center = center + dist * torch.cat([torch.cos(orient), torch.sin(orient)], dim=-1)

    can_cross = [[True, False, not i, False] for i in closed]
    has_endpoints = [[True, False, i, False] for i in has_endpoints]

    # Generate the roads
    roads = [
        Road(
            f"{name}_{i}",
            center[i:(i + 1), :],
            length,
            road_width,
            orient[i],
            can_cross=can_cross[i],
            has_endpoints=has_endpoints[i],
        )
        for i in range(n)
    ]

    network = RoadNetwork()
    # Add the roads to the road network
    [network.add_road(roads[i]) for i in range(n)]

    # Add connections between the roads to generate the graph
    for i1, i2 in combinations(range(n), 2):
        network.join_roads(roads[i1].name, 1, roads[i2].name, 1)

    return network


def generate_intersection_world_4signals(
    closed: List[bool] = [True] * 4,
    length: float = 40.0,
    road_width: float = 20.0,
    name: str = "intersection",
    center: torch.Tensor = torch.zeros(1, 2),
    orientation: torch.Tensor = torch.zeros(1),
    has_endpoints: List[bool] = [False] * 4,
    time_green: int = 100,
    ordering: int = 0,
) -> World:
    net = generate_nway_intersection_block(
        4, closed, length, road_width, name, center, orientation, has_endpoints
    )
    net.construct_graph()

    world = World(net)

    world.add_traffic_signal(
        f"{name}_0",
        f"{name}_2",
        val=torch.as_tensor([0.0, 0.5, 1.0, 0.5]),
        start_signal=0 if ordering == 0 else 2,
        times=torch.as_tensor([time_green - 10, 10, time_green - 10, 10]),
        colors=["g", "y", "r", "y"],
        add_reverse=True,
    )
    world.add_traffic_signal(
        f"{name}_1",
        f"{name}_3",
        val=torch.as_tensor([0.0, 0.5, 1.0, 0.5]),
        start_signal=2 if ordering == 0 else 0,
        times=torch.as_tensor([time_green - 10, 10, time_green - 10, 10]),
        colors=["g", "y", "r", "y"],
        add_reverse=True,
    )

    return world