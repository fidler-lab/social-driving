import math
from itertools import combinations
from typing import List, Tuple

import torch
from sdriving.trafficsim.road import Road, RoadNetwork
from sdriving.trafficsim.traffic_signal import TrafficSignal
from sdriving.trafficsim.utils import transform_2d_coordinates
from sdriving.trafficsim.world import World


def generate_straight_road(
    closed: List[bool] = [True] * 2,
    length: float = 100.0,
    road_width: float = 20.0,
    name: str = "straight_road",
    center: Tuple[float] = (0.0, 0.0),
    orientation: float = 0.0,
    has_endpoints: List[bool] = [True] * 2,
) -> RoadNetwork:
    road = Road(
        name,
        center,
        length,
        road_width,
        orientation,
        can_cross=[not closed[0], False, not closed[1], False],
        has_endpoints=[has_endpoints[0], False, has_endpoints[1], False],
    )

    network = RoadNetwork()
    network.add_road(road)

    return network


def generate_nway_intersection_block(
    n: int,
    closed: List[bool] = [False] * 4,
    length: float = 40.0,
    road_width: float = 20.0,
    name: str = "intersection",
    center: Tuple[float] = (0.0, 0.0),
    orientation: float = 0.0,
    has_endpoints: List[bool] = [True] * 4,
) -> RoadNetwork:
    road_centers = []
    road_orientations = []

    base_angle = 2 * math.pi / n
    dist = (length + road_width / math.tan(base_angle / 2)) / 2
    for i in range(n):
        phi = base_angle * i
        orient = phi + orientation
        road_centers.append(
            (
                center[0] + dist * math.cos(orient),
                center[1] + dist * math.sin(orient),
            )
        )
        road_orientations.append(orient)

    can_cross = [[True, False, not i, False] for i in closed]
    has_endpoints = [[True, False, i, False] for i in has_endpoints]

    # Generate the roads
    roads = [
        Road(
            f"{name}_{i}",
            road_centers[i],
            length,
            road_width,
            road_orientations[i],
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
    center: Tuple[float] = (0.0, 0.0),
    orientation: float = 0.0,
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
        val=[0.0, 0.5, 1.0, 0.5],
        start_signal=0 if ordering == 0 else 2,
        times=[time_green - 10, 10, time_green - 10, 10],
        colors=["g", "y", "r", "y"],
        add_reverse=True,
    )
    world.add_traffic_signal(
        f"{name}_1",
        f"{name}_3",
        val=[0.0, 0.5, 1.0, 0.5],
        start_signal=2 if ordering == 0 else 0,
        times=[time_green - 10, 10, time_green - 10, 10],
        colors=["g", "y", "r", "y"],
        add_reverse=True,
    )

    return world


def generate_intersection_world_12signals(
    closed: List[bool] = [True] * 4,
    length: float = 40.0,
    road_width: float = 20.0,
    name: str = "intersection",
    center: Tuple[float] = (0.0, 0.0),
    orientation: float = 0.0,
    has_endpoints: List[bool] = [False] * 4,
    time_green: int = 100,
    ordering: int = 0,
) -> World:
    net = generate_nway_intersection_block(
        4, closed, length, road_width, name, center, orientation, has_endpoints
    )
    net.construct_graph()

    world = World(net)

    # The routing is based on left lanes
    col_map = {0.0: "g", 1.0: "r", 0.5: "y"}

    vals = []
    vals.append([0.0, 0.5] + [1.0, 1.0] * 2 + [1.0, 0.5])
    for i in range(2, len(vals[0]), 2):
        vals.append(vals[0][-i:] + vals[0][:-i])

    mapping = {}
    mapping[(0, 1)] = vals[0]
    mapping[(0, 2)] = vals[0]
    mapping[(0, 3)] = vals[0]
    mapping[(1, 0)] = vals[1]
    mapping[(1, 2)] = vals[1]
    mapping[(1, 3)] = vals[1]
    mapping[(2, 0)] = vals[2]
    mapping[(2, 1)] = vals[2]
    mapping[(2, 3)] = vals[2]
    mapping[(3, 0)] = vals[3]
    mapping[(3, 1)] = vals[3]
    mapping[(3, 2)] = vals[3]
    colors = {}

    for idx, v in mapping.items():
        colors[idx] = [col_map[value] for value in v]

    times = [time_green - 20, 20] * len(vals)

    for rd_pair in [(0, 2), (2, 0), (1, 3), (3, 1)]:
        world.add_traffic_signal(
            f"{name}_{rd_pair[0]}",
            f"{name}_{rd_pair[1]}",
            val=mapping[rd_pair],
            start_signal=ordering,
            times=times,
            colors=colors[rd_pair],
        )

    orientation = torch.as_tensor(orientation)
    center = torch.as_tensor(center)
    for i, loc in enumerate(
        [
            (road_width / 5, road_width / 2),
            (-road_width / 2, road_width / 5),
            (-road_width / 5, -road_width / 2),
            (road_width / 2, -road_width / 5),
        ]
    ):
        idx = ((i + 1) % 4, i)
        world.add_traffic_signal(
            f"{name}_{idx[0]}",
            f"{name}_{idx[1]}",
            val=mapping[idx],
            start_signal=ordering,
            times=times,
            colors=colors[idx],
            location=transform_2d_coordinates(
                torch.as_tensor(loc), orientation, center
            ),
        )

    for i, loc in enumerate(
        [
            (road_width / 2, road_width / 5),
            (-road_width / 5, road_width / 2),
            (-road_width / 2, -road_width / 5),
            (road_width / 5, -road_width / 2),
        ]
    ):
        idx = (i, (i + 1) % 4)
        world.add_traffic_signal(
            f"{name}_{idx[0]}",
            f"{name}_{idx[1]}",
            val=mapping[idx],
            start_signal=ordering,
            times=times,
            colors=colors[idx],
            location=transform_2d_coordinates(
                torch.as_tensor(loc), orientation, center
            ),
        )

    return world
