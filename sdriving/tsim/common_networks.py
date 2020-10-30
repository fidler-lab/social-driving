import math
from itertools import combinations
from typing import List

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
    pass

    base_angle = 2 * math.pi / n
    dist = (length + road_width / math.tan(base_angle / 2)) / 2

    orient = (
        orientation + torch.arange(0, n).float().unsqueeze(1) * base_angle
    )  # N x 1
    center = center + dist * torch.cat(
        [torch.cos(orient), torch.sin(orient)], dim=-1
    )

    can_cross = [[True, False, not i, False] for i in closed]
    has_endpoints = [[True, False, i, False] for i in has_endpoints]

    # Generate the roads
    roads = [
        Road(
            f"{name}_{i}",
            center[i : (i + 1), :],
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


def generate_intersection_world_12signals(
    closed: List[bool] = [True] * 4,
    length: float = 40.0,
    road_width: float = 20.0,
    name: str = "intersection",
    center: torch.Tensor = torch.zeros(1, 2),
    orientation: torch.Tensor = torch.zeros(1),
    has_endpoints: List[bool] = [False] * 4,
    time_green: int = 100,
    ordering: int = 0,
    default_colmap: bool = True,
    merge_same_signals: bool = False,
) -> World:
    net = generate_nway_intersection_block(
        4, closed, length, road_width, name, center, orientation, has_endpoints
    )
    net.construct_graph()

    world = World(net)

    if default_colmap:
        col_map = {0.0: "g", 1.0: "r", 0.5: "y"}
    else:
        col_map = {1.0: "g", 0.0: "r", 0.5: "y"}

    if merge_same_signals:
        for i in range(4):
            if i != 0:
                world.add_traffic_signal(
                    f"{name}_0",
                    f"{name}_{i}",
                    val=torch.as_tensor([0.0, 0.5, 1.0, 0.5]),
                    start_signal=0 if ordering == 0 else 2,
                    times=torch.as_tensor(
                        [time_green - 10, 10, time_green - 10, 10]
                    ),
                    colors=["g", "y", "r", "y"],
                    add_reverse=False,
                )
            if i != 2:
                world.add_traffic_signal(
                    f"{name}_2",
                    f"{name}_{i}",
                    val=torch.as_tensor([0.0, 0.5, 1.0, 0.5]),
                    start_signal=0 if ordering == 0 else 2,
                    times=torch.as_tensor(
                        [time_green - 10, 10, time_green - 10, 10]
                    ),
                    colors=["g", "y", "r", "y"],
                    add_reverse=False,
                )
            if i != 1:
                world.add_traffic_signal(
                    f"{name}_1",
                    f"{name}_{i}",
                    val=torch.as_tensor([0.0, 0.5, 1.0, 0.5]),
                    start_signal=2 if ordering == 0 else 0,
                    times=torch.as_tensor(
                        [time_green - 10, 10, time_green - 10, 10]
                    ),
                    colors=["g", "y", "r", "y"],
                    add_reverse=False,
                )
            if i != 3:
                world.add_traffic_signal(
                    f"{name}_3",
                    f"{name}_{i}",
                    val=torch.as_tensor([0.0, 0.5, 1.0, 0.5]),
                    start_signal=2 if ordering == 0 else 0,
                    times=torch.as_tensor(
                        [time_green - 10, 10, time_green - 10, 10]
                    ),
                    colors=["g", "y", "r", "y"],
                    add_reverse=True,
                )

        return world

    vals = []
    vals.append([0.0, 0.5] + [1.0, 1.0] * 2 + [1.0, 0.5])
    for i in range(2, len(vals[0]), 2):
        vals.append(vals[0][-i:] + vals[0][:-i])
    vals = torch.as_tensor(vals)

    mapping = {(i, j): vals[i] for i in range(4) for j in range(4) if i != j}
    colors = {
        idx: [col_map[val.item()] for val in v] for (idx, v) in mapping.items()
    }

    times = torch.as_tensor([time_green - 20, 20] * len(vals))

    for rd_pair in [(0, 2), (2, 0), (1, 3), (3, 1)]:
        world.add_traffic_signal(
            f"{name}_{rd_pair[0]}",
            f"{name}_{rd_pair[1]}",
            val=mapping[rd_pair],
            start_signal=ordering,
            times=times,
            colors=colors[rd_pair],
        )

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
            )[0],
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
            )[0],
        )

    return world
