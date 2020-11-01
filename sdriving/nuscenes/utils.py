from typing import List, Tuple, Union

import numpy as np
import torch
from nuscenes.map_expansion.map_api import NuScenesMap
from shapely.geometry import LineString, Polygon

from sdriving.tsim import transform_2d_coordinates


def get_coordinates_of_polygon(poly: Polygon) -> List[Tuple[float]]:
    coords = list(poly.exterior.coords)
    for interior in poly.interiors:
        coords.extend(list(interior.coords))
    return coords


def get_edges_of_polygon(poly: Polygon) -> Tuple[List[Tuple[float]]]:
    coords1 = list(poly.exterior.coords)
    coords2 = coords1[1:] + [coords1[0]]
    for interior in poly.interiors:
        coords = interior.coords
        coords1.extend(coords)
        coords2.extend(coords[1:] + [coords[0]])
    return (coords1, coords2)


def get_edges_of_polygon_in_patch(
    poly: Polygon, box: Union[List[float], List[Tuple[float]]]
) -> Tuple[List[Tuple[float]]]:
    if isinstance(box[0], float):
        box = [
            (box[0], box[1]),
            (box[0], box[3]),
            (box[2], box[3]),
            (box[2], box[1]),
        ]

    pt1, pt2 = get_edges_of_polygon(poly)
    box = Polygon(box)
    coords1, coords2 = [], []
    for p1, p2 in zip(pt1, pt2):
        if LineString([p1, p2]).intersects(box):
            coords1.append(p1)
            coords2.append(p2)
    return (coords1, coords2)


def preprocess_map_edges(
    pt1: torch.Tensor, pt2: torch.Tensor, passes: int = 4, tol: float = 1e-2
) -> Tuple[torch.Tensor]:
    start_nodes = pt1.size(0)
    for i in range(passes):
        pt1_processed, pt2_processed = [], []
        diff = pt2 - pt1
        theta = torch.atan2(diff[:, 1], diff[:, 0])
        tdiff = theta[1:] - theta[:-1]
        idx = 0
        while idx < tdiff.size(0):
            if torch.abs(tdiff[idx]) < tol and torch.all(
                pt1[idx + 1] == pt2[idx]
            ):
                pt1_processed.append(pt1[idx].unsqueeze(0))
                pt2_processed.append(pt2[idx + 1].unsqueeze(0))
                if idx == tdiff.size(0) - 2:
                    pt1_processed.append(pt1[-1].unsqueeze(0))
                    pt2_processed.append(pt2[-1].unsqueeze(0))
                idx += 2
            else:
                pt1_processed.append(pt1[idx].unsqueeze(0))
                pt2_processed.append(pt2[idx].unsqueeze(0))
                if idx == tdiff.size(0) - 1:
                    pt1_processed.append(pt1[-1].unsqueeze(0))
                    pt2_processed.append(pt2[-1].unsqueeze(0))
                idx += 1
        pt1 = torch.cat(pt1_processed)
        pt2 = torch.cat(pt2_processed)
        if start_nodes == pt1.size(0):
            break
        start_nodes = pt1.size(0)

    return pt1, pt2


def realign_map_edges(
    pt1: torch.Tensor, pt2: torch.Tensor, theta: float
) -> Tuple[torch.Tensor]:
    theta = torch.as_tensor(theta)
    offset = torch.mean(pt1 + pt2, dim=0) / 2
    pt1 = transform_2d_coordinates(pt1, theta, -offset)
    pt2 = transform_2d_coordinates(pt2, theta, -offset)
    return pt1, pt2


def nuscenes_map_to_line_representation(
    nusc_map: NuScenesMap, patch: List[float], realign: bool = False
) -> Tuple[torch.Tensor]:
    record = nusc_map.get_records_in_patch(patch, ["drivable_area"])
    pt1, pt2 = [], []
    for da_token in record["drivable_area"]:
        da = nusc_map.get("drivable_area", da_token)
        for poly in map(nusc_map.extract_polygon, da["polygon_tokens"]):
            p1, p2 = get_edges_of_polygon_in_patch(poly, patch)
            if len(p1) > 0 and len(p2) > 0:
                p1, p2 = preprocess_map_edges(
                    torch.as_tensor(p1),
                    torch.as_tensor(p2),
                    passes=10,
                    tol=0.1,
                )
                pt1.append(p1)
                pt2.append(p2)
    pt1, pt2 = torch.cat(pt1), torch.cat(pt2)
    if realign:
        pt1, pt2 = realign_map_edges(pt1, pt2, 0.0)

    centers = (pt1 + pt2) / 2
    centers1 = centers.unsqueeze(1)
    centers2 = centers.unsqueeze(0)
    dist = (centers2 - centers1).pow(2).sum(dim=-1).sqrt()
    for i in range(centers.size(0)):
        dist[i, i] = 1e12
    very_close = (dist < 0.01).any(dim=-1)

    to_remove = []
    for i, c in enumerate(very_close):
        if c:
            to_remove.append(i)

    for i, rem in enumerate(to_remove):
        rem = rem - i
        pt1 = torch.cat([pt1[:rem], pt1[rem + 1 :]])
        pt2 = torch.cat([pt2[:rem], pt2[rem + 1 :]])

    return pt1, pt2


# Not optimizing these as they are only for preprocessing and used only
# once.
def get_drivable_area_matrix(
    data: dict, patch: List[float], res: int = 100
) -> Tuple[torch.Tensor]:
    xs = np.array(
        [
            np.linspace(
                data["center"][0] - data["width"] / 2 * 1.1,
                data["center"][0] + data["width"] / 2 * 1.1,
                res,
            )
            for _ in range(res)
        ]
    ).T.flatten()
    ys = np.array(
        [
            np.linspace(
                data["center"][1] - data["height"] / 2 * 1.1,
                data["center"][1] + data["height"] / 2 * 1.1,
                res,
            )
            for _ in range(res)
        ]
    ).flatten()

    drivable_area = np.array(data["road_img"])
    ixes = (
        np.array([xs, ys]).T
        - np.array([data["center"]])
        - np.array(data["bx"])[:2]
        + np.array(data["dx"])[:2] / 2.0
    ) / np.array(data["dx"])[:2]
    ixes = ixes.astype(int)
    within = np.logical_and(0 <= ixes[:, 0], ixes[:, 0] < data["nx"])
    within = np.logical_and(within, 0 <= ixes[:, 1])
    within = np.logical_and(within, ixes[:, 1] < data["ny"])
    drivable = np.zeros(len(ixes))
    drivable[within] = drivable_area[ixes[within, 0], ixes[within, 1]]

    return (
        torch.as_tensor(drivable.reshape(res, res)).float(),
        torch.as_tensor(xs.reshape(res, res)).float(),
        torch.as_tensor(ys.reshape(res, res)).float(),
    )


def lies_in_drivable_area(
    pos: torch.Tensor,  # N x 2
    center: np.array,  # 1 x 2
    bx: np.array,  # 2
    dx: np.array,  # 2
    drivable_area: np.array,
) -> torch.Tensor:
    pos = pos.detach().cpu().numpy()
    ixes = ((pos - center - bx + dx / 2) / dx).astype(int)
    nx, ny = drivable_area.shape
    within = np.logical_and(0 <= ixes[:, 0], ixes[:, 0] < nx)
    within = np.logical_and(within, 0 <= ixes[:, 1])
    within = np.logical_and(within, ixes[:, 1] < ny)
    drivable = np.zeros(len(ixes))
    drivable[within] = drivable_area[ixes[within, 0], ixes[within, 1]]
    return torch.as_tensor(drivable).bool()
