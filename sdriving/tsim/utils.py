import math

import numpy as np
import torch

TWO_PI = 2 * math.pi


@torch.jit.script
def angle_normalize(angle: torch.Tensor) -> torch.Tensor:
    """
    Normalize the `angle` to have a value in [-pi, pi]

    Args:
        angle: Tensor of angles of shape N
    """
    TWO_PI = 2 * math.pi
    angle = torch.fmod(torch.fmod(angle, TWO_PI) + TWO_PI, TWO_PI)
    return torch.where(angle > math.pi, angle - TWO_PI, angle)


@torch.jit.script
def get_2d_rotation_matrix(theta: torch.Tensor) -> torch.Tensor:
    """
    If theta has 1 element returns a 2D tensor. If theta has N elements
    it is considered as a batched data and the returned value is a
    N x 2 x 2 tensor.
    """
    # theta --> N
    if theta.numel() == 1:
        while theta.ndim < 2:
            theta = theta.unsqueeze(0)
        ctheta, stheta = torch.cos(theta), torch.sin(theta)
        row1 = torch.cat([ctheta, stheta], dim=-1)
        row2 = torch.cat([-stheta, ctheta], dim=-1)
        return torch.cat([row1, row2], dim=0)
    else:
        ctheta = torch.cos(theta).unsqueeze(1).unsqueeze(1)
        stheta = torch.sin(theta).unsqueeze(1).unsqueeze(1)
        row1 = torch.cat([ctheta, stheta], dim=-1)
        row2 = torch.cat([-stheta, ctheta], dim=-1)
        return torch.cat([row1, row2], dim=1)


@torch.jit.script
def transform_2d_coordinates_rotation_matrix(
    coordinates: torch.Tensor, rot_matrix: torch.Tensor, offset: torch.Tensor
) -> torch.Tensor:
    if not coordinates.ndim == 3:
        return torch.matmul(coordinates, rot_matrix) + offset
    else:
        return torch.bmm(coordinates, rot_matrix) + offset


@torch.jit.script
def transform_2d_coordinates(
    coordinates: torch.Tensor, theta: torch.Tensor, offset: torch.Tensor,
) -> torch.Tensor:
    return transform_2d_coordinates_rotation_matrix(
        coordinates, get_2d_rotation_matrix(theta), offset
    )


@torch.jit.script
def invtransform_2d_coordinates_rotation_matrix(
    coordinates: torch.Tensor, rot_matrix: torch.Tensor, offset: torch.Tensor
) -> torch.Tensor:
    if not coordinates.ndim == 3:
        return torch.matmul(coordinates - offset, rot_matrix.inverse())
    else:
        return torch.bmm(coordinates - offset, rot_matrix.inverse())


@torch.jit.script
def circle_segment_area(
    dist: torch.Tensor, radius: torch.Tensor
) -> torch.Tensor:
    theta = 2 * torch.acos(torch.clamp(dist / radius, -1.0 + 1e-7, 1.0 - 1e-7))
    return (dist < radius) * (theta - torch.sin(theta)) * 0.5 * (radius ** 2)


@torch.jit.script
def circle_area_overlap(
    center1: torch.Tensor,  # N x 2
    center2: torch.Tensor,  # N x 2
    radius1: torch.Tensor,  # N x 1
    radius2: torch.Tensor,  # N x 1
) -> torch.Tensor:
    d_sq = ((center1 - center2) ** 2).sum(1, keepdim=True)  # N x 1
    d = torch.sqrt(d_sq)  # N x 1

    d1 = (d_sq + radius1.pow(2) - radius2.pow(2)) / (2 * d)  # N x 1
    d2 = d - d1  # N x 1

    seg_areas = (
        circle_segment_area(torch.cat([radius1, radius2]), torch.cat([d1, d2]))
        .view(2, d.size(0), 1)
        .sum(0)
    )  # N x 1

    return (d < radius1 + radius2) * seg_areas  # N x 1


@torch.jit.script
def _is_bound(val: torch.Tensor) -> torch.Tensor:
    return ~torch.isfinite(val) + ((val >= 0.0) * (val <= 1.0))


@torch.jit.script
def check_intersection_lines(
    pt1_lines: torch.Tensor,
    pt2_lines: torch.Tensor,
    point1: torch.Tensor,
    point2: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        pt1_lines: End Point 1 of the Lines (N x 2)
        pt2_lines: End Point 2 of the Lines (N x 2)
        point1: End Point 1 (B x 2)
        point2: End Point 2 (B x 2)
    """
    pt_diff_lines = pt2_lines - pt1_lines  # N x 2
    diff = point2 - point1  # B x 2
    diff_ends_1 = pt1_lines.unsqueeze(0) - point1.unsqueeze(1)  # B x N x 2

    denominator = (
        diff[:, 1:2] * pt_diff_lines[:, 0:1].T
        - diff[:, 0:1] * pt_diff_lines[:, 1:2].T
    )  # B x N

    ua = (
        diff[:, 0:1] * diff_ends_1[:, :, 1]
        - diff[:, 1:2] * diff_ends_1[:, :, 0]
    ) / denominator  # B x N
    ub = (
        pt_diff_lines[None, :, 0] * diff_ends_1[:, :, 1]
        - pt_diff_lines[None, :, 1] * diff_ends_1[:, :, 0]
    ) / denominator  # B x N

    return (ua >= 0.0) * (ua <= 1.0) * (ub >= 0.0) * (ub <= 1.0)


@torch.jit.script
def distance_from_point_direction(
    point: torch.Tensor,  # 2
    theta: torch.Tensor,  # B x 1
    pt1: torch.Tensor,  # N x 2
    pt2: torch.Tensor,  # N x 2
    min_range: float = 0.5,
    max_range: float = 12.0,
) -> torch.Tensor:
    theta = theta.view(-1, 1)
    dir1 = torch.cat([-torch.sin(theta), torch.cos(theta)], dim=1)  # B x 2

    num = torch.cat(
        [point[1] - pt2[:, 1:], pt2[:, 0:1] - point[0]], dim=1
    )  # N x 2

    dir2 = pt1 - pt2  # N x 2

    ndir = (num * dir2).sum(1, keepdim=True).permute(1, 0)  # 1 x N
    vdir = dir1 @ dir2.permute(1, 0)  # B x N
    distances = ndir / (vdir + 1e-7)  # B x N

    t1 = (point[0] + distances * dir1[:, 1:] - pt2[:, 0:1].T) / dir2[:, 0:1].T
    t2 = (point[1] - distances * dir1[:, 0:1] - pt2[:, 1:].T) / dir2[:, 1:].T

    return torch.min(
        torch.where(
            (distances >= min_range)
            * (distances <= max_range)
            * _is_bound(t1)
            * _is_bound(t2),
            distances,
            torch.as_tensor(np.inf).type_as(distances),
        ),
        dim=1,
    )[
        0
    ]  # B


@torch.jit.script
def generate_lidar_data(
    point: torch.Tensor,
    theta: torch.Tensor,
    pt1: torch.Tensor,
    pt2: torch.Tensor,
    npoints: int,
    min_range: float = 0.5,
    max_range: float = 12.0,
) -> torch.Tensor:
    return distance_from_point_direction(
        point,
        angle_normalize(
            theta
            + torch.linspace(
                0.0,
                2 * math.pi * (1 - 1 / npoints),
                npoints,
                device=theta.device,
            )
        ),
        pt1,
        pt2,
        min_range,
        max_range,
    )
