import math

import numpy as np
import torch

TWO_PI = 2 * math.pi


def angle_normalize(angle: torch.Tensor) -> torch.Tensor:
    """
    Normalize the `angle` to have a value in [-pi, pi]

    Args:
        angle: Tensor of angles of shape N
    """
    angle = torch.fmod(torch.fmod(angle, TWO_PI) + TWO_PI, TWO_PI)
    return torch.where(angle > math.pi, angle - TWO_PI, angle)


def get_2d_rotation_matrix(theta: torch.Tensor) -> torch.Tensor:
    """
    If theta has 1 element returns a 2D tensor. If theta has N elements
    it is considered as a batched data and the returned value is a
    N x 2 x 2 tensor.
    """
    ctheta = torch.cos(theta)
    stheta = torch.sin(theta)
    if theta.shape == torch.Size([]) or theta.shape == torch.Size([1]):
        return torch.as_tensor([
            [ctheta, stheta], [-stheta, ctheta]
        ], device=theta.device)
    else:
        rot_matrix = torch.zeros(
            (theta.size(0), 2, 2), dtype=torch.float, device=theta.device
        )
        for i in range(theta.size(0)):
            rot_matrix[i, 0, 0] = ctheta[i]
            rot_matrix[i, 0, 1] = stheta[i]
            rot_matrix[i, 1, 0] = -stheta[i]
            rot_matrix[i, 1, 1] = ctheta[i]
        return rot_matrix


def transform_2d_coordinates(
    coordinates: torch.Tensor, theta: torch.Tensor, offset: torch.Tensor,
) -> torch.Tensor:
    return transform_2d_coordinates_rotation_matrix(
        coordinates, get_2d_rotation_matrix(theta), offset
    )


def transform_2d_coordinates_rotation_matrix(
    coordinates: torch.Tensor, rot_matrix: torch.Tensor, offset: torch.Tensor
) -> torch.Tensor:
    if not coordinates.ndim == 3:
        return torch.matmul(coordinates, rot_matrix) + offset
    else:
        return torch.bmm(coordinates, rot_matrix) + offset


def circle_segment_area(
    dist: torch.Tensor, radius: torch.Tensor
) -> torch.Tensor:
    if dist.size(0) == 1:
        if dist > radius:
            return torch.zeros(1)
        theta = 2 * torch.acos(
            torch.clamp(dist / radius, -1.0 + 1e-5, 1.0 - 1e-5)
        )
        return (theta - torch.sin(theta)) * (radius ** 2) / 2
    else:
        theta = 2 * torch.acos(
            torch.clamp(dist / radius, -1.0 + 1e-5, 1.0 - 1e-5)
        )
        zeros = torch.zeros_like(dist)
        return torch.where(
            dist > radius,
            zeros,
            (theta - torch.sin(theta)) * (radius ** 2) / 2,
        )


def circle_area_overlap(
    center1: torch.Tensor,
    center2: torch.Tensor,
    radius1: torch.Tensor,
    radius2: torch.Tensor,
) -> torch.Tensor:
    d_sq = ((center1 - center2) ** 2).sum()
    d = torch.sqrt(d_sq)

    if d < radius1 + radius2:
        a = radius1 ** 2
        b = radius2 ** 2

        x = (a - b + d_sq) / (2 * d)
        z = x ** 2

        if d <= abs(radius2 - radius1):
            return torch.as_tensor(math.pi * min(a, b))

        y = torch.sqrt(a - z)
        return (
            a * torch.asin(y / radius1)
            + b * torch.asin(y / radius2)
            - y * (x + torch.sqrt(z + b - a))
        )

    return torch.zeros(1)


def _is_bound(val: torch.Tensor):
    return ~torch.isfinite(val) + ((val >= 0.0) * (val <= 1.0))


def check_intersection_lines(
    pt1_lines: torch.Tensor,
    pt2_lines: torch.Tensor,
    point1: torch.Tensor,
    point2: torch.Tensor,
    pt_diff_lines: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        pt1_lines: End Point 1 of the Lines (N x 2)
        pt2_lines: End Point 2 of the Lines (N x 2)
        point1: End Point 1 (2)
        point2: End Point 2 (2)
        pt_diff_lines: pt2_lines - pt1_lines (N x 2)
    """
    diff = point2 - point1
    diff_ends_1 = pt1_lines - point1

    denominator = diff[1] * pt_diff_lines[:, 0] - diff[0] * pt_diff_lines[:, 1]

    ua = (
        diff[0] * diff_ends_1[:, 1] - diff[1] * diff_ends_1[:, 0]
    ) / denominator
    ub = (
        pt_diff_lines[:, 0] * diff_ends_1[:, 1]
        - pt_diff_lines[:, 1] * diff_ends_1[:, 0]
    ) / denominator

    return torch.any((ua >= 0.0) * (ua <= 1.0) * (ub >= 0.0) * (ub <= 1.0))


def distance_from_point_direction(
    point: torch.Tensor,  # 2
    theta: torch.Tensor,  # B x 1
    pt1: torch.Tensor,  # N x 2
    pt2: torch.Tensor,  # N x 2
    min_range: float = 0.5,
    max_range: float = 12.0,
) -> torch.Tensor:
    if theta.ndim == 1:
        theta = theta.unsqueeze(1)
    elif theta.ndim == 0:
        theta = theta.view(1, 1)

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
            torch.as_tensor(np.inf),
        ),
        dim=1,
    )[0]  # B x 1


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
            theta + torch.linspace(0.0, TWO_PI * (1 - 1 / npoints), npoints,)
        ),
        pt1,
        pt2,
        min_range,
        max_range,
    )
