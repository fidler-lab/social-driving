import numpy as np
import torch
from torch import nn


class IterativeWayPointPredictor(nn.Module):
    def __init__(
        self,
        hdim: int,
        max_length: int,
        max_width: int,
        separate_goal_model: bool = False,
    ):
        super().__init__()
        obs_dim = 6  # Current LOC, Target LOC, Length, Width
        self.wpoint_disp = nn.Sequential(
            nn.Linear(obs_dim, hdim), nn.ReLU(), nn.Linear(hdim, 2), nn.Tanh()
        )
        if separate_goal_model:
            self.goal_disp = nn.Sequential(
                nn.Linear(obs_dim, hdim),
                nn.ReLU(),
                nn.Linear(hdim, 2),
                nn.Tanh(),
            )
        self.max_length = float(max_length)
        self.max_width = float(max_width)

    def forward(
        self,
        start_pos: torch.Tensor,
        goals: torch.Tensor,
        length: float,
        width: float,
    ):
        # start_pos --> B x 2
        # goals --> B x N x 2
        waypoints = [start_pos.unsqueeze(1)]
        normalize = torch.ones((1, 2)) * (length + width / 2)
        for i in range(goals.size(1)):
            spos = waypoints[-1][:, 0, :]
            gpos = goals[:, i, :]
            obs = torch.cat(
                [
                    spos / normalize,
                    gpos / normalize,
                    torch.ones((spos.size(0), 1)) * length / self.max_length,
                    torch.ones((spos.size(0), 1)) * width / self.max_width,
                ],
                dim=1,
            )  # B x 6
            if hasattr(self, "goal_disp"):
                model = (
                    self.goal_disp
                    if i == goals.size(0) - 1
                    else self.wpoint_disp
                )
            else:
                model = self.wpoint_disp
            point = torch.reshape(model(obs) * width / 2, (-1, 2)) + gpos
            waypoints.append(point.unsqueeze(1))
        return torch.cat(waypoints, dim=1)  # B x (N + 1) x 2
