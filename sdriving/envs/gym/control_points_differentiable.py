import torch
from sdriving.envs.gym.control_points import ControlPointEnv


class ControlPointEnvDifferentiable(ControlPointEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.continuous_actions, Exception(
            "Differentiable Objective only works for the continuous actions"
        )
        self.discount_factor = torch.linspace(1, self.p_num) / self.p_num

    def __call__(self, action: torch.Tensor):
        cps = [self.start_pos.unsqueeze(0)]
        cps.append(action.reshape(2, -1) * self.max_val)
        cps = torch.cat(cps)

        self.cps = cps
        self.points = self.spline(cps.unsqueeze(0)).squeeze(0)

        # The agent needs to reach the goal fast
        goal_distance = (
            (self.points - self.goal_pos).pow(2).sum(-1) * self.discount_factor
        ).sum()
        # print(goal_distance)
        # Minimize the length of the path
        distances = (self.points[1:, :] - self.points[:-1, :]).pow(2)
        road_length = distances.sum()
        # print(road_length)
        # Equal spacing of the points
        mean_distance = road_length.item() / self.p_num
        deviation = torch.pow(distances - mean_distance, 2).sum()
        # print(deviation)
        return goal_distance * 1e-3 + road_length * 0.1 + deviation
