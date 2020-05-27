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

        goal_distance = ((
            (self.points - self.goal_pos) ** 2
        ).sum(-1) * self.discount_factor).sum()
        road_length = ((self.points[1:, :] - self.points[:-1, :]) ** 2).sum()
        return goal_distance + road_length



