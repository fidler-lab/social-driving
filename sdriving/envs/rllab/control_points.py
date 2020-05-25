import itertools
import random
from typing import Optional

import gym
from gym.spaces import Discrete, Box
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from ray.tune.registry import register_env
from sdriving.agents.model import ActiveSplineTorch


class ControlPointEnv(gym.Env):
    def __init__(self, config):
        self.cp_num = config.get("cp_num", 5)
        self.p_num = config.get("p_num", 100)
        self.min_length = config.get("min_length", 30)
        self.max_length = config.get("max_length", 70)
        self.min_width = config.get("min_width", 15)
        self.max_width = config.get("max_width", 30)

        actions, action_space = self._configure_action_space()
        self.actions = actions
        self.action_space = action_space
        self.observation_space = self._configure_observation_space()

        self.spline = ActiveSplineTorch(self.cp_num, self.p_num)

        self.length = None
        self.width = None
        self.max_val = None
        self.min_val = None
        self.start_pos = None
        self.goal_pos = None
        self.distance_norm = None
        self.points = None
        self.cps = None

    def _configure_action_space(self) -> tuple:
        vals = np.arange(-1.0, 1.01, 0.2)
        xy_vals = [
            torch.as_tensor(xy).unsqueeze(0)
            for xy in itertools.product(vals, vals)
        ]
        actions = list(
            itertools.product(*[xy_vals for _ in range(self.cp_num - 1)])
        )
        acts = []
        for ac in actions:
            if len(set(ac)) == self.cp_num - 1:
                acts.append(ac)
        actions = acts
        return actions, Discrete(len(actions))

    def _configure_observation_space(self) -> Box:
        return Box(
            low=np.array([-1.0] * 4 + [self.min_width, self.min_length]),
            high=np.array([1.0] * 4 + [self.max_width, self.max_length]),
        )

    def _sample_point(self) -> np.ndarray:
        # Return a point inside the intersection
        if random.choice([True, False]):
            x = (random.random() - 0.5) * self.width
            y = (2 * random.random() - 1) * (self.length + self.width / 2)
        else:
            x = (2 * random.random() - 1) * (self.length + self.width / 2)
            y = (random.random() - 0.5) * self.width
        return np.array([x, y])

    def _get_state(self) -> np.ndarray:
        rng = self.length + self.width / 2
        return torch.as_tensor(
            [
                self.start_pos[0] / rng,
                self.start_pos[1] / rng,
                self.goal_pos[0] / rng,
                self.goal_pos[1] / rng,
                self.width,
                self.length,
            ]
        ).numpy()

    def _outside_point(self, point: torch.Tensor) -> bool:
        x, y = point
        if x > self.min_val:
            if y > self.min_val or y < -self.min_val:
                return True
        elif x < -self.min_val:
            if y > self.min_val or y < -self.min_val:
                return True
        return False

    def _get_reward(self, points: torch.Tensor) -> float:
        # If successful pathing the penalty is the overall distance
        # If the pathing fails then give a -5.0 penalty
        reward = 0.0
        prev_point = points[0]
        for i, point in enumerate(points[1:], start=1):
            dist = ((point - prev_point) ** 2).sum()
            reward -= dist.item()
            prev_point = points[i]
            if self._outside_point(point):
                return -5.0
        return reward / self.distance_norm

    def step(self, action) -> tuple:
        cps = [self.start_pos.unsqueeze(0)]
        cps.extend([ac * self.max_val for ac in self.actions[action]])
        cps.append(self.goal_pos.unsqueeze(0))
        cps = torch.cat(cps)

        self.cps = cps
        self.points = self.spline(cps.unsqueeze(0)).squeeze(0)

        return self._get_state(), self._get_reward(self.points), True, {}

    def reset(self) -> np.ndarray:
        self.length = (
            random.random() * (self.max_length - self.min_length)
            + self.min_length
        )
        self.width = (
            random.random() * (self.max_width - self.min_width)
            + self.min_width
        )

        self.max_val = self.length + self.width / 2
        self.min_val = self.width / 2

        self.start_pos = torch.as_tensor(self._sample_point()).float()
        self.goal_pos = torch.as_tensor(self._sample_point()).float()

        self.distance_norm = self.p_num * np.sqrt(
            (self.length * 2 + self.width) ** 2 + self.width ** 2
        )

        self.points = None
        self.cps = None

        return self._get_state()

    def _collision_lines(self):
        p1 = self.min_val
        p2 = self.max_val
        return [
            [-p1, -p2],
            [p1, -p2],
            [p1, -p1],
            [p2, -p1],
            [p2, p1],
            [p1, p1],
            [p1, p2],
            [-p1, p2],
            [-p1, p1],
            [-p2, p1],
            [-p2, -p1],
            [-p1, -p1]
        ]

    def render(self, fname: Optional[str] = None):
        plt.xlim((-100.0, 100.0))
        plt.ylim((-100.0, 100.0))

        if self.start_pos is not None:
            plt.plot(
                self.start_pos[0].item(),
                self.start_pos[1].item(),
                color='r',
                marker='o',
            )
        if self.goal_pos is not None:
            plt.plot(
                self.goal_pos[0].item(),
                self.goal_pos[1].item(),
                color='g',
                marker='o',
            )

        if self.cps is not None:
            for cp in self.cps:
                plt.plot(cp[0].item(), cp[1].item(), color='y', marker='x')

        if self.points is not None:
            for i, pt in enumerate(self.points):
                if not i % 4 == 1:
                    continue
                plt.plot(pt[0].item(), pt[1].item(), color='b', marker='.')

        edges = self._collision_lines()
        for i in range(len(edges)):
            ed1 = edges[i]
            ed2 = edges[(i + 1) % len(edges)]
            plt.plot([ed1[0], ed2[0]], [ed1[1], ed2[1]], color="black")

        if fname:
            plt.savefig(fname)


register_env("ControlPoint-v0", lambda config: ControlPointEnv(config))
