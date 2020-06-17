import inspect
import sys

import gym
from sdriving.envs.gym.control_points import ControlPointEnv
from sdriving.envs.gym.control_points_differentiable import (
    ControlPointEnvDifferentiable,
)

REGISTRY = {}

clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)

for clsname, cls in clsmembers:
    if issubclass(cls, gym.Env):
        REGISTRY[clsname] = cls
