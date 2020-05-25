import inspect
import sys

import gym

from sdriving.envs.gym.control_points import ControlPointEnv


REGISTRY = {}

clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)

for clsname, cls in clsmembers:
    if issubclass(cls, gym.Env):
        REGISTRY[clsname] = cls
