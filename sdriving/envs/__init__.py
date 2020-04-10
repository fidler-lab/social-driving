import inspect
import sys

from sdriving.envs.base_env import BaseEnv
from sdriving.envs.intersection_env import (
    RoadIntersectionControlEnv,
    RoadIntersectionEnv,
)
from sdriving.envs.meta_control import MetaControlEnv

REGISTRY = {}

clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)

for clsname, cls in clsmembers:
    if issubclass(cls, BaseEnv):
        REGISTRY[clsname] = cls
