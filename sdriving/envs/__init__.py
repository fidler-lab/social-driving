import inspect
import sys

from sdriving.envs.base_env import BaseEnv
from sdriving.envs.fixed_track_env import (
    RoadIntersectionControlAccelerationEnv,
)
from sdriving.envs.intersection_env import (
    RoadIntersectionControlEnv,
    RoadIntersectionControlImitateEnv,
    RoadIntersectionEnv,
)
from sdriving.envs.continuous_cars import (
    RoadIntersectionContinuousFlowControlEnv
)
from sdriving.envs.meta_control import MetaControlEnv

REGISTRY = {}

clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)

for clsname, cls in clsmembers:
    if issubclass(cls, BaseEnv):
        REGISTRY[clsname] = cls
