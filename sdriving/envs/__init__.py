import inspect
import sys

# Environments using a single model for end-to-end prediction
from sdriving.envs.base_env import BaseEnv

# Hierarchical Environments
from sdriving.envs.base_hierarchical import BaseHierarchicalEnv
from sdriving.envs.continuous_cars import (
    RoadIntersectionContinuousFlowControlAccelerationEnv,
    RoadIntersectionContinuousFlowControlEnv,
)
from sdriving.envs.fixed_track_env import (
    RoadIntersectionControlAccelerationEnv,
    RoadIntersectionContinuousAccelerationEnv,
)
from sdriving.envs.intersection_env import (
    RoadIntersectionControlEnv,
    RoadIntersectionControlImitateEnv,
    RoadIntersectionEnv,
)

# Legacy code for experimentaion
from sdriving.envs.meta_control import MetaControlEnv
from sdriving.envs.spline_env import (
    SplineRoadIntersectionAccelerationControlEnv,
)

REGISTRY = {}

clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)

for clsname, cls in clsmembers:
    if issubclass(cls, BaseEnv):
        REGISTRY[clsname] = cls
