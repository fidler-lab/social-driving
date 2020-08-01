import inspect
import sys

# Environments using a single model for end-to-end prediction
from sdriving.envs.base_env import BaseEnv

from sdriving.envs.continuous_cars import (
    RoadIntersectionContinuousFlowControlAccelerationEnv,
    RoadIntersectionContinuousFlowControlEnv,
)
from sdriving.envs.fixed_track_env import (
    RoadIntersectionContinuousAccelerationEnv,
    RoadIntersectionControlAccelerationEnv,
)
from sdriving.envs.intersection_env import (
    RoadIntersectionContinuousControlEnv,
    RoadIntersectionControlEnv,
    RoadIntersectionEnv,
)
from sdriving.envs.spline_env import (
    RoadIntersectionLeftRightControlEnv,
    RoadIntersectionSplineEnv,
    RoadIntersectionSplineNPointsNavigationEnv,
)
from sdriving.envs.spline_two_objectives import (
    RoadIntersectionDualObjective,
    RoadIntersectionDualObjectiveNWaypoints
)
from sdriving.envs.straight_road import StraightRoadPedestrianAvoidanceEnv

REGISTRY = {}

clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)

for clsname, cls in clsmembers:
    if issubclass(cls, BaseEnv):
        REGISTRY[clsname] = cls
