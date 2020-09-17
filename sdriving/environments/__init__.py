import inspect
import sys

# Environments using a single model for end-to-end prediction
from sdriving.environments.base_env import BaseMultiAgentDrivingEnvironment
from sdriving.environments.fixed_track import (
    MultiAgentRoadIntersectionFixedTrackDiscreteCommunicationEnvironment,
    MultiAgentRoadIntersectionFixedTrackDiscreteEnvironment,
    MultiAgentRoadIntersectionFixedTrackEnvironment,
)
from sdriving.environments.highway import (
    MultiAgentHighwayBicycleKinematicsDiscreteModel,
    MultiAgentHighwayBicycleKinematicsModel,
    MultiAgentHighwayPedestriansFixedTrackDiscreteModel,
    MultiAgentHighwayPedestriansSplineAccelerationDiscreteModel,
    MultiAgentHighwaySplineAccelerationDiscreteModel,
)
from sdriving.environments.intersection import (
    MultiAgentRoadIntersectionBicycleKinematicsDiscreteEnvironment,
    MultiAgentRoadIntersectionBicycleKinematicsEnvironment,
)
from sdriving.environments.nuscenes import (
    MultiAgentNuscenesIntersectionBicycleKinematicsDiscreteEnvironment,
    MultiAgentNuscenesIntersectionBicycleKinematicsEnvironment,
    MultiAgentNuscenesIntersectionDrivingCommunicationDiscreteEnvironment,
    MultiAgentNuscenesIntersectionDrivingDiscreteEnvironment,
    MultiAgentNuscenesIntersectionDrivingEnvironment,
)
from sdriving.environments.spline_dual_objective_env import (
    MultiAgentIntersectionSplineAccelerationDiscreteEnvironment,
)
from sdriving.environments.spline_env import (
    MultiAgentOneShotSplinePredictionEnvironment,
)

REGISTRY = {}

clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)

for clsname, cls in clsmembers:
    if issubclass(cls, BaseMultiAgentDrivingEnvironment):
        REGISTRY[clsname] = cls
