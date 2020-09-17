import inspect
import sys

# Environments using a single model for end-to-end prediction
from sdriving.environments.base_env import BaseMultiAgentDrivingEnvironment
from sdriving.environments.intersection import (
    MultiAgentRoadIntersectionBicycleKinematicsEnvironment,
    MultiAgentRoadIntersectionBicycleKinematicsDiscreteEnvironment,
)
from sdriving.environments.fixed_track import (
    MultiAgentRoadIntersectionFixedTrackEnvironment,
    MultiAgentRoadIntersectionFixedTrackDiscreteEnvironment,
    MultiAgentRoadIntersectionFixedTrackDiscreteCommunicationEnvironment,
)
from sdriving.environments.spline_env import (
    MultiAgentOneShotSplinePredictionEnvironment,
)
from sdriving.environments.spline_dual_objective_env import (
    MultiAgentIntersectionSplineAccelerationDiscreteEnvironment,
)
from sdriving.environments.nuscenes import (
    MultiAgentNuscenesIntersectionDrivingEnvironment,
    MultiAgentNuscenesIntersectionDrivingDiscreteEnvironment,
    MultiAgentNuscenesIntersectionBicycleKinematicsEnvironment,
    MultiAgentNuscenesIntersectionBicycleKinematicsDiscreteEnvironment,
    MultiAgentNuscenesIntersectionDrivingCommunicationDiscreteEnvironment,
)
from sdriving.environments.highway import (
    MultiAgentHighwayBicycleKinematicsModel,
    MultiAgentHighwayBicycleKinematicsDiscreteModel,
    MultiAgentHighwaySplineAccelerationDiscreteModel,
    MultiAgentHighwayPedestriansFixedTrackDiscreteModel,
    MultiAgentHighwayPedestriansSplineAccelerationDiscreteModel
)

REGISTRY = {}

clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)

for clsname, cls in clsmembers:
    if issubclass(cls, BaseMultiAgentDrivingEnvironment):
        REGISTRY[clsname] = cls
