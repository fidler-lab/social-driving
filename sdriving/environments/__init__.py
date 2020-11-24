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
    MultiAgentIntersectionSplineAccelerationDiscreteV2Environment,
)
from sdriving.environments.spline_env import (
    MultiAgentOneShotSplinePredictionEnvironment,
)

REGISTRY = {}

clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)

for clsname, cls in clsmembers:
    if issubclass(cls, BaseMultiAgentDrivingEnvironment):
        REGISTRY[clsname] = cls

def prettify_signature(sig):
    vals = [f"{p.name}: {p.annotation} = {p.default if not p.default is inspect._empty else 'NO DEFAULT'}" for p in list(sig.parameters.values()) if not p.name in ["args", "kwargs"]]
    return (
        vals,
        list(map(lambda x: x.split(":")[0], vals)),
        list(map(lambda x: x.split(":")[1], vals))
    )


def get_parameter_list(env):
    if isinstance(env, str):
        env = REGISTRY[env]
    sigs = [inspect.signature(e) for e in inspect.getmro(env)]
    vals, lefts = [], []
    for sig in sigs:
        val, left, right = prettify_signature(sig)
        for v, l, r in zip(val, left, right):
            if l in lefts:
                continue
            vals.append(v)
            lefts.append(l)
    return "\n".join(vals)
