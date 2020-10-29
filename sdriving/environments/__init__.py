import inspect
import sys

# Environments using a single model for end-to-end prediction
from sdriving.environments.base_env import BaseMultiAgentDrivingEnvironment

REGISTRY = {}

clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)

for clsname, cls in clsmembers:
    if issubclass(cls, BaseMultiAgentDrivingEnvironment):
        REGISTRY[clsname] = cls
