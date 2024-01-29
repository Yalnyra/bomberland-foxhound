import os

from components.types import Unit

ACTIONS = ["up", "down", "left", "right", "bomb", "detonate", "idle"]
UNITS = [unit.value for unit in Unit]
MAX_UNIT_VALUE = max(UNITS)

FWD_MODEL_URI = os.environ.get(
    "FWD_MODEL_CONNECTION_STRING") or "ws://127.0.0.1:6969/?role=admin"
FWD_MODEL_CONNECTION_RETRIES = 10
FWD_MODEL_CONNECTION_DELAY = 5
