from components.environment.config import ACTIONS
from components.types import Observation
from components.utils.observation import get_bomb_to_detonate

def make_action(observation: Observation, agent_id: str, unit_id: str, action: int):
    action = ACTIONS[action]
    if action == "bomb":
        agent_packet = {
            "type": "bomb",
            "unit_id": unit_id
        }
    elif action == "detonate":
        bomb_coordinates = get_bomb_to_detonate(observation, unit_id)
        if bomb_coordinates is not None:
            agent_packet = {
                "type": "bomb",
                "coordinates": bomb_coordinates,
                "unit_id": unit_id
            }
        return None
    elif action in ["up", "down", "left", "right"]:
        agent_packet = {
            "type": "move",
            "move": action, 
            "unit_id": unit_id
        }
    elif action == 'idle':
        # np-op
        return None
    else:
        raise NotImplementedError("Action is not available")
    return {
        "action": agent_packet,
        "agent_id": agent_id
    }
