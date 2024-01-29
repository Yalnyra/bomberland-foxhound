import time
import torch

from agent_base import Agent
from components.environment.config import (
    ACTIONS,
    FWD_MODEL_CONNECTION_DELAY,
    FWD_MODEL_CONNECTION_RETRIES,
)
from components.models.ppo import PPO_AGENT_PATH 
from components.state import observation_to_state


class PPOAgent(Agent):
    async def _make_action(self, game_state, my_agent_id: str, my_unit_id: str):
        state = observation_to_state(game_state, current_agent_id=my_agent_id, current_unit_id=my_unit_id)
        action, *_ = self._model(state)
        action = ACTIONS[action.item()]
        return action


def main():
    print("============================================================================================")
    print("PPO agent")
    print("Connecting to game")
    for retry in range(FWD_MODEL_CONNECTION_RETRIES):
        try:
            PPOAgent(model=torch.load(PPO_AGENT_PATH))
        except Exception as e:
            print(f"Retrying to connect with {retry} attempt... Due to: {str(e)}")
            time.sleep(FWD_MODEL_CONNECTION_DELAY)
            continue
        break
    print("============================================================================================")

main()
