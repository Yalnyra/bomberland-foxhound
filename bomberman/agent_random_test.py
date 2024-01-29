import random
import time

from agent_base import Agent
from components.environment.config import (
    ACTIONS, 
    FWD_MODEL_CONNECTION_RETRIES, 
    FWD_MODEL_CONNECTION_DELAY
)


class BaselineAgent(Agent):
    async def _make_action(self, game_state, my_agent_id: str, my_unit_id: str):
        action = random.choice(ACTIONS)
        return action


def main():
    print("============================================================================================")
    print("Random agent")
    print("Connecting to game")
    for retry in range(FWD_MODEL_CONNECTION_RETRIES):
        try:
            BaselineAgent()
        except:
            print(f"Retrying to connect with {retry} attempt...")
            time.sleep(FWD_MODEL_CONNECTION_DELAY)
            continue
        break
    print("============================================================================================")

main()
