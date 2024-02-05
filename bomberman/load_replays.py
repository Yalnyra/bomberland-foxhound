import json
import os
import glob
from tqdm import tqdm
import numpy as np
import asyncio
import components.environment.state as game_state
from components.environment.observation import observation_from_state
from components.utils.observation import guess_action_based_on_gamestate_change
from components.reward import calculate_reward
import copy

async def load_replay_file_as_trajectory(source_file):
    with open(source_file, 'rt') as f:
        # NOTE: we're replacing "owner_unit_id" with "unit_id" to make older replays from the 1068 server compatible, too.
        raw_data = f.read().replace("owner_unit_id","unit_id")
        data = json.loads(raw_data)
    # initialize a GameState object from the "initial_state" JSON packet
    game = game_state.GameState(None)
    game._state = data['payload']['initial_state']
    game._current_tick = game._state.get("tick")
    # initialize output array
    trajectory = []
    trajectory.append(copy.deepcopy(game._state))
    # get a list of all updates that the server sent. we will replay those
    update_packets = data['payload']['history']
    # do the packet replay
    while len(update_packets) > 0:
        # advance time
        game._current_tick += 1
        # apply all packets for that time
        while len(update_packets) > 0 and update_packets[0]['tick'] <= game._current_tick:
            await game._on_game_tick(update_packets[0])
            del update_packets[0]
        game._state['tick'] = game._current_tick
        # store the result
        trajectory.append(copy.deepcopy(game._state))
    return trajectory

async def convert_replay_to_unit_data(replay_file, unit_id, output_file):
    # load replay as trajectory
    trajectory = await load_replay_file_as_trajectory(replay_file)
    agent_id = 'a' if unit_id in ['c', 'e', 'g'] else 'b'
    obs, actions, rewards = [], [], []
    for i in range(len(trajectory)-1):
        # convert time step i
        obs.append(observation_from_state(trajectory[i], unit_id))
        actions.append(guess_action_based_on_gamestate_change(trajectory[i],trajectory[i+1], unit_id, old_schema=True))
        reward = calculate_reward(trajectory[i],trajectory[i+1], agent_id, unit_id)
        rewards.append(reward)
        game_over = (trajectory[i+1]['unit_state'][unit_id]['hp'] <= 0)
        if game_over:
            # add obs after last action
            obs.append(observation_from_state(trajectory[i+1], unit_id))
            break
        if i > 1000:
            print("CONVERSION ERROR, REPLAY TOO LONG", replay_file)
            return
    obs = np.array(obs[0], np.float32)
    actions = np.array(actions, np.int32)
    rewards = np.array(rewards[0], np.float32)
    np.savez_compressed(output_file, obs=obs, actions=actions, rewards=rewards)


if __name__ == "__main__":
    os.makedirs("./converted", exist_ok=True)
    for replay_file in tqdm(glob.glob('./replays/*.json')):
        for unit_id in list('cegdfh'):
            output_file = f'./converted/{os.path.basename(replay_file).replace(".json", "")}_{unit_id}.npz'
            asyncio.run(convert_replay_to_unit_data(replay_file, unit_id, output_file))