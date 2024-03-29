import torch

from components.types import Observation
from components.utils.observation import (
    get_nearest_active_bomb,
    get_nearest_obstacle,
    get_unit_activated_bombs,
)
from components.utils.metrics import manhattan_distance
from components.utils.observation import guess_action_based_on_gamestate_change


def find_my_units_alive(observation: Observation, current_agent_id: str) -> int:
    alive = 0
    for unit_props in observation['unit_state'].values():
        if unit_props['agent_id'] == current_agent_id:
            if unit_props['hp'] != 0:
                alive += 1
    return alive


def find_enemy_units_alive(observation: Observation, current_agent_id: str) -> int:
    alive = 0
    for unit_props in observation['unit_state'].values():
        if unit_props['agent_id'] != current_agent_id:
            if unit_props['hp'] != 0:
                alive += 1
    return alive


def find_my_units_hps(observation: Observation, current_agent_id: str) -> int:
    hps = 0
    for unit_props in observation['unit_state'].values():
        if unit_props['agent_id'] == current_agent_id:
            hps += unit_props['hp']
    return hps


def find_enemy_units_hps(observation: Observation, current_agent_id: str) -> int:
    hps = 0
    for unit_props in observation['unit_state'].values():
        if unit_props['agent_id'] != current_agent_id:
            hps += unit_props['hp']
    return hps


def find_current_tick(observation: Observation) -> int:
    tick = observation['tick']
    return tick


def unit_within_reach_of_a_bomb(observation: Observation, current_unit_id: str):
    unit = observation['unit_state'][current_unit_id]
    unit_coords = unit['coordinates']
    nearest_bomb = get_nearest_active_bomb(observation, current_unit_id)
    if nearest_bomb is None:
        return False
    nearest_bomb_coords = [nearest_bomb['x'], nearest_bomb['y']]
    within_reach_of_a_bomb = manhattan_distance(unit_coords, nearest_bomb_coords) <= nearest_bomb['blast_diameter']
    return within_reach_of_a_bomb


def unit_within_safe_cell_nearby_bomb(observation: Observation, current_unit_id: str):
    unit = observation['unit_state'][current_unit_id]
    unit_coords = unit['coordinates']
    nearest_bomb = get_nearest_active_bomb(observation, current_unit_id)
    if nearest_bomb is None:
        return False
    nearest_bomb_coords = [nearest_bomb['x'], nearest_bomb['y']]
    within_safe_cell_nearby_bomb = manhattan_distance(unit_coords, nearest_bomb_coords) > nearest_bomb['blast_diameter']
    return within_safe_cell_nearby_bomb


# 11: Positive reward for maintaining distance with 2 or more agents:
def reward_for_maintaining_distance(observation: Observation, current_agent_id: str, current_unit_id: str):
    reward = 0
    unit_coords = observation['unit_state'][current_unit_id]['coordinates']
    for unit_id, unit_props in observation['unit_state'].items():
        if unit_props['agent_id'] != current_agent_id and unit_props['hp'] > 0:
            distance = manhattan_distance(unit_coords, unit_props['coordinates'])
            if distance >= 2:
                reward += 0.1  # Приклад значення, можна налаштувати
    return reward


# 12: Negative reward for friendly fire after activating your own bomb,
# multiplied by the casualty ratio created by one bomb:
def reward_for_friendly_fire(observation: Observation, current_agent_id: str):
    reward = 0
    units = ['c', 'e', 'g'] if current_agent_id == 'a' else ['d', 'f', 'h']
    # Перевіряємо, чи агент активував свою бомбу та завдав шкоди своїм товаришам
    for e in observation['entities']:
        if e['type'] != 'x': continue
        if e.get('owner_unit_id') in units or e.get('unit_id') in units:
            for unit in units:
                u_x, u_y = observation['unit_state'][unit]['coordinates']
                x, y = e['x'], e['y']
                if u_x == x and u_y == y:
                    reward -= 1 / (find_my_units_alive(observation, current_agent_id) + 0.1)  # Приклад значення, можна налаштувати
    return reward


# 13: Negative reward for killing an agent by getting it stuck between level obstacles and sudden death border tiles:
def reward_for_agent_stuck(observation: Observation, current_unit_id: str):
    reward = 0
    u_x, u_y = observation['unit_state'][current_unit_id]['coordinates']
    for e in observation['entities']:
        type = e['type']
        x, y = e['x'], e['y']
        if type in ['m', 'o', 'w']:
            if u_x == x and u_y == y:
                reward -= 1  # Приклад значення, можна налаштувати
    return reward


# 14: Positive multiplicative reward for a positive bomb/enemy losses' ratio:
def reward_for_bomb_enemy_ratio(prev_observation: Observation, next_observation: Observation,
                                current_agent_id: str):
    reward = 0

    prev_enemy_units_alive = find_enemy_units_alive(prev_observation, current_agent_id)
    next_enemy_units_alive = find_enemy_units_alive(next_observation, current_agent_id)

    bomb_ratio = (prev_enemy_units_alive - next_enemy_units_alive) / 2  # Приклад формули, можна налаштувати
    if bomb_ratio > 0:
        reward += bomb_ratio * 0.2  # Приклад значення, можна налаштувати
    return reward


# 15: dropped action
def is_action_drop(prev_observation: Observation, next_observation: Observation, unit_id, action_is_idle: bool):
    return (not action_is_idle) and \
             guess_action_based_on_gamestate_change(prev_observation, next_observation, unit_id) == 6


"""
Bomb definition: {'created': 74, 'x': 11, 'y': 10, 'type': 'b', 'unit_id': 'd', 'agent_id': 'b', 'expires': 104, 'hp': 1, 'blast_diameter': 3}
"""


def unit_activated_bomb_near_an_obstacle(observation: Observation, current_unit_id: str):
    unit_activated_bombs = get_unit_activated_bombs(observation, current_unit_id)
    if not len(unit_activated_bombs):
        return False
    for unit_bomb in unit_activated_bombs:
        unit_bomb_coords = [unit_bomb['x'], unit_bomb['y']]
        nearest_obstacle = get_nearest_obstacle(observation, unit_bomb_coords)
        if nearest_obstacle is None:
            continue
        nearest_obstacle_coords = [nearest_obstacle['x'], nearest_obstacle['y']]
        if manhattan_distance(unit_bomb_coords, nearest_obstacle_coords) <= unit_bomb['blast_diameter']:
            return True
    return False


"""
Reward function definition:
1. +0.5: when dealing 1 hp for 1 enemy
2. +1: when killing opponent
3. +1: when killing all 3 opponents
4. -0.25: when losing 1 hp for 1 teammate
5. -0.5: when losing teammate
6. -1: when losing all 3 teammates
7. -0.01: the longer game the bigger punishment is
8. -0.000666: the unit is in a cell within reach of a bomb
9. +0.002: the unit is in a safe cell when there is an active bomb nearby 
10. +0.1: the unit activated bomb near an obstacle

11. +0.1: Positive reward for maintaining distance with 2 or more agents
12. -0.5: Negative reward for friendly fire after activating your own bomb, 
                        multiplied by the casualty ratio created by one bomb
13. -1: Negative reward for killing an agent by getting it stuck between level obstacles and sudden death border tiles
14. +0.2: Positive multiplicative reward for a positive bomb/enemy losses ratio
"""


def calculate_reward(prev_observation: Observation, next_observation: Observation, current_agent_id: str,
                     current_unit_id: str, action_is_idle: bool = True):
    reward = 0

    # 1. +0.85: when dealing 1 hp for 1 enemy

    prev_enemy_units_hps = find_enemy_units_hps(prev_observation, current_agent_id)
    next_enemy_units_hps = find_enemy_units_hps(next_observation, current_agent_id)

    enemy_units_hps_diff = prev_enemy_units_hps - next_enemy_units_hps
    if enemy_units_hps_diff > 0:
        reward += (enemy_units_hps_diff * 0.85)

    # 2. +1: when killing opponent

    prev_enemy_units_alive = find_enemy_units_alive(prev_observation, current_agent_id)
    next_enemy_units_alive = find_enemy_units_alive(next_observation, current_agent_id)

    if prev_enemy_units_alive > next_enemy_units_alive:
        reward += 1

    # 3. +1: when killing all 3 opponents

    if next_enemy_units_alive == 0:
        reward += 1

    # 4. -0.5: when losing 1 hp for 1 teammate

    prev_my_units_hps = find_my_units_hps(prev_observation, current_agent_id)
    next_my_units_hps = find_my_units_hps(next_observation, current_agent_id)

    my_units_hps_diff = prev_my_units_hps - next_my_units_hps
    if my_units_hps_diff > 0:
        reward += (my_units_hps_diff * -0.5)

    # 5. -0.9: when losing teammate

    prev_my_units_alive = find_my_units_alive(prev_observation, current_agent_id)
    next_my_units_alive = find_my_units_alive(next_observation, current_agent_id)

    if next_my_units_alive < prev_my_units_alive:
        reward += (-0.9)

    # 6. -1: when losing all 3 teammates

    if next_my_units_alive == 0:
        reward += (-1)

    # 7. -0.001: the longer game the bigger punishment is

    reward += (-0.001)

    # 8. -0.0666: the agent is in a cell within reach of a bomb

    prev_within_reach_of_a_bomb = unit_within_reach_of_a_bomb(prev_observation, current_unit_id)
    next_within_reach_of_a_bomb = unit_within_reach_of_a_bomb(next_observation, current_unit_id)

    if not prev_within_reach_of_a_bomb and next_within_reach_of_a_bomb:
        reward += (-0.0666)

    # 9. +0.02: the agent is in a safe cell when there is an active bomb nearby

    prev_within_safe_cell_nearby_bomb = unit_within_safe_cell_nearby_bomb(prev_observation, current_unit_id)
    next_within_safe_cell_nearby_bomb = unit_within_safe_cell_nearby_bomb(next_observation, current_unit_id)

    if not prev_within_safe_cell_nearby_bomb and next_within_safe_cell_nearby_bomb:
        reward += 0.2

    # 10. +0.1: the unit activated bomb near an obstacle

    prev_activated_bomb_near_an_obstacle = unit_activated_bomb_near_an_obstacle(prev_observation, current_unit_id)
    next_activated_bomb_near_an_obstacle = unit_activated_bomb_near_an_obstacle(next_observation, current_unit_id)

    if not prev_activated_bomb_near_an_obstacle and next_activated_bomb_near_an_obstacle:
        reward += 0.1

    # 11.
    reward += reward_for_maintaining_distance(next_observation, current_agent_id, current_unit_id)

    # 12.
    reward += reward_for_friendly_fire(next_observation, current_agent_id)

    # 13.
    reward += reward_for_agent_stuck(next_observation, current_unit_id)

    # 14.
    reward += reward_for_bomb_enemy_ratio(prev_observation, next_observation, current_agent_id)

    # 15
    if is_action_drop(prev_observation, next_observation, current_unit_id, action_is_idle):
        reward -= 0.05

    return torch.tensor(reward, dtype=torch.float32).reshape(1)
