import torch
import typing as t
import numpy as np

from components.types import Map, Observation, Unit
from components.environment.config import ACTIONS, MAX_UNIT_VALUE

"""
Utils for encoding
"""

def dec2bin(n: int, length: t.Optional[int] = None) -> str:
    b = bin(n)[2:]
    if length is None:
        return b
    return b.zfill(length)


def encode_number(n: int, length: int): 
    b = dec2bin(n, length)
    return [int(c) for c in b]


"""
State and action dimensions: HYBRID approach
"""

def map_dimensions(observation: Observation):
    width, height = (
        observation['world']['width'], 
        observation['world']['height']
    )
    return width, height


def unit_dimensions():
    max_binary_unit_value = dec2bin(MAX_UNIT_VALUE)
    unit_dimensionality = len(max_binary_unit_value)
    return unit_dimensionality

def action_dimensions():
    action_dimensionality = len(ACTIONS)
    return action_dimensionality


def state_dimensions(observation: Observation):
    n_units = unit_dimensions()
    n_coordinates = coordinates_dimensions(observation)
    n_width, n_height = map_dimensions(observation)
    n_cells = n_width * n_height
    state_dimensionality = n_units * n_cells + n_coordinates
    return state_dimensionality 


def coordinates_length(observation: Observation):
    decimal_width, decimal_height = map_dimensions(observation)
    binary_width, binary_height = (
        dec2bin(decimal_width),
        dec2bin(decimal_height)
    )
    max_binary_dimension = max(
        len(binary_width), 
        len(binary_height)
    )
    return max_binary_dimension


def coordinates_dimensions(observation: Observation):
    max_binary_dimension = coordinates_length(observation)
    return max_binary_dimension * 2 # as there are 2 dimensions: width and height 


"""
Map encoding: HYBRID approach
"""

def empty_map(observation: Observation) -> Map:
    return np.zeros(map_dimensions(observation), dtype=np.int8)


def entities_map(observation: Observation) -> Map:
    map = empty_map(observation)
    for entity in observation['entities']:
        if entity['type'] == 'w':
            map[entity['x']][entity['y']] = Unit.WoodenBlock
        if entity['type'] == 'o':
            map[entity['x']][entity['y']] = Unit.OreBlock
        if entity['type'] == 'm':
            map[entity['x']][entity['y']] = Unit.MetalBlock
        if entity['type'] == 'fp':
            map[entity['x']][entity['y']] = Unit.FreezePowerup
        if entity['type'] == 'bp':
            map[entity['x']][entity['y']] = Unit.BlastPowerup
        if entity['type'] == 'x':
            map[entity['x']][entity['y']] = Unit.Blast
        if entity['type'] == 'b':
            map[entity['x']][entity['y']] = Unit.Bomb
        if entity['type'] == 'a':
            map[entity['x']][entity['y']] = Unit.Ammunition
    return map
    

def units_map(observation: Observation, current_agent_id: str) -> Map:
    map = empty_map(observation) 
    for observed_agent_id, observed_agent_config in observation['agents'].items():
        for unit_id in observed_agent_config['unit_ids']:
            unit_config = observation['unit_state'][unit_id]
            if observed_agent_id == current_agent_id:
                map[unit_config['coordinates'][0]][unit_config['coordinates'][1]] = Unit.Friend
            else:
                map[unit_config['coordinates'][0]][unit_config['coordinates'][1]] = Unit.Enemy
    return map

"""
State encoding: HYBRID approach
"""

def encode_unit(unit: int):
    unit_binary_dimension = unit_dimensions()
    return encode_number(n=unit, length=unit_binary_dimension)


def encode_map(map: Map):
    ohe_vector = []
    for i, _ in enumerate(map):
        for _, unit in enumerate(map[i]):
            ohe_vector.append(encode_unit(unit))
    return np.array(ohe_vector, dtype=np.int8)


def encode_coordinates(observation: Observation, current_unit_id: str):
    binary_coords_length = coordinates_length(observation)
    binary_unit_coords = [
        encode_number(n=coord, length=binary_coords_length) 
        for coord in observation['unit_state'][current_unit_id]['coordinates']
    ] 
    return np.array(binary_unit_coords, dtype=np.int8)


def observation_to_state(observation: Observation, current_agent_id: str, current_unit_id: str):
    map: Map = (
        empty_map(observation) 
        | units_map(observation, current_agent_id) 
        | entities_map(observation)
    )
    state = np.append(
        encode_map(map).astype(np.float32),
        encode_coordinates(observation, current_unit_id).astype(np.float32)
    )
    return torch.tensor(state).flatten()
