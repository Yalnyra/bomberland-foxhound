import math 
import typing as t

from components.types import Coordinate, Observation
from components.utils.metrics import manhattan_distance


"""
Bomb definition: 
    {'created': 74, 'x': 11, 'y': 10, 'type': 'b', 'unit_id': 'd', 'agent_id': 'b', 'expires': 104, 'hp': 1, 'blast_diameter': 3}
"""
def get_bomb_to_detonate(observation: Observation, unit_id: str) -> Coordinate or None:
    entities = observation["entities"]
    bombs = list(filter(lambda entity: entity.get("unit_id") == unit_id and entity.get("type") == "b", entities))
    bomb = next(iter(bombs or []), None)
    if bomb != None:
        return [bomb.get("x"), bomb.get("y")]
    else:
        return None


"""
Bomb definition: 
    {'created': 74, 'x': 11, 'y': 10, 'type': 'b', 'unit_id': 'd', 'agent_id': 'b', 'expires': 104, 'hp': 1, 'blast_diameter': 3}
"""
def get_nearest_active_bomb(observation: Observation, unit_id: str):
    unit = observation["unit_state"][unit_id]
    unit_coords = unit['coordinates']

    entities = observation["entities"]
    bombs = list(filter(lambda entity: entity.get("type") == "b", entities))

    min_distance, nearest_bomb = +math.inf, None
    for bomb in bombs:
        bomb_coords = [bomb['x'], bomb['y']]
        bomb_distance = manhattan_distance(unit_coords, bomb_coords)
        if bomb_distance < min_distance:
            min_distance = bomb_distance
            nearest_bomb = bomb

    return nearest_bomb


"""
Bomb definition: 
    {'created': 74, 'x': 11, 'y': 10, 'type': 'b', 'unit_id': 'd', 'agent_id': 'b', 'expires': 104, 'hp': 1, 'blast_diameter': 3}
"""
def get_unit_activated_bombs(observation: Observation, unit_id: str):
    entities = observation["entities"]
    unit_bombs = list(filter(lambda entity: entity["type"] == "b" and entity["unit_id"] == unit_id, entities))
    return unit_bombs
    

"""
Obstacle definitions: 
    a. Wooden Block: {"created":0, "x":10, "y":1, "type":"w", "hp":1}
    b. Ore Block: {"created":0, "x":0, "y":13, "type":"o", "hp":3}
    c. Metal Block: {"created":0, "x":3, "y":7, "type":"m"}
"""
def get_obtacles(observation: t.Dict):
    entities = observation["entities"]
    obstacles = list(filter(lambda entity: entity.get("type") in ["w", "o", "m"], entities))
    return obstacles


def get_nearest_obstacle(observation: Observation, coords: Coordinate):
    entities = observation["entities"]
    obstacles = list(filter(lambda entity: entity.get("type") in ["w", "o", "m"], entities))

    min_distance, nearest_obstacle = +math.inf, None
    for obstacle in obstacles:
        obstacle_coords = [obstacle['x'], obstacle['y']]
        obstacle_distance = manhattan_distance(coords, obstacle_coords)
        if obstacle_distance < min_distance:
            min_distance = obstacle_distance
            nearest_obstacle = obstacle

    return nearest_obstacle
