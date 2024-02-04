import math
import typing as t

from components.types import Coordinate, Observation
from components.utils.metrics import manhattan_distance


def get_entity_of_type(entity_list: Observation, entity_type: str) -> Observation or None:
    entities = dict(filter(lambda entity: entity.get("type") == entity_type, entity_list))
    return entities


def get_unit_entity_of_type(entity_list: Observation, entity_type: str, unit_id) -> Observation or None:
    entities = dict(
        filter(lambda entity: entity.get("unit_id") == unit_id and entity.get("type") == entity_type, entity_list))
    return entities


def guess_action_based_on_gamestate_change(old_state, new_state, my_unit_id):
    # get old and new positions
    ox, oy = old_state['unit_state'][my_unit_id]['coordinates']
    nx, ny = new_state['unit_state'][my_unit_id]['coordinates']
    # check if we moved
    if ny > oy: return 0
    if ny < oy: return 1
    if nx > ox: return 3
    if nx < ox: return 2
    # where are our bombs on the board?
    old_bombs = set([str(e['x']) + ',' + str(e['y']) for e in old_state['entities'] if
                     e.get('unit_id') == my_unit_id and e['type'] == 'b'])
    new_bombs = set([str(e['x']) + ',' + str(e['y']) for e in new_state['entities'] if
                     e.get('unit_id') == my_unit_id and e['type'] == 'b'])
    # is there a new one?
    if len(new_bombs.difference(old_bombs)) > 0: return 4
    # where is fire caused by our bombs on the board?
    old_fire = set([str(e['x']) + ',' + str(e['y']) for e in old_state['entities'] if
                    e.get('unit_id') == my_unit_id and e['type'] == 'x'])
    new_fire = set([str(e['x']) + ',' + str(e['y']) for e in new_state['entities'] if
                    e.get('unit_id') == my_unit_id and e['type'] == 'x'])
    # is there new fire?
    if len(new_fire.difference(old_fire)) > 0: return 5
    # apparently, we did nothing
    return 6


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
