import asyncio
import os
import typing

from components.environment.state import GameState

GAME_CONNECTION_URI = os.environ.get(
    'GAME_CONNECTION_STRING') or "ws://127.0.0.1:3000/?role=agent&agentId=agentId&name=defaultName"


class Agent():
    def __init__(self, model = None):
        self._model = model
        self._client = GameState(GAME_CONNECTION_URI)
        self._client.set_game_tick_callback(self._on_game_tick)

        loop = asyncio.get_event_loop()
        connection = loop.run_until_complete(self._client.connect())
        tasks = [
            asyncio.ensure_future(self._client._handle_messages(connection)),
        ]
        loop.run_until_complete(asyncio.wait(tasks))

    def _get_bomb_to_detonate(self, unit) -> typing.Union[int, int] or None:
        entities = self._client._state.get("entities")
        bombs = list(filter(lambda entity: entity.get("unit_id") == unit and entity.get("type") == "b", entities))
        bomb = next(iter(bombs or []), None)
        if bomb != None:
            return [bomb.get("x"), bomb.get("y")]
        else:
            return None

    def _is_in_bounds(self, location):

        width = self._client.get("world").get("width")
        height = self._client.get("world").get("height")

        return (location[0] >= 0 & location[0] <= width & location[1] >= 0 & location[1] <= height)
        
    def _get_surrounding_tiles(self, location):

        tile_north = (location[0], location[1] + 1)
        tile_south = (location[0], location[1] - 1)
        tile_west = (location[0] - 1, location[1])
        tile_east = (location[0] + 1, location[1])

        surrounding_tiles = [tile_north, tile_south, tile_west, tile_east]
        
        for tile in surrounding_tiles:
            if not self._is_in_bounds(tile):
                 surrounding_tiles.remove(tile)

        return surrounding_tiles
    
    def _is_occupied(self, location):

        entities = self._client.get("entities")
        units = self._client.get("unit_state")

        list_of_entity_locations = [[entity[c] for c in ['x', 'y']] for entity in entities]
        list_of_unit_locations = [units[u]["coordinates"] for u in ['c', 'd', 'e', 'f', 'g']]
        list_of_occupied_locations = list_of_entity_locations + list_of_unit_locations

        return location in list_of_occupied_locations

    def _get_empty_tiles(self, tiles):
        
        empty_tiles = []

        for tile in tiles:
            if not self._is_occupied(tile):
                empty_tiles.append(tile)

        return empty_tiles

    def _move_to_tile(self, tile, location):

        diff = tuple(x-y for x, y in zip(tile, location))

        if diff == (0,1):
            action = 'up'
        elif diff == (0,-1):
            action = 'down'
        elif diff == (1,0):
            action = 'right'
        elif diff == (-1,0):
            action = 'left'
        else:
            action = ''

        return action

    async def _on_game_tick(self, tick_number, game_state):

        my_agent_id = game_state.get("connection").get("agent_id")
        my_units = game_state.get("agents").get(my_agent_id).get("unit_ids")

        for my_unit_id in my_units:
            action = await self._make_action(game_state, my_agent_id, my_unit_id)

            if action in ["up", "left", "right", "down"]:
                await self._client.send_move(action, my_unit_id)
            elif action == "bomb":
                await self._client.send_bomb(my_unit_id)
            elif action == "detonate":
                bomb_coordinates = self._get_bomb_to_detonate(my_unit_id)
                if bomb_coordinates != None:
                    x, y = bomb_coordinates
                    await self._client.send_detonate(x, y, my_unit_id)
            elif action == "idle":
                # no-op
                continue 
            else:
                print(f"Unhandled action: {action} for unit {my_unit_id}")
