import numpy as np
import torch
from matplotlib import pyplot as plt
from components.utils.metrics import manhattan_distance, broadcasting_distance


def observation_from_state(state, my_unit_id):
    # get general variables
    unit_state = state['unit_state']
    my_unit = unit_state[my_unit_id]
    my_agent_id = my_unit['agent_id']
    entities = state['entities']
    tick = state['tick']
    # sort team IDs so that my team comes first
    agent_order = ['a', 'b']
    agent_order.sort(key=lambda x: int(x != my_agent_id))
    agent_order = ['x'] + agent_order
    # create a dictionary of all units grouped by team. team 'x' is us
    agent2units = {
        'x': my_unit_id,
        'a': [u for u in ['c', 'e', 'g'] if u != my_unit_id],
        'b': [u for u in ['d', 'f', 'h'] if u != my_unit_id],
    }
    # we will now loop through the teams and units to create different grayscale images
    # where each image represents one aspect of the game state, such as HP or bombs
    layers = []
    for agent in agent_order:
        tmp = np.zeros([15, 15, 5], np.float32)
        for unit_id in agent2units[agent]:
            unit = unit_state[unit_id]
            cux, cuy = unit['coordinates']
            tmp[cuy, cux, 0] = 0.5
            tmp[cuy, cux, 1] = float(max(0, unit['hp']))
            tmp[cuy, cux, 2] = float(max(0, unit['invulnerable'] - tick)) / 6.0
            # tmp[cuy, cux, 3] = min(float(unit['inventory']['bombs']), 7)
            tmp[cuy, cux, 3] = float(max(0, unit['stunned'] - tick)) / 6.0
            tmp[cuy, cux, 4] = min(float(unit['blast_diameter']) / 3.0, 7)
        layers.append((f'agent {agent} positions', tmp[:, :, 0], '01.0f'))
        layers.append((f'agent {agent} HP', tmp[:, :, 1], '01.0f'))
        layers.append((f'agent {agent} invulnerability', tmp[:, :, 2], '3.1f'))
        layers.append((f'agent {agent} stun', tmp[:, :, 3], '01.0f'))
        layers.append((f'agent {agent} blast_diameter', tmp[:, :, 4], '01.0f'))

    # draw the environment HP and fire expiration times into a map
    tiles = np.zeros([15, 15], np.uint8)
    for e in entities:
        type = e['type']
        x, y = e['x'], e['y']
        if type in ['m', 'o', 'w']:
            tiles[y, x] = e.get('hp', 99)
        elif type in ['x']:
            tiles[y, x] = 100 + max(0, e.get('expires', tick + 99) - tick - 1)

    layers.append(('environment HP 1', np.float32(tiles == 1), '01.0f'))
    layers.append(('environment HP 2', np.float32(tiles == 2), '01.0f'))
    layers.append(('environment HP 3', np.float32(tiles == 3), '01.0f'))
    layers.append(('environment HP 99', np.float32(tiles == 99), '01.0f'))

    # Endgame fire
    fire_time = np.maximum(np.float32(tiles) - 100, np.zeros_like(tiles)) / 100.0
    layers.append(('fire time', np.float32(fire_time), '3.1f'))

    # draw powerup positions
    for type in ['bp', 'fp']:
        layer = np.zeros([15, 15], np.float32)
        for e in entities:
            if e['type'] != type: continue
            layer[e['y'], e['x']] = 1.0
        layers.append((f'entity {type} pos', layer, '01.0f'))

    # draw bomb ranges
    layer = np.zeros([15, 15], np.float32)
    for e in entities:
        if e['type'] != 'b': continue
        y, x = e['y'], e['x']
        # Find all tiles with smaller distances from bombs than their blast ranges
        if e.get('owner_unit_id') != None:
            layer = np.float32(broadcasting_distance([y, x], layer)
                               < unit_state[e['owner_unit_id']]['blast_diameter'])
        else:
            layer = np.float32(broadcasting_distance([y, x], layer) < 3)
    layers.append((f'entity {type} pos', layer, '01.0f'))

    # how long will that bomb or fire still remain?
    for type in ['b', 'x']:
        layer = np.zeros([15, 15], np.float32)
        for e in entities:
            if e['type'] != type: continue
            layer[e['y'], e['x']] = float(e.get('expires', 9999) > tick + 1)
        layers.append((f'entity {type} remain', layer, '01.0f'))

    # how long until that bomb expires?
    for type in ['b']:
        layer = np.zeros([15, 15], np.float32)
        for e in entities:
            if e['type'] != type: continue
            if 'expires' not in e: continue
            layer[e['y'], e['x']] = float(e['expires'] - tick) / 40.0
        layers.append((f'entity {type} expires', layer, '3.1f'))

    # we need to specify where the game world ends because we will crop it to be relative to the unit
    layers.append(('world', np.ones([15, 15], np.float32), '01.0f'))

    # crop our observations to be relative to the unit
    cx, cy = unit_state[my_unit_id]['coordinates']
    view = 7
    sx, ex = max(0, cx - view), min(cx + view, 15) + 1
    sy, ey = max(0, cy - view), min(cy + view, 15) + 1
    layers = [(k, v[sy:ey, sx:ex], f) for (k, v, f) in layers]
    sx, ex = max(0, view - cx), max(0, cx - view)
    sy, ey = max(0, view - cy), max(0, cy - view)
    layers = [(k, np.pad(v, [(sy, ey), (sx, ex)]), f) for (k, v, f) in layers]
    plt.rcParams["figure.figsize"] = [18, 25]
    # Build plot out of observation
    for i, (n, l, f) in enumerate(layers):
        plt.subplot(6, 5, 1 + i)
        plt.title(n)
        plt.imshow(l, cmap='gray')
    plt.show()

    return torch.tensor(np.array([v for n, v, f in layers], np.float32)).flatten()
