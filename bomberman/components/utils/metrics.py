import numpy as np
from components.types import Coordinate
# Two points distance
def manhattan_distance(coord1: Coordinate, coord2: Coordinate):
    x1, y1 = coord1
    x2, y2 = coord2
    return abs(x2 - x1) + abs(y2 - y1)


# Fast abs distance function for a matrix/map from an origin point
def broadcasting_distance(origin: Coordinate, map):
    m, n = map.shape
    return np.sqrt((np.arange(m)[:, None] - origin[0]) ** 2 + (np.arange(n) - origin[1]) ** 2)

def engame_fire_sdf ():
    pass
