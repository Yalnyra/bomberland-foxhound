from components.types import Coordinate

def manhattan_distance(coord1: Coordinate, coord2: Coordinate):
    x1, y1 = coord1
    x2, y2 = coord2
    return abs(x2 - x1) + abs(y2 - y1)
