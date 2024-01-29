import enum
import typing as t

Coordinate = t.Union[int, int]
Map = t.List[t.List[str]]
Observation = t.Dict
State = t.List[t.List[int]]

class Unit(enum.IntEnum):
    WoodenBlock = 0
    OreBlock = 1
    MetalBlock = 2
    FreezePowerup = 3
    BlastPowerup = 4
    Blast = 5
    Bomb = 6
    Ammunition = 7
    Friend = 8
    Enemy = 9
