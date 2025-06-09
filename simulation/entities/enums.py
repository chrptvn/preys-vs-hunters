from enum import Enum


class EntityType(Enum):
    PREY = 0
    HUNTER = 1
    WALL = 2


class Movement(Enum):
    UP = 0
    UPPER_RIGHT = 1
    RIGHT = 2
    LOWER_RIGHT = 3
    DOWN = 4
    LOWER_LEFT = 5
    LEFT = 6
    UPPER_LEFT = 7
    STAY = 8