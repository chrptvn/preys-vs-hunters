from enum import Enum


class EntityType(Enum):
    VOID = 0
    HUNTER = 1
    PREY = 2
    WALL = 3


class Movement(Enum):
    UP = 0
    UPPER_RIGHT = 1
    RIGHT = 2
    LOWER_RIGHT = 3
    DOWN = 4
    LOWER_LEFT = 5
    LEFT = 6
    UPPER_LEFT = 7


class Entity:
    def __init__(self,
                 id: int,
                 type: EntityType,
                 location: tuple[int, int],
                 label: str,
                 eats: [EntityType] = None,
                 color: str = "black"
                 ):
        self.id = id
        self.type = type
        self.location = location
        self.label = label
        if eats is None:
            self.eats = []
        else:
            self.eats = eats
        self.color = color

    def move_to(self, x: int, y: int):
        self.location = (x, y)

    def move(self, movement: Movement):
        x, y = self.location
        if movement == Movement.UP:
            self.move_to(x, y - 1)
        elif movement == Movement.UPPER_RIGHT:
            self.move_to(x + 1, y - 1)
        elif movement == Movement.RIGHT:
            self.move_to(x + 1, y)
        elif movement == Movement.LOWER_RIGHT:
            self.move_to(x + 1, y + 1)
        elif movement == Movement.DOWN:
            self.move_to(x, y + 1)
        elif movement == Movement.LOWER_LEFT:
            self.move_to(x - 1, y + 1)
        elif movement == Movement.LEFT:
            self.move_to(x - 1, y)
        elif movement == Movement.UPPER_LEFT:
            self.move_to(x - 1, y - 1)