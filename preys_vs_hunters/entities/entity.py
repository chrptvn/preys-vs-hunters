from typing import Optional, List

from ai.brain import EntityType, Movement


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
        self.target: Optional[Entity] = None

    def move_to(self, x: int, y: int):
        self.location = (x, y)

    def move(self, movement: Movement, size: tuple[int, int], entities: []):
        x, y = self.location
        max_x, max_y = size

        if movement == Movement.UP:
            x_new, y_new = x, y - 1
        elif movement == Movement.UPPER_RIGHT:
            x_new, y_new = x + 1, y - 1
        elif movement == Movement.RIGHT:
            x_new, y_new = x + 1, y
        elif movement == Movement.LOWER_RIGHT:
            x_new, y_new = x + 1, y + 1
        elif movement == Movement.DOWN:
            x_new, y_new = x, y + 1
        elif movement == Movement.LOWER_LEFT:
            x_new, y_new = x - 1, y + 1
        elif movement == Movement.LEFT:
            x_new, y_new = x - 1, y
        elif movement == Movement.UPPER_LEFT:
            x_new, y_new = x - 1, y - 1
        else:
            x_new, y_new = x, y  # No movement

        # Apply toroidal wrapping
        x_new %= max_x
        y_new %= max_y

        if entities:
            # Check for collisions with other entities
            for entity in entities:
                if entity.id != self.id and entity.location == (x_new, y_new):
                    return

        self.move_to(x_new, y_new)

