from typing import Optional
from simulation.entities.enums import EntityType, Movement


class Entity:
    def __init__(self,
                 id: int,
                 type: EntityType,
                 location: tuple[int, int],
                 label: str,
                 eats: [EntityType] = None,
                 fears: [EntityType] = None,
                 color: str = "black"
                 ):
        self.id = id  # Unique identifier for the entity
        self.type = type  # Type of entity (e.g., HUNTER, PREY, WALL)
        self.location = location  # (x, y) position on the grid
        self.label = label  # Display label or name
        self.eats = eats if eats is not None else []  # List of EntityTypes this entity can eat
        self.fears = fears if fears is not None else []  # List of EntityTypes this entity fears
        self.color = color  # Optional display color
        self.target: Optional[Entity] = None  # Optional target entity (e.g., for chasing)

    def move_to(self, x: int, y: int):
        """Update the entity's location to the new (x, y) position."""
        self.location = (x, y)

    def move(self, movement: Movement, size: tuple[int, int], entities: []):
        """
        Move the entity based on the specified direction (Movement),
        wrap around the grid if necessary (toroidal behavior),
        and handle collisions/eating behavior.

        :param movement: Direction to move (from Movement enum)
        :param size: Size of the grid as (width, height)
        :param entities: List of all entities (to detect collisions)
        """
        x, y = self.location
        max_x, max_y = size

        # Determine target position based on movement direction
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
            x_new, y_new = x, y  # Movement.NONE → stay in place

        # Apply toroidal wrapping (grid edges wrap around)
        x_new %= max_x
        y_new %= max_y

        if entities:
            # Collision detection
            for entity in entities:
                if entity.id != self.id and entity.location == (x_new, y_new):
                    if entity.type in self.eats:
                        # If the entity is edible, consume it
                        print(
                            f"{self.label} (ID: {self.id}) eats {entity.label} (ID: {entity.id}) at ({x_new}, {y_new})")
                        entities.remove(entity)
                    else:
                        # Blocked by non-edible entity — cancel movement
                        return

        # Perform the move if no collision or if edible
        self.move_to(x_new, y_new)
