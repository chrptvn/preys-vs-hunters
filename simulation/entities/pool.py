from simulation.entities.entity import EntityType


class Pool:
    """
    Manages a collection of entities in the simulation, assigning unique IDs
    and supporting creation, removal, and reset operations.
    """

    def __init__(self):
        self.entities = []  # List of all entities (hunters, preys, walls, etc.)
        self.next_id = 0  # Auto-incrementing ID for new entities

    def add(self, type: EntityType, location: tuple[int, int]):
        """
        Create and add a new entity of the specified type at the given location.

        :param type: The type of entity to create (HUNTER, PREY, WALL)
        :param location: A tuple (x, y) representing the entity's position
        """
        id = self.next_id
        self.next_id += 1

        # Import class dynamically to avoid circular imports
        if type == EntityType.HUNTER:
            from simulation.entities.hunter import Hunter
            entity = Hunter(id, location)
        elif type == EntityType.PREY:
            from simulation.entities.prey import Prey
            entity = Prey(id, location)
        elif type == EntityType.WALL:
            from simulation.entities.wall import Wall
            entity = Wall(id, location)
        else:
            raise ValueError(f"Unknown entity type: {type}")

        self.entities.append(entity)

    def remove(self, id: int):
        """
        Remove an entity by its ID.

        :param id: The unique identifier of the entity to remove
        """
        self.entities = [e for e in self.entities if e.id != id]

    def clear(self):
        """
        Remove all entities and reset the ID counter.
        """
        self.entities = []
        self.next_id = 0
