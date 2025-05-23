from preys_vs_hunters.entities.entity import Entity, EntityType


class Wall(Entity):
    def __init__(self, id: int, location: tuple[int, int]):
        super().__init__(id, EntityType.WALL, location, "Wall", color="white")
