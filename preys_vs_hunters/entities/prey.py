from preys_vs_hunters.entities.entity import Entity, EntityType


class Prey(Entity):
    def __init__(self, id: int, location: tuple[int, int] = (0, 0)):
        super().__init__(id, EntityType.PREY, location, "Prey", color="blue")
