from simulation.entities.entity import Entity, EntityType


class Hunter(Entity):
    def __init__(self, id: int, location: tuple[int, int]=(0, 0)):
        super().__init__(id, EntityType.HUNTER, location, "Hunter", eats=[EntityType.PREY], color="red")
