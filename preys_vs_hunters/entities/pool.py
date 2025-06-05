from preys_vs_hunters.entities.entity import EntityType


class Pool:
    def __init__(self):
        self.entities = []
        self.next_id = 0

    def add(self, type: EntityType, location: tuple[int, int]):
        id = self.next_id
        self.next_id += 1
        if type == EntityType.HUNTER:
            from preys_vs_hunters.entities.hunter import Hunter
            entity = Hunter(id, location)
        elif type == EntityType.PREY:
            from preys_vs_hunters.entities.prey import Prey
            entity = Prey(id, location)
        elif type == EntityType.WALL:
            from preys_vs_hunters.entities.wall import Wall
            entity = Wall(id, location)
        else:
            raise ValueError(f"Unknown entity type: {type}")
        self.entities.append(entity)

    def remove(self, id: int):
        self.entities = [e for e in self.entities if e.id != id]

    def clear(self):
        self.entities = []
        self.next_id = 0