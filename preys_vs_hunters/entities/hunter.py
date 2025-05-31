from preys_vs_hunters.entities.entity import Entity, EntityType
from pathfinding.core.grid import Grid as PFGrid
from pathfinding.finder.a_star import AStarFinder
from typing import List


class Hunter(Entity):
    def __init__(self, id: int, location: tuple[int, int]=(0, 0)):
        super().__init__(id, EntityType.PREDATOR, location, "Hunter", eats=[EntityType.PREY], color="red")

    def calculate_reward(
            self,
            previous_entities: list[Entity],
            current_entities: list[Entity],
    ):
        return 1
