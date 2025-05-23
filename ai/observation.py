from typing import List

import torch

from preys_vs_hunters.entities.entity import Movement, EntityType, Entity

def entities_to_tensor(entities: List[Entity], main_entity: Entity, grid_size: int = 11) -> torch.Tensor:
    tensor = torch.zeros((4, grid_size, grid_size), dtype=torch.float32)
    center = grid_size // 2

    for entity in entities:
        rel_x = entity.location[0] - main_entity.location[0]
        rel_y = entity.location[1] - main_entity.location[1]
        gx = center + rel_x
        gy = center + rel_y

        if not (0 <= gx < grid_size and 0 <= gy < grid_size):
            continue  # out of local view

        if entity.type == EntityType.HUNTER:
            if entity.id == main_entity.id:
                tensor[0, gy, gx] = 1.0  # main hunter at center
            else:
                tensor[3, gy, gx] = 1.0
        elif entity.type == EntityType.PREY:
            tensor[1, gy, gx] = 1.0
        elif entity.type == EntityType.WALL:
            tensor[2, gy, gx] = 1.0

    return tensor


def movement_to_onehot(movement: Movement) -> torch.Tensor:
    onehot = torch.zeros(8)
    onehot[movement.value] = 1.0
    return onehot
