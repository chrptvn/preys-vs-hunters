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

        if entity.type == EntityType.PREDATOR:
            if entity.id == main_entity.id:
                tensor[0, gy, gx] = 1.0  # main hunter at center
            else:
                tensor[3, gy, gx] = 1.0
        elif entity.type == EntityType.PREY:
            tensor[1, gy, gx] = 1.0
        elif entity.type == EntityType.WALL:
            tensor[2, gy, gx] = 1.0

    return tensor


def create_selection_tensor(selected: Entity, preys: list[Entity]) -> torch.Tensor:
    tensor = torch.zeros((len(preys),), dtype=torch.float32)
    index = preys.index(selected)
    tensor[index] = 1.0
    return tensor


def movement_to_onehot(movement: Movement) -> torch.Tensor:
    onehot = torch.zeros(8)
    onehot[movement.value] = 1.0
    return onehot


def get_relative_location(entity: Entity, observer: Entity, grid_size: tuple[int, int]) -> tuple[int, int]:
    ex, ey = entity.location
    tx, ty = observer.location

    dx = ex - tx
    dy = ey - ty

    grid_width, grid_height = grid_size

    # Handle horizontal wrapping
    if abs(dx) > grid_width // 2:
        if dx > 0:
            dx -= grid_width
        else:
            dx += grid_width

    # Handle vertical wrapping
    if abs(dy) > grid_height // 2:
        if dy > 0:
            dy -= grid_height
        else:
            dy += grid_height

    return dx, dy


def get_normalized_distance(relative_location: tuple[int, int], grid_size: tuple[int, int]):
    dx, dy = relative_location
    distance = (dx ** 2 + dy ** 2) ** 0.5
    # Exact max distance in the grid
    max_distance = ((grid_size[0] - 1) ** 2 + (grid_size[1] - 1) ** 2) ** 0.5
    normalized_distance = distance / max_distance if max_distance > 0 else 0.0
    return normalized_distance