from numpy import ndarray
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from typing import List, Optional

from preys_vs_hunters.entities.entity import Entity, EntityType


def get_entities_at_location(location: tuple[int, int], entities: List[Entity]):
    return [entity for entity in entities if entity.location == location]


def get_local_observation(location: tuple[int, int], grid: ndarray, entities: List[Entity], size: int = 11):
    half = size // 2
    rows, cols = grid.shape
    x, y = location
    observed_entities = []

    for dx in range(-half, half + 1):
        for dy in range(-half, half + 1):
            xi = (x + dx) % rows
            yi = (y + dy) % cols
            observed_entities.extend(get_entities_at_location((xi, yi), entities))

    return observed_entities


from typing import List, Tuple, Optional


def get_entities_by_distance(
        entities: List[Entity],
        start_id: int,
        target_types: List[EntityType]
) -> List[Tuple[Entity, int]]:
    # Find the start entity
    start_entity = next((e for e in entities if e.id == start_id), None)
    if not start_entity:
        raise ValueError(f"No entity with id {start_id}")

    start_pos = start_entity.location

    # Determine grid size
    max_x = max(e.location[0] for e in entities)
    max_y = max(e.location[1] for e in entities)
    width, height = max_x + 1, max_y + 1

    # Create walkable matrix
    matrix = [[1 for _ in range(width)] for _ in range(height)]

    # Mark walls
    for e in entities:
        if e.type == EntityType.WALL:
            x, y = e.location
            matrix[y][x] = 0

    results: List[Tuple[Entity, int]] = []

    for entity in entities:
        if entity.id == start_id:
            continue
        if entity.type not in target_types:
            continue

        end_pos = entity.location
        grid = Grid(matrix=matrix)
        start_node = grid.node(*start_pos)
        end_node = grid.node(*end_pos)

        finder = AStarFinder()
        path, _ = finder.find_path(start_node, end_node, grid)

        if path:
            distance = len(path) - 1  # Number of steps
            results.append((entity, distance))

    return results


def calculate_distance_reward(
    old_distances: list[tuple["Entity", int]],
    new_distances: list[tuple["Entity", int]],
) -> float:
    reward = 0.0

    # Build dicts: {entity.id: distance}
    old_dict = {e.id: d for e, d in old_distances}
    new_dict = {e.id: d for e, d in new_distances}

    seen_ids = set(old_dict.keys()).union(new_dict.keys())

    for entity_id in seen_ids:
        old_d = old_dict.get(entity_id)
        new_d = new_dict.get(entity_id)

        if old_d is not None and new_d is not None:
            delta = old_d - new_d
            reward += delta * 0.2  # Encourage closeness
        elif old_d is None and new_d is not None:
            reward += 0.5  # newly seen
        elif old_d is not None and new_d is None:
            reward -= 0.5  # lost sight

    return reward