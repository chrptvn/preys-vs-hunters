import torch
from torch_geometric.data import Data

from simulation.entities.entity import Entity
from simulation.entities.enums import Movement, EntityType


def get_action_from_delta(relative_location: tuple[int, int], is_fleeing: bool):
    """
    Given a relative (dx, dy) position, determine the most appropriate movement.
    If is_fleeing is True, return the opposite direction.

    :param relative_location: Tuple of (dx, dy)
    :param is_fleeing: Whether to invert direction
    :return: Movement enum indicating direction to move
    """
    dx, dy = relative_location
    if dx == 0 and dy == 0:
        return Movement.STAY

    # Choose the dominant direction
    if abs(dx) > abs(dy):
        action = Movement.LEFT if dx > 0 else Movement.RIGHT
    elif abs(dy) > abs(dx):
        action = Movement.UP if dy > 0 else Movement.DOWN
    else:
        # Diagonal movement preference when dx == dy
        if dx > 0 and dy > 0:
            action = Movement.UPPER_LEFT
        elif dx < 0 < dy:
            action = Movement.UPPER_RIGHT
        elif dx > 0 > dy:
            action = Movement.LOWER_LEFT
        else:
            action = Movement.LOWER_RIGHT

    if is_fleeing:
        # Reverse the movement direction
        opposite_map = {
            Movement.UP: Movement.DOWN,
            Movement.UPPER_RIGHT: Movement.LOWER_LEFT,
            Movement.RIGHT: Movement.LEFT,
            Movement.LOWER_RIGHT: Movement.UPPER_LEFT,
            Movement.DOWN: Movement.UP,
            Movement.LOWER_LEFT: Movement.UPPER_RIGHT,
            Movement.LEFT: Movement.RIGHT,
            Movement.UPPER_LEFT: Movement.LOWER_RIGHT,
            Movement.STAY: Movement.STAY
        }
        action = opposite_map[action]

    return action


def toroidal_delta(a, b, size: int):
    """
    Compute shortest signed distance between two points in a toroidal space.

    :param a: Position A
    :param b: Position B
    :param size: Grid size in one dimension
    :return: Signed delta considering wrap-around
    """
    delta = (a - b + size) % size
    if delta > size // 2:
        delta -= size
    return delta


def get_relative_position(observer: Entity, entity: Entity, grid_size: tuple[int, int]):
    """
    Compute the toroidal relative (dx, dy) from observer to another entity.

    :param observer: The entity doing the observing
    :param entity: The other entity
    :param grid_size: (width, height) of the grid
    :return: Tuple of (dx, dy)
    """
    dx = toroidal_delta(observer.location[0], entity.location[0], grid_size[0])
    dy = toroidal_delta(observer.location[1], entity.location[1], grid_size[1])
    return dx, dy


def get_relative_distance(observer: Entity, entity: Entity, grid_size: tuple[int, int]):
    """
    Compute the Euclidean distance from observer to entity using toroidal coordinates.

    :param observer: The observer entity
    :param entity: The target entity
    :param grid_size: Grid dimensions
    :return: Euclidean distance (float)
    """
    dx, dy = get_relative_position(observer, entity, grid_size)
    return (dx ** 2 + dy ** 2) ** 0.5


def get_nearest_entity(observer: Entity, entities: list[Entity], grid_size: tuple[int, int], entity_type: EntityType):
    """
    Return the nearest entity of a given type, relative to the observer.

    :param observer: The observer entity
    :param entities: List of all entities
    :param grid_size: Grid dimensions
    :param entity_type: Type of entity to look for
    :return: Closest entity of the specified type, or None
    """
    nearest = None
    min_distance = float('inf')

    for entity in entities:
        if entity.type != entity_type:
            continue
        dx = toroidal_delta(entity.location[0], observer.location[0], grid_size[0])
        dy = toroidal_delta(entity.location[1], observer.location[1], grid_size[1])
        distance = (dx ** 2 + dy ** 2) ** 0.5

        if distance < min_distance:
            min_distance = distance
            nearest = entity

    return nearest


def get_entities_at_location(location: tuple[int, int], entities: list[Entity]):
    """
    Return all entities located at a specific grid cell.

    :param location: (x, y) coordinates
    :param entities: List of all entities
    :return: List of entities at the given location
    """
    return [entity for entity in entities if entity.location == location]


def generate_data(observer: Entity, entities: list[Entity], grid_size: tuple[int, int]):
    """
    Generate input features and edge index for GNN model.

    Features per entity:
    - Normalized index
    - Observer type
    - Entity type
    - Normalized distance in x
    - Normalized distance in y

    Edges are self-loops only (can be extended).

    :param observer: Observer entity
    :param entities: List of entities in the environment
    :param grid_size: (width, height) of the grid
    :return: Tuple of (entity ID → feature index mapping, PyG Data object)
    """
    rows = []
    edges = []
    entities_indexes = {}

    for i, entity in enumerate(entities):
        normalized_dist_x = toroidal_delta(entity.location[0], observer.location[0], grid_size[0]) / grid_size[0]
        normalized_dist_y = toroidal_delta(entity.location[1], observer.location[1], grid_size[1]) / grid_size[1]
        norm_idx = i / len(entities)

        # Build node feature vector
        rows.append([
            norm_idx,
            observer.type.value,
            entity.type.value,
            normalized_dist_x,
            normalized_dist_y
        ])

        # Map entity ID to its index in the feature list
        entity_index = len(rows) - 1
        entities_indexes[entity.id] = entity_index

        # Create a self-loop edge (can add more edges later)
        edges.append([entity_index, entity_index])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(rows, dtype=torch.float32)

    return entities_indexes, Data(x=x, edge_index=edge_index)
