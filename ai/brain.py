from typing import List, Dict

import torch

from ai.models import Model
from ai.observation import entities_to_tensor, movement_to_onehot
from preys_vs_hunters.entities.entity import EntityType, Entity, Movement
from preys_vs_hunters.utils import get_entities_by_distance

def decide_movement(entity_id: int, entities: list, model: Model, device) -> Movement:
    model.load_checkpoint("path_finder")

    entity = next((e for e in entities if e.id == entity_id), None)
    # Build input grid
    grid_tensor = entities_to_tensor(entities, main_entity=entity, grid_size=11).unsqueeze(0).to(device)
    # Test each movement
    best_move = None
    best_reward = float('-inf')
    for movement in Movement:
        movement_tensor = movement_to_onehot(movement).unsqueeze(0).to(device)

        with torch.no_grad():
            predicted_reward = model(grid_tensor, movement_tensor).item()

        if predicted_reward > best_reward:
            best_reward = predicted_reward
            best_move = movement
    return best_move

def calculate_target_selector_reward(
    entity: Entity,
    entities: List[Entity],
    priorities: Dict[EntityType, int],
    target: Entity
) -> float:
    """
    Calculates a reward for selecting a given target entity.

    The reward is based on:
    - Distance to the target (closer is better)
    - Priority assigned to the target's type

    :param entity: The hunter (or agent) making the decision.
    :param entities: All visible entities in the environment.
    :param priorities: A mapping of EntityType to priority value.
    :param target: The target entity being evaluated.
    :return: A float reward value.
    """
    # Get all entities of interest by type
    distances = get_entities_by_distance(entities, entity.id, list(priorities.keys()))

    # Find the distance to the selected target
    distance = next((dist for ent, dist in distances if ent.id == target.id), None)

    if distance is not None:
        priority = priorities.get(target.type, 0)

        # Option 1: Inverse distance reward (simple & stable)
        reward = (1 / (distance + 1)) * priority

        # Optional: use exponential shaping for stronger preference to close targets
        # reward = priority * math.exp(-distance)

        return reward

    # Target not visible or not relevant — penalize selection slightly
    return -1.0


def compute_reward(hunter: Entity, prey_list: List[Entity], entities: List[Entity], movement: Movement) -> float:
    # Save original position
    original_position = hunter.location

    # Save prey positions
    prey_positions = {prey.id: prey.location for prey in prey_list}

    # Get distances before moving
    before = get_entities_by_distance(entities, hunter.id, [EntityType.PREY])
    before_dict = {entity.id: dist for entity, dist in before}

    # Move hunter
    hunter.move(movement)

    # Get distances after moving
    after = get_entities_by_distance(entities, hunter.id, [EntityType.PREY])
    after_dict = {entity.id: dist for entity, dist in after}

    # Calculate reward
    reward = 0.0
    for prey_id, before_dist in before_dict.items():
        after_dist = after_dict.get(prey_id)
        if after_dist is not None:
            reward += (before_dist - after_dist) * (10 - after_dist)

    # Restore original positions (non-destructive simulation)
    hunter.move_to(original_position[0], original_position[1])
    for prey in prey_list:
        prey.move_to(prey_positions[prey.id][1], prey_positions[prey.id][0])

    return reward