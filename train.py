from typing import List

import torch
from torch import nn

from ai.RewardPredictor import RewardPredictor
from preys_vs_hunters.entities.entity import EntityType, Movement, Entity
from preys_vs_hunters.entities.hunter import Hunter
from preys_vs_hunters.entities.prey import Prey
from preys_vs_hunters.utils import get_entities_by_distance

def compute_reward(hunter: Entity, prey_list: List[Entity], entities: List[Entity], movement: Movement) -> float:
    """
    Computes the reward for a hunter performing a movement,
    based on change in distance to all preys.

    :param hunter: The hunter entity (must support .move and .move_to).
    :param prey_list: The list of prey entities (must be part of entities).
    :param entities: All entities in the environment.
    :param movement: The movement direction to test.
    :return: A float representing the cumulative reward.
    """
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
            reward += before_dist - after_dist

    # Restore original positions (non-destructive simulation)
    hunter.move_to(original_position[0], original_position[1])
    for prey in prey_list:
        prey.move_to(prey_positions[prey.id][1], prey_positions[prey.id][0])

    return reward * 10  # Scale reward for better learning

def entities_to_tensor(entities: List[Entity], grid_size: int = 11, main_hunter_id: int = 0) -> torch.Tensor:
    tensor = torch.zeros((4, grid_size, grid_size), dtype=torch.float32)

    for entity in entities:
        x, y = entity.location
        if entity.type == EntityType.HUNTER:
            if entity.id == main_hunter_id:
                tensor[0, y, x] = 1.0  # main hunter
            else:
                tensor[3, y, x] = 1.0  # other hunters
        elif entity.type == EntityType.PREY:
            tensor[1, y, x] = 1.0
        elif entity.type == EntityType.WALL:
            tensor[2, y, x] = 1.0

    return tensor


def movement_to_onehot(movement: Movement) -> torch.Tensor:
    onehot = torch.zeros(8)
    onehot[movement.value] = 1.0
    return onehot


import torch
from torch.utils.data import TensorDataset, DataLoader

if __name__ == '__main__':
    hunter = Hunter(0, (5, 5))
    prey = Prey(1)
    entities = [hunter, prey]

    model = RewardPredictor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.SmoothL1Loss()

    for epoch in range(700):
        grid_batch = []
        move_batch = []
        reward_batch = []

        for x in range(11):
            for y in range(11):
                if (x, y) == (5, 5): continue
                prey.move_to(x, y)

                for movement in Movement:
                    reward = compute_reward(hunter, [prey], entities, movement)

                    grid_tensor = entities_to_tensor(entities, grid_size=11, main_hunter_id=hunter.id)
                    move_tensor = movement_to_onehot(movement)
                    reward_tensor = torch.tensor([reward], dtype=torch.float32)

                    grid_batch.append(grid_tensor)
                    move_batch.append(move_tensor)
                    reward_batch.append(reward_tensor)

        # Stack into full batch
        grid_batch = torch.stack(grid_batch)         # (B, 4, 11, 11)
        move_batch = torch.stack(move_batch)         # (B, 8)
        reward_batch = torch.stack(reward_batch)     # (B, 1)

        # Move to GPU if available
        grid_batch = grid_batch.to(device)
        move_batch = move_batch.to(device)
        reward_batch = reward_batch.to(device)

        # Forward + backward
        pred = model(grid_batch, move_batch)         # (B, 1)
        loss = loss_fn(pred, reward_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "reward_predictor.pt")