import random
import torch
import torch.nn as nn

from ai.model import SwarmBrain
from simulation.entities.enums import EntityType, Movement
from simulation.entities.hunter import Hunter
from simulation.entities.prey import Prey
from simulation.entities.wall import Wall
from simulation.utils import (
    get_nearest_entity,
    get_relative_distance,
    get_relative_position,
    get_action_from_delta,
    generate_data, get_normalized_distance
)


def train(epochs: int, n_entities: int):
    """
    Train the SwarmBrain model using randomly generated prey/hunter interactions.

    :param epochs: Number of training iterations
    :param n_entities: Number of target entities to include per sample
    """
    # Loss weight multipliers
    attention_loss_mult = 100.0
    target_location_loss_mult = 1.0
    action_loss_mult = 50.0

    for epoch in range(epochs):
        # Alternate between Hunter and Prey as observer
        if epoch % 2:
            observer = Hunter(0, (int(grid_size[0] / 2), int(grid_size[1] / 2)))
        else:
            observer = Prey(0, (int(grid_size[0] / 2), int(grid_size[1] / 2)))

        entities = []
        distance_targets = []

        # Generate n_entities of the opposite type randomly placed
        for i in range(n_entities):
            entity = Prey(i, (random.randint(0, grid_size[0] - 1), random.randint(0, grid_size[1] - 1)))
            if observer.type == EntityType.HUNTER:
                relative_distance = get_relative_distance(observer, entity, grid_size)
                normalized_distance = get_normalized_distance(relative_distance, grid_size)
                distance_targets.append(normalized_distance)
            else:
                distance_targets.append(1.0)
            entities.append(entity)

            entity = Hunter(i, (random.randint(0, grid_size[0] - 1), random.randint(0, grid_size[1] - 1)))
            if observer.type == EntityType.PREY:
                relative_distance = get_relative_distance(observer, entity, grid_size)
                normalized_distance = get_normalized_distance(relative_distance, grid_size)
                distance_targets.append(normalized_distance)
            else:
                distance_targets.append(1.0)
            entities.append(entity)

            entity = Wall(i, (random.randint(0, grid_size[0] - 1), random.randint(0, grid_size[1] - 1)))
            distance_targets.append(1.0)
            entities.append(entity)

        # Generate input graph and index mapping
        entities_indexes, data = generate_data(observer, entities, grid_size)
        data = data.to(device)

        # Determine type of entity to chase
        target_type = EntityType.PREY if observer.type == EntityType.HUNTER else EntityType.HUNTER
        nearest_entity = get_nearest_entity(observer, entities, grid_size, target_type)

        nearest_entity_position = get_relative_position(observer, nearest_entity, grid_size)
        action_to_nearest_entity = get_action_from_delta(nearest_entity_position, observer.type == EntityType.PREY)

        # Target tensors for supervision
        attention_targets = torch.tensor(entities_indexes[nearest_entity.id], dtype=torch.long, device=device)
        distance_targets = torch.tensor(distance_targets, dtype=torch.float32, device=device)
        target_location_target = torch.tensor(
            [nearest_entity_position[0], nearest_entity_position[1]],
            dtype=torch.float32,
            device=device
        )
        action_target = torch.tensor(action_to_nearest_entity.value, dtype=torch.long, device=device)

        # Forward pass
        distance_scores, attention_scores, target_location_score, action_logits = model(data)

        # Compute losses
        distance_loss = mse_loss_fn(distance_scores, distance_targets)
        attention_loss = cross_entropy_loss_fn(attention_scores, attention_targets)
        target_location_loss = mse_loss_fn(target_location_score, target_location_target)
        action_loss = cross_entropy_loss_fn(action_logits, action_target)

        # Combine all loss components
        total_loss = (
            distance_loss +
            attention_loss_mult * attention_loss +
            target_location_loss_mult * target_location_loss +
            action_loss_mult * action_loss
        )

        # Print debug info every 101 epochs
        if epoch % 100 == 0:
            print("Target entity:", attention_targets.item())
            print("Predicted entity:", torch.argmax(attention_scores).item())

            print("Target distances:   ", distance_targets.squeeze().tolist())
            print("Predicted distances:", distance_scores.squeeze().tolist())

            print("Target relative location:   ", [nearest_entity_position[0], nearest_entity_position[1]])
            print("Predicted relative location:", target_location_score.tolist())

            print("Target action:", Movement(action_to_nearest_entity.value))
            print("Predicted action:", Movement(torch.argmax(action_logits).item()))

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss.item()}")

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


if __name__ == '__main__':
    # Setup training environment
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SwarmBrain().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    mse_loss_fn = nn.MSELoss()
    cross_entropy_loss_fn = nn.CrossEntropyLoss()
    grid_size = (50, 50)
    epochs = 50000

    # Curriculum training with increasing complexity
    train(epochs, 1)
    train(epochs, 2)
    train(epochs, 3)
    train(epochs, 5)
    train(epochs * 5, 8)

    # Save trained model
    torch.save(model.state_dict(), "swarm_brain.pt")
