from ai.brain import compute_reward
from ai.observation import entities_to_tensor, movement_to_onehot
from ai.models import HunterModel
from preys_vs_hunters.entities.entity import Movement
from preys_vs_hunters.entities.hunter import Hunter
from preys_vs_hunters.entities.prey import Prey

import torch

if __name__ == '__main__':
    hunter = Hunter(0, (5, 5))
    prey = Prey(1)
    entities = [hunter, prey]

    model = HunterModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.SmoothL1Loss()

    for epoch in range(2000):
        grid_batch = []
        move_batch = []
        reward_batch = []

        for x in range(11):
            for y in range(11):
                if (x, y) == (5, 5): continue
                prey.move_to(x, y)

                for movement in Movement:
                    reward = compute_reward(hunter, [prey], entities, movement)

                    grid_tensor = entities_to_tensor(entities, hunter, grid_size=11)
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

    torch.save({
        "path_finder": model.state_dict()
    }, "hunter.pt")