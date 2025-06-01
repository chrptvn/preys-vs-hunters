import os

from ai.models import ActionType
from ai.tensor.DataBuilder import Entity, DataBuilder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
import random
from preys_vs_hunters.entities.entity import EntityType
from preys_vs_hunters.goal.Goal import Escape, Chase, Stay


# --- DATASET ---
def generate_training_sample(grid_size=(11, 11), max_entities=6):
    entity_id = 0
    observer = Entity(entity_id, EntityType.PREY, x=5, y=5)
    builder = DataBuilder(observer=observer, grid_size=grid_size)

    entity_id += 1
    entities = []

    for _ in range(max_entities):
        ex, ey = random.randint(0, 10), random.randint(0, 10)
        if (ex, ey) == (observer.x, observer.y): continue
        etype = random.choice([EntityType.PREDATOR, EntityType.PREY])
        e = Entity(entity_id, etype, ex, ey)
        builder.add_entity(e)
        entities.append(e)
        entity_id += 1

    predator_entities = [e for e in entities if e.type == EntityType.PREDATOR]
    prey_entities = [e for e in entities if e.type == EntityType.PREY]

    if predator_entities:
        target = min(predator_entities, key=lambda e: abs(e.x - observer.x) + abs(e.y - observer.y))
        goal = Escape(target.id)
        action = ActionType.ESCAPE.value
    elif prey_entities:
        target = min(prey_entities, key=lambda e: abs(e.x - observer.x) + abs(e.y - observer.y))
        goal = Chase(target.id)
        action = ActionType.FOLLOW.value
    else:
        goal = Stay()
        target = observer
        action = ActionType.IDLE.value

    builder.add_goal(goal)
    data = builder.build()

    data.y_action = torch.tensor(action, dtype=torch.long)
    data.y_goal = torch.tensor(goal.type.value, dtype=torch.long)
    data.y_target_id = torch.tensor(builder.get_index_by_entity_id(target.id), dtype=torch.long)
    data.y_move_target = torch.tensor([target.x - observer.x, target.y - observer.y], dtype=torch.float32)

    return data

class PreyVsHunterDataset(Dataset):
    def __init__(self, num_samples=500):
        self.samples = [generate_training_sample() for _ in range(num_samples)]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


# --- MODEL ---
class HunterModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, max_target_entities=20):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.pre_head = nn.Linear(hidden_dim, hidden_dim)

        self.heads = nn.ModuleDict({
            "action": nn.Linear(hidden_dim, len(ActionType)),
            "move": nn.Linear(hidden_dim, 2),
            "goal": nn.Linear(hidden_dim, 3),
            "target": nn.Linear(hidden_dim, max_target_entities)
        })

    def forward(self, x, edge_index, node_index: int = 0):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        node = F.relu(self.pre_head(x[node_index]))
        return (
            self.heads["action"](node),
            self.heads["move"](node),
            self.heads["goal"](node),
            self.heads["target"](node)
        )

# --- TRAINING LOOP ---
def train_model(model, dataloader, optimizer, device, max_epoch=1000):
    model.train()
    loss_action_fn = nn.CrossEntropyLoss()
    loss_goal_fn = nn.CrossEntropyLoss()
    loss_target_fn = nn.CrossEntropyLoss()
    loss_move_fn = nn.MSELoss()

    for epoch in range(max_epoch):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out_action, out_move, out_goal, out_target = model(batch.x, batch.edge_index)

            y_action = batch.y_action.to(device)
            y_goal = batch.y_goal.to(device)
            y_target_id = batch.y_target_id.to(device)
            y_move = batch.y_move_target.to(device)

            loss = (
                    loss_action_fn(out_action.unsqueeze(0), y_action) +
                    loss_goal_fn(out_goal.unsqueeze(0), y_goal) +
                    loss_target_fn(out_target.unsqueeze(0), y_target_id) +
                    loss_move_fn(out_move, y_move)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch+1}] Loss: {loss:.4f}")

        if epoch == max_epoch or loss < 0.01:
            break

    return model



# --- RUN ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PreyVsHunterDataset(num_samples=200)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = HunterModel().to(device)
    model = nn.DataParallel(model)

    model_path = "preys_vs_hunters.pt"

    if os.path.exists(model_path):
        print("📦 Found existing model, loading weights and continuing training...")
        model.module.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("🚀 No existing model found. Starting training from scratch...")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trained_model = train_model(model, loader, optimizer, device, max_epoch=1000)

    torch.save(trained_model.module.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")